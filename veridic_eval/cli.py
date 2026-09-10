"""
veridic-eval command line.

Two commands cover every metric: `init` writes the two yamls, `run` measures.

    veridic-eval init
        One-shot setup after the in-app runs: doctor checks, write
        benchmark.yaml + conditions.yaml (from the example files, else from the
        built-in copies in templates.py; --force overwrites existing ones),
        list recent conversation ids to paste into conditions.yaml, then re-read
        that file and print what every cell declares, so a misspelled key or a
        reference pointed at an off-grid cell fails here and not mid-run.

    veridic-eval sit CELL [CELL ...]
        Point a cell at the chat that actually holds its answers. Finds the
        logged chat carrying that cell's benchmark questions, writes its real
        window and conversation id into conditions.yaml, and keeps a .bak. The
        starter file ships fabricated dates, and a window that misses the
        sitting links nothing, so this is what turns `0/10 queries linked` into
        a scored cell. `run --sit <cell>` does it and then runs.

    veridic-eval run
        Every metric in one command: snapshot each declared cell to
        out/cells/<cell>.json while its ingest is still live (an existing
        snapshot is reused untouched), score the 2x2 plus the off-grid
        before/after window, add the GBP CFCA from the declared cost inputs,
        write out/report.json + out/report.md, then re-derive every statistic
        with the post-run math check. Safe to repeat after each in-app cell:
        `--dump-only` saves the cell you just finished and stops.

    veridic-eval verify
        Post-run math check on out/report.json: Bonferroni divisor, CI ordering,
        significance flags, CFCA subtraction, off-grid isolation. Runs at the
        end of `run` as well, and on its own against any saved report.

    veridic-eval reprice [CELL ...]
        Price the dumped cells at the end instead of during the run: the same
        answer spans, re-read off the finished power log under the `power:`
        block as it stands now, rewritten into out/cells/<cell>.json. A dump
        freezes the price of the moment it was taken, so this is what carries a
        later `power:` edit (min_samples, basis, idle_w) into a cell already
        dumped. No database, no re-ingest; `run` scores the new watts.

    veridic-eval view
        Open the snapshot viewer on out/cells/*.json (Streamlit). Optional: no
        metric comes from it, `run` already produced them all.

    veridic-eval doctor
        Telemetry kill-switch status, DB reachability, the chunking policy label
        the newest index build wrote, and how much benchmark gold the live
        ingest holds before any answer is spent.

    veridic-eval cost / veridic-eval power
        CFCA cost calculator / GPU power poller; flags are
        forwarded, see `veridic-eval cost --help`. `run` already prints the CFCA
        for every cell whose conditions entry carries a `cost:` block; these two
        stay for measuring watts and for one-off arithmetic.

Conditions file (YAML/JSON), the single declaration of the experiment::

    reference: control            # the 2x2 cell every delta is measured against
    offgrid: [prebaseline]        # measured, never scored inside the 2x2
    cost_defaults: {watts: 43.7, kwh_gbp: 0.26, A: 1000.0, Q: 1000.0}
    conditions:
      prebaseline:
        start: 2026-06-30T00:00:00Z
        end:   2026-07-01T00:00:00Z
        per_answer_cost: 0.0      # USD API bill per answer, converted to GBP
        cost: {onetime_gpu_hours: 0.0, query_gpu_seconds: 2.1}
      control:    {start: 2026-07-01T00:00:00Z, end: 2026-07-02T00:00:00Z, per_answer_cost: 0.0}
      chunk_opt:  {start: 2026-07-02T00:00:00Z, end: 2026-07-03T00:00:00Z}
      qdora:      {start: 2026-07-03T00:00:00Z, end: 2026-07-04T00:00:00Z}
      combined:   {start: 2026-07-04T00:00:00Z, end: 2026-07-05T00:00:00Z}

Cell names come from veridic_eval/cells.py: control is the 2x2 reference cell,
prebaseline is the app before the one-time baseline construction. `run` scores
the four grid cells and puts prebaseline in a separate uncorrected before/after
block, so declaring it never changes the 2x2 numbers or their Bonferroni family.
Any other name works too and is scored inside the grid. Full key list and the
accepted `cost:` quantities: veridic_eval/conditions.py.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

# importing the package applies telemetry_off first
from . import telemetry_off
from .conditions import load_conditions_file
from .config import settings
from .db import ping
from .pipeline import resolve_refresh, run_experiment
from .verify import render_verify_text, verify_report_file


def say_live_chunking_policy(
    *,
    printer: Callable[[str], None] = print,
    indent: str = "  ",
    label_text: str = "chunking policy now",
    none_text: str = "no labelled build",
    **latest_kwargs: Any,
) -> Optional[str]:
    """
    Print the label the newest index build actually wrote, and return it.

    The label is the one fact that says which builder a re-ingest ran, and the env
    file cannot say it: `services/chunker.py` reads ``CHUNKING_MODE`` out of the
    process environment per ingest, so a flip that never reached the container is
    invisible until a cell is dumped. Printed here it costs one query.

    Args:
        printer / indent / label_text / none_text: how the line is written.
        latest_kwargs: `ingest_cost.latest_build_policy` knobs (``since``,
            ``as_of``, ``session``, ``read_kwargs``, ``on_error``). ``on_error``
            stays ``note``, so an unreadable table prints its reason.
    """
    from .ingest_cost import latest_build_policy

    label, note = latest_build_policy(**latest_kwargs)
    printer(f"{indent}{label_text}: {label or none_text} ({note})")
    return label


def say_gold_gate(
    *,
    benchmark_path: Optional[str] = None,
    printer: Callable[[str], None] = print,
    indent: str = "  ",
    label_text: str = "gold in live ingest",
    max_unmatched: int = 0,
    max_reasons: int = 10,
    no_file_text: str = "no benchmark file",
    no_gold_text: str = "benchmark carries no gold_evidence",
    on_error: str = "note",
    gold_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """
    Print how much of the benchmark's gold evidence the live ingest holds.

    The match runs against the chunk table, not against a dump, so this reads
    before any question is asked. That is the only cheap moment for it: gold text
    that matches nothing is scored as a retrieval miss, and a fix found after a
    cell is dumped needs that cell's ingest back to re-dump it, which the next
    cell's re-ingest has already destroyed.

    Args:
        benchmark_path: benchmark YAML/JSON; None uses ``settings``.
        printer / indent / label_text: how the line is written.
        max_unmatched: unmatched gold texts tolerated before the line says FAIL.
        max_reasons: how many unmatched texts are named under it, 0 for none.
        no_file_text / no_gold_text: the two nothing-to-gate lines, both None.
        on_error: ``note`` prints the reason for an unreadable benchmark or an
            unreadable chunk table and returns None; ``raise`` propagates.
        gold_kwargs: `gold_evidence.gold_chunks_per_query` knobs
            (``document_match``, ``match_mode``, ``min_shingle_coverage``, ...).

    Returns:
        True when unmatched is within ``max_unmatched``, False when it is over,
        None when there was nothing to gate or the read failed under ``note``.
        An empty chunk table reports itself and gates nothing: no document the
        gold names holds a single chunk, so no quote can be called missing yet.
    """
    from .benchmark import load_benchmark
    from .gold_evidence import gold_chunks_per_query

    if on_error not in ("note", "raise"):
        raise ValueError("on_error must be 'note' or 'raise'")
    path = benchmark_path or settings.benchmark_path
    try:
        queries = load_benchmark(path)
    except Exception as exc:  # missing file, bad yaml, renamed gold key
        if on_error == "raise":
            raise
        printer(f"{indent}{label_text}: {no_file_text} ({type(exc).__name__}: {exc})")
        return None

    n_carry = sum(1 for q in queries if getattr(q, "gold_evidence", None))
    if not n_carry:
        printer(f"{indent}{label_text}: {no_gold_text} ({len(queries)} queries)")
        return None

    try:
        _, report = gold_chunks_per_query(queries, require_all=False, **(gold_kwargs or {}))
    except Exception as exc:  # DB down, unreadable chunk table
        if on_error == "raise":
            raise
        printer(f"{indent}{label_text}: unreadable ({type(exc).__name__}: {exc})")
        return None

    summary = report.summary()
    chunks_seen = summary.get("chunks_seen") or {}
    if not sum(int(v or 0) for v in chunks_seen.values()):
        printer(
            f"{indent}{label_text}: no ingest yet (0 chunks under the "
            f"{len(chunks_seen)} documents the gold names)"
        )
        return None
    matched = int(summary.get("matched") or 0)
    unmatched = int(summary.get("unmatched") or 0)
    ok = unmatched <= max_unmatched
    printer(
        f"{indent}{label_text}: {matched}/{matched + unmatched} texts matched "
        f"across {n_carry} queries [{'ok' if ok else 'FAIL'}]"
    )
    if not ok:
        for eid, reason in list(report.unmatched.items())[:max_reasons]:
            printer(f"{indent}  {eid}: {reason}")
    return ok


def cmd_doctor(_args, *, gold: bool = True) -> int:
    """
    DB reachability, the telemetry kill-switch, the live chunking policy, and
    the gold read. ``gold=False`` drops the gold line for callers that run
    before any ingest exists (`init`), where it can only say `no ingest yet`.
    """
    applied = telemetry_off.disable_all_telemetry()
    print("Telemetry kill-switch: ALL disabled")
    for key in sorted(applied):
        print(f"  {key} = {applied[key]!r}")
    print(f"\nPostgres: {settings.postgres_url}")
    reachable = ping()
    print(f"  reachable: {reachable}")
    if not reachable:
        print("  hint: the app's Postgres is the Docker container veridic-postgres-1;")
        print("        `docker compose ps` shows which host port maps to 5432 (6432 by")
        print("        default here), then set VERIDIC_EVAL_POSTGRES_URL to that port.")
    else:
        say_live_chunking_policy()
        if gold:
            say_gold_gate(benchmark_path=getattr(_args, "benchmark", None))
    print(f"Ollama (optional judge): {settings.ollama_url}  model={settings.ollama_judge_model}")
    print(f"Faithfulness backend: {settings.faithfulness_backend}")
    print(f"promptfoo cmd: {settings.promptfoo_cmd or '(pure-python fallback)'}")
    return 0


def cmd_init(args) -> int:
    """Doctor checks + yaml templates + recent conversation ids, in one shot."""
    from .templates import write_yaml_templates

    rc = cmd_doctor(args, gold=False)

    print()
    # Falls back to the built-in copies, so init writes the yamls from any CWD.
    write_yaml_templates(overwrite=bool(getattr(args, "force", False)))

    print("\nRecent conversations (newest first) - paste ids into conditions.yaml:")
    try:
        from sqlalchemy import text

        from .db import session_scope

        with session_scope() as s:
            rows = s.execute(
                text(
                    "SELECT c.id, c.title, c.created_at, COUNT(m.id) AS messages "
                    "FROM conversations c "
                    "LEFT JOIN messages m ON m.conversation_id = c.id "
                    "WHERE c.deleted_at IS NULL "
                    "GROUP BY c.id, c.title, c.created_at "
                    "ORDER BY c.created_at DESC LIMIT :n"
                ),
                {"n": args.limit},
            ).fetchall()
        if not rows:
            print("  (none found - run the benchmark questions through the app first)")
        for r in rows:
            print(f"  {r.id}  title={r.title!r}  created={r.created_at}  messages={r.messages}")
    except Exception as exc:
        print(f"  (could not list conversations: {exc})")

    _describe_conditions(getattr(args, "conditions", "conditions.yaml"))

    print("\nNext: fill benchmark.yaml + conditions.yaml, then `veridic-eval run`.")
    return rc


def _describe_conditions(path: str) -> None:
    """Parse the conditions file init just wrote and say what it declares.

    A yaml that cannot be parsed, names an off-grid reference or misspells a key
    is caught here, one command before the run that depends on it.
    """
    if not os.path.exists(path):
        print(f"\n{path}: not found, so nothing to check yet")
        return
    print(f"\n{path} declares:")
    try:
        parsed = load_conditions_file(path, printer=None)
    except Exception as exc:
        print(f"  UNUSABLE: {type(exc).__name__}: {exc}")
        return
    for cell in parsed.cells:
        role = "reference" if cell.name == parsed.reference else (
            "grid" if cell.name in parsed.grid else "off-grid, not scored in the 2x2"
        )
        window = (
            f"{len(cell.condition.conversation_ids)} conversation id(s)"
            if cell.condition.conversation_ids
            else f"{cell.condition.start} -> {cell.condition.end}"
        )
        cost = "cost inputs" if cell.cost else "no cost: block, no CFCA"
        pac = "per_answer_cost set" if cell.per_answer_cost is not None else "no per_answer_cost, CFCA degenerate"
        print(f"  {cell.name}: {role}; {window}; {pac}; {cost}; "
              f"snapshot={'on' if cell.snapshot else 'off'}")
    for note in parsed.notes:
        print(f"  note: {note}")


def _sit_kwargs(args) -> Dict:
    """Arguments for `sitting.open_sitting`, shared by `sit` and `run --sit`."""
    from datetime import timedelta

    from .conditions import parse_timestamp

    pad = timedelta(seconds=getattr(args, "pad_s", 60.0))
    return {
        "conditions_path": args.conditions,
        "benchmark_path": getattr(args, "benchmark", None) or settings.benchmark_path,
        "conversation_ids": tuple(getattr(args, "conversation", ()) or ()),
        "since": parse_timestamp(getattr(args, "since", None)),
        "until": parse_timestamp(getattr(args, "until", None)),
        "merge": not bool(getattr(args, "single_chat", False)),
        "pad_before": pad,
        "pad_after": pad,
        "run": getattr(args, "run_entry", None),
        "dry_run": bool(getattr(args, "dry_run", False)),
    }


def cmd_sit(args) -> int:
    """Point each named cell at the chat that actually holds its answers."""
    from .sitting import open_sittings

    results = open_sittings(args.cells, **_sit_kwargs(args))
    if getattr(args, "json", False):
        print(json.dumps([r.as_dict() for r in results], indent=2, default=str))
    stuck = [r for r in results if not r.ok]
    if stuck:
        print("\nNothing written for: " + ", ".join(r.cell for r in stuck))
        print("Ask that cell's benchmark questions in the app first, or name the "
              "chat yourself with --conversation <id>.")
        return 1
    print("\nNext: `veridic-eval run --dump-only --only "
          + " ".join(r.cell for r in results) + "`")
    return 0


def dump_only_status(
    names: List[str],
    cell_dir: str,
    next_command: str = "veridic-eval run --require-measured-power --strict-conditions",
    repeat_note: str = "Run the next cell in the app, then this command again.",
    suffix: str = ".json",
    printer: Callable[[str], None] = print,
) -> List[str]:
    """The tail lines of a --dump-only run: read which declared cells have a
    snapshot in `cell_dir`, print the command that scores those cells right
    now, plus the cells still missing, and return the missing names. A dump
    that linked 0 rows deleted its own file, so file existence is the state."""
    missing = [n for n in names
               if not os.path.isfile(os.path.join(cell_dir, n + suffix))]
    present = [n for n in names if n not in missing]
    head = "\nSnapshots only (--dump-only), nothing scored. "
    if not missing:
        printer(head + f"All {len(names)} declared cells have snapshots. "
                f"Next: `{next_command}`")
    elif present:
        printer(head + "No snapshot yet for: " + ", ".join(missing)
                + ". " + repeat_note)
        printer("Report over the cells already on disk, any time: `"
                + next_command + " --only " + " ".join(present) + "`")
    else:
        printer(head + "No snapshot yet for: " + ", ".join(missing)
                + ". " + repeat_note)
    return missing


def cmd_run(args) -> int:
    """Snapshots, the full 2x2 + off-grid report, CFCA and the math check."""
    out_dir = args.out or settings.output_dir
    if getattr(args, "sit", None) is not None:
        from .sitting import open_sittings

        cells = list(args.sit) or list(args.only or ())
        if not cells:
            print("--sit needs a cell name, or --only to say which cells to open")
            return 1
        open_sittings(cells, **_sit_kwargs(args))
        print()
    try:
        refresh, _refresh_tag, refresh_note = resolve_refresh(args.refresh, only=args.only)
    except ValueError as exc:
        print(f"Nothing re-dumped: {exc}")
        return 1
    if refresh_note:
        print(refresh_note)
    try:
        result = run_experiment(
            args.conditions,
            benchmark_path=args.benchmark,
            out_dir=out_dir,
            cell_dir=args.cell_dir,
            snapshot=not args.no_snapshot,
            refresh=refresh,
            only=args.only,
            skip=args.skip or (),
            source="live" if args.live else "snapshot",
            live_fallback=not args.snapshots_only,
            keep_empty=args.keep_empty,
            score=not args.dump_only,
            reference=args.reference,
            conditions_kwargs={"strict_top_level": bool(args.strict_conditions)},
            with_cfca=not args.no_cfca,
            with_measured_ingest=not args.no_ingest,
            require_measured_power=_require_measured_power(args.require_measured_power),
            dump_kwargs={"power_on_error": args.power_on_error},
            verify=not args.no_verify,
            verify_kwargs=_verify_kwargs(args),
            verify_max_rows=_verify_rows(args),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nNot scored: {exc}")
        return 1
    if args.dump_only:
        dump_only_status(result["conditions"].names, result["paths"]["cell_dir"])
        return 0
    if not result["ok"] and args.verify_strict:
        return 1
    return 0


def _verify_rows(args) -> Optional[int]:
    """0 lists every row; anything else caps the listing."""
    return None if args.verify_rows == 0 else args.verify_rows


def _verify_kwargs(args) -> Dict:
    """Verifier arguments shared by `run` and `verify`."""
    return {
        "alpha": args.alpha,
        "expect_grid_contrasts": args.expect_contrasts,
        "expect_offgrid_contrasts": (
            None if args.offgrid_as_declared else args.expect_offgrid_contrasts
        ),
        "require_offgrid": args.require_offgrid,
        "fail_on": ("fail", "warn") if args.warn_is_error else ("fail",),
    }


def _require_measured_power(
    value: Optional[str],
    *,
    default: Union[str, List[str]] = "scored",
    scored_word: str = "scored",
    none_word: str = "none",
    separator: str = ",",
    fold_keywords: bool = True,
    strip_names: bool = True,
    drop_empty_names: bool = True,
    dedupe_names: bool = True,
    allow_empty_list: bool = False,
    flag: str = "--require-measured-power",
) -> Union[str, List[str]]:
    """
    `--require-measured-power` text -> the `run_experiment` argument.

    The two keywords pass through as themselves and anything else is a list of
    cell names, so a cell named `scored` or `none` can only be demanded from
    Python. Nothing here reads the yaml: an unknown name fails later, where
    `run_experiment` knows which cells were declared.

    Args:
        value: the raw flag text. None or blank takes `default`.
        default: what an absent flag means.
        scored_word / none_word: the keywords, matched whole and never split.
        separator: what splits a list of names.
        fold_keywords: match the keywords case-insensitively. Cell names keep
            their case, since they are compared to the yaml verbatim.
        strip_names: trim each name, for `a, b`.
        drop_empty_names: ignore blank entries, for a trailing separator.
        dedupe_names: keep the first spelling of a repeated name.
        allow_empty_list: return `[]` when no name survives. Off, because an
            empty list demands measured watts of nothing, which is what
            `none_word` already says out loud, so a typo would silently stop
            requiring them.
        flag: the name printed in the error, for a caller wiring its own flag.

    Returns:
        `scored_word`, `none_word`, or the list of cell names.
    """
    if value is None:
        return default
    text = value.strip() if strip_names else value
    if not text:
        return default
    probe = text.casefold() if fold_keywords else text
    for word in (scored_word, none_word):
        if probe == (word.casefold() if fold_keywords else word):
            return word
    names: List[str] = []
    for raw in text.split(separator):
        name = raw.strip() if strip_names else raw
        if not name and drop_empty_names:
            continue
        if dedupe_names and name in names:
            continue
        names.append(name)
    if not names and not allow_empty_list:
        raise ValueError(
            f"{flag} named no cell: pass {scored_word!r}, {none_word!r}, or names "
            f"like {separator.join(('control', 'combined'))}, got {value!r}"
        )
    return names


def _add_sit_args(parser, *, with_paths: bool) -> None:
    """
    Flags for opening a sitting, shared by `sit` and `run --sit`.

    with_paths: adds --conditions/--benchmark, which `run` already carries.
    """
    if with_paths:
        parser.add_argument("--conditions", default="conditions.yaml",
                            help="conditions YAML to patch (default conditions.yaml)")
        parser.add_argument("--benchmark", default=settings.benchmark_path,
                            help="benchmark YAML whose questions are searched for")
    parser.add_argument("--conversation", action="append", default=[], metavar="ID",
                        help="use this chat instead of searching for one; repeatable")
    parser.add_argument("--run", dest="run_entry", default=None, metavar="ID|N",
                        help="which `runs:` entry to write (id like r2, or 1-based number); "
                             "default is the first live one")
    parser.add_argument("--since", default=None, metavar="ISO",
                        help="ignore chats older than this bound; same spelling as a "
                             "yaml window bound (2026-07-01T00:00:00Z)")
    parser.add_argument("--until", default=None, metavar="ISO",
                        help="ignore chats newer than this bound")
    parser.add_argument("--single-chat", action="store_true",
                        help="pin one chat only; by default a sitting spans every chat needed "
                             "to cover the benchmark, since the app opens a new chat per reload")
    parser.add_argument("--pad-s", type=float, default=60.0, metavar="SECONDS",
                        help="slack written around the first and last question (default 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the conditions.yaml patch instead of writing it")


def _add_verify_args(parser, *, with_switches: bool) -> None:
    """
    Verifier flags on both `run` and `verify`.

    with_switches: adds the flags that only make sense while running the eval
    (`--no-verify`, `--verify-strict`), so `verify` does not advertise them.
    """
    parser.add_argument("--alpha", type=float, default=None,
                        help="significance level for the CI-vs-p cross-read "
                             "(default 1 - confidence_level)")
    parser.add_argument("--expect-contrasts", type=int, default=None,
                        help="Bonferroni family size the grid must declare "
                             "(default len(grid) - 1)")
    parser.add_argument("--expect-offgrid-contrasts", type=int, default=1,
                        help="family size the off-grid block must use (default 1, uncorrected)")
    parser.add_argument("--offgrid-as-declared", action="store_true",
                        help="accept whatever off-grid divisor the report declares "
                             "and only check its arithmetic")
    parser.add_argument("--require-offgrid", action="store_true",
                        help="fail when the report carries no before/after block")
    parser.add_argument("--warn-is-error", action="store_true",
                        help="treat warnings as failures")
    parser.add_argument("--verify-rows", type=int, default=40,
                        help="max rows listed; 0 lists all (default 40)")
    if with_switches:
        parser.add_argument("--no-verify", action="store_true",
                            help="skip the post-run math check")
        parser.add_argument("--verify-strict", action="store_true",
                            help="exit non-zero when the post-run check fails")


def cmd_verify(args) -> int:
    """Re-derive every statistic in a written report from the report's own numbers."""
    result = verify_report_file(args.report, **_verify_kwargs(args))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(args.report)
        print(render_verify_text(
            result,
            show=("fail", "warn", "pass") if args.all else ("fail", "warn"),
            max_rows=_verify_rows(args),
        ))
    return 0 if result["ok"] else 1


def cmd_view(args) -> int:
    """Open the snapshot viewer. The only UI in the repo, and no metric needs it.

    Every flag is explicit: the app file, the streamlit executable, the port and
    address, whether to force headless (`.streamlit/config.toml` already sets it),
    and `--print-only` to see the command instead of running it.
    """
    import shutil
    import subprocess
    import sys

    app = args.app or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_app.py")
    if not os.path.exists(app):
        print(f"viewer app not found: {app}")
        return 1

    exe = args.streamlit or shutil.which("streamlit")
    cmd = [exe, "run", app] if exe else [sys.executable, "-m", "streamlit", "run", app]
    if args.port is not None:
        cmd += ["--server.port", str(args.port)]
    if args.address:
        cmd += ["--server.address", args.address]
    if args.headless is not None:
        cmd += ["--server.headless", "true" if args.headless else "false"]

    print(" ".join(cmd))
    if args.print_only:
        return 0
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("streamlit is missing from this environment; `pip install -e .` brings it back")
        return 1


def cmd_meter(args) -> int:
    """Log watts for the whole experiment; `run` slices it per cell."""
    from .devices import probe_device, resolve_device
    from .power_meter import log_power

    device, power = args.device, {}
    if device is None:
        if not os.path.exists(args.conditions):
            print(f"no --device and no {args.conditions}; name the machine, e.g. "
                  f"`veridic-eval meter --device omen`")
            return 1
        cf = load_conditions_file(args.conditions, printer=None)
        device, power = cf.device, dict(cf.power or {})
        if device is None:
            print(f"{args.conditions} declares no `device:`; add one (device: omen) "
                  f"or pass --device")
            return 1

    profile = resolve_device(device, where=args.conditions)
    if args.probe:
        print(json.dumps(probe_device(profile), indent=2))
        return 0

    out = args.out or power.get("log_path") or "out/power/power.csv"
    summary = log_power(
        profile,
        out_path=out,
        tag=args.tag,
        interval_s=args.interval_s,
        duration_s=args.duration_s,
        append=not args.truncate,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "device"}, indent=2))
    return 0 if summary["valid_rows"] else 1


def cmd_reprice(args) -> int:
    """Price the dumped cells at the end, off the finished power log."""
    from .cells_app import reprice_cells

    device, power = args.device, {}
    if os.path.exists(args.conditions):
        cf = load_conditions_file(args.conditions, printer=None)
        power = dict(cf.power or {})
        device = device or cf.device
    if device is None:
        print(f"no --device and no `device:` in {args.conditions}; name the machine, "
              f"e.g. `veridic-eval reprice --device omen`")
        return 1
    if args.log:
        power["log_path"] = args.log

    overrides = {k: v for k, v in (("basis", args.basis), ("tag", args.tag)) if v}
    where = {}
    if args.cell_dir:
        where["cell_dir"] = args.cell_dir
    print(f"Re-pricing off {power.get('log_path') or 'out/power/power.csv'}:")
    got = reprice_cells(
        args.cells,
        device=device,
        power=power,
        overrides=overrides,
        min_coverage=args.min_coverage,
        fallback_tdp=args.fallback_tdp,
        require_measured=args.require_measured,
        write=not args.dry_run,
        missing="skip" if args.skip_missing else "raise",
        **where,
    )
    if not got:
        print(f"no dumped cells found; `veridic-eval run --dump-only` writes them")
        return 1
    unmeasured = sorted(c for c, r in got.items() if not (r.get("after") or {}).get("ok"))
    if unmeasured:
        print(f"still unmeasured: {', '.join(unmeasured)}")
        return 1
    if not args.dry_run:
        print("Score it: `veridic-eval run --require-measured-power --strict-conditions`")
    return 0


def cmd_warmup(args) -> int:
    """Price the written cells again without each sitting's first answer.

    The load of a cold model is billed to whichever answer waits for it, and on
    a 10-question cell that one wait can be half the cell's serving time. This
    re-prices the seconds with it removed and files the result as a sensitivity
    next to the reported figure: nothing is re-scored and nothing is replaced.
    """
    from .warmup import WARM_FIELD, backfill_report

    if not os.path.exists(args.report):
        print(f"no report at {args.report}; `veridic-eval run` writes one")
        return 1

    cell_dir = args.cell_dir
    if not cell_dir:
        tried = [os.path.join(os.path.dirname(os.path.abspath(args.report)), "cells"),
                 os.path.join(settings.output_dir, "cells")]
        cell_dir = next((p for p in tried if os.path.isdir(p)), "")
        if not cell_dir:
            print("no cells directory found; pass --cell-dir (tried "
                  f"{', '.join(tried)})")
            return 1

    md = "" if args.no_md else args.md
    try:
        block = backfill_report(
            args.report, cell_dir,
            drop=args.drop,
            md_path=md,
            write=not args.dry_run,
            on_mismatch="warn" if args.allow_stale_cells else "raise",
            printer=print,
        )
    except ValueError as exc:
        print(f"warm-only pricing refused: {exc}")
        return 1

    cells = block.get("cells") or {}
    warmup = block.get("warmup") or {}
    old = ((_read_json(args.report) or {}).get("cfca_gbp") or {}).get("cells") or {}
    print(f"\n{'cell':<14} {'dropped':>9} {'share':>6} {'s/ans':>8} {'warm':>8} "
          f"{'CFCA':>12} {'warm':>12}")
    for name, entry in cells.items():
        spans = warmup.get(name) or {}
        share = spans.get("dropped_share")
        print(f"{name:<14} {spans.get('dropped_s', 0):>9.2f} "
              f"{'' if share is None else f'{100 * share:.0f}%':>6} "
              f"{spans.get('per_answer_s') or 0:>8.2f} {spans.get('warm_per_answer_s') or 0:>8.2f} "
              f"{(old.get(name) or {}).get('cfca') or 0:>12.3e} {entry.get('cfca') or 0:>12.3e}")
    if args.dry_run:
        print("\ndry run: nothing was written")
        return 0
    print(f"\nCheck it: `veridic-eval verify --report {args.report} --all` "
          f"(the block reads as {WARM_FIELD})")
    return 0


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)

    # `cost` and `power` own their full argparse (every flag explicit); route
    # before the top-level parser so their --help stays intact.
    if argv and argv[0] == "cost":
        from .cfca_cost import main as cost_main

        return cost_main(argv[1:])
    if argv and argv[0] == "power":
        from .power_log import main as power_main

        return power_main(argv[1:])

    parser = argparse.ArgumentParser(prog="veridic_eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check DB + telemetry status").set_defaults(func=cmd_doctor)

    ini = sub.add_parser("init", help="doctor + yaml templates + recent conversation ids")
    ini.add_argument("--limit", type=int, default=10, help="conversations to list (default 10)")
    ini.add_argument("--force", action="store_true", help="overwrite existing benchmark/conditions yaml")
    ini.add_argument("--conditions", default="conditions.yaml",
                     help="conditions file to write and then describe (default ./conditions.yaml)")
    ini.set_defaults(func=cmd_init)

    sit = sub.add_parser(
        "sit",
        help="edit conditions.yaml only: point a cell's window and chat ids at every chat "
             "holding its answers. Scores nothing, writes nothing to the DB",
    )
    sit.add_argument("cells", nargs="+", metavar="CELL",
                     help="cells to patch, in the order they were asked")
    sit.add_argument("--json", action="store_true", help="print the full result as JSON")
    _add_sit_args(sit, with_paths=True)
    sit.set_defaults(func=cmd_sit)

    run = sub.add_parser("run", help="eval only: snapshots + full 2x2 + off-grid + CFCA + math "
                                     "check, on the windows conditions.yaml already declares")
    run.add_argument("--conditions", default="conditions.yaml", help="conditions YAML/JSON")
    run.add_argument("--benchmark", default=settings.benchmark_path, help="benchmark YAML/JSON")
    run.add_argument("--out", default=None, help="output directory (default ./out)")
    run.add_argument("--cell-dir", default=None,
                     help="per-cell snapshot directory (default <out>/cells)")
    run.add_argument("--dump-only", action="store_true",
                     help="snapshot the declared cells and stop, for use right after each in-app cell")
    run.add_argument("--no-snapshot", action="store_true",
                     help="score the live DB without writing snapshots first")
    run.add_argument("--refresh", nargs="*", default=None, metavar="CELL",
                     help="re-dump even though a snapshot exists (kept if the re-dump is empty). "
                          "Bare --refresh takes the --only cells, and refuses to run without one")
    run.add_argument("--only", nargs="*", default=None, metavar="CELL",
                     help="restrict snapshotting and scoring to these cells")
    run.add_argument("--skip", nargs="*", default=[], metavar="CELL",
                     help="leave these cells out entirely")
    run.add_argument("--live", action="store_true",
                     help="read every cell from Postgres, treat the snapshots as archive only")
    run.add_argument("--snapshots-only", action="store_true",
                     help="score only cells that have a snapshot; never read the live DB")
    run.add_argument("--keep-empty", action="store_true",
                     help="archive a snapshot that captured 0 rows or 0 linked answers instead of rejecting it")
    run.add_argument("--reference", default=None,
                     help="override the 2x2 reference cell named in conditions.yaml")
    run.add_argument("--no-cfca", action="store_true",
                     help="skip the GBP CFCA block built from the `cost:` quantities")
    run.add_argument("--no-ingest", action="store_true",
                     help="price the `onetime_gpu_hours` typed in conditions.yaml instead of the "
                          "chunking hours measured off the ingest stamps")
    run.add_argument("--power-on-error", choices=("warn", "raise", "skip"), default="warn",
                     help="a failed power measurement warns and still writes the snapshot "
                          "(default), with the reason under `meta.power.error`; `skip` writes it "
                          "without the warning line, `raise` fails that dump and writes nothing")
    run.add_argument("--require-measured-power", nargs="?", const="scored",
                     default="scored", metavar="scored|none|CELL,CELL",
                     help="which cells must carry measured watts before the GBP cost blocks are "
                          "built: every scored cell with a `cost:` block (default scored, and the "
                          "bare flag says the same), `none` to price the typed watts, or just "
                          "these cells by name. The check needs a `device:` in conditions.yaml, "
                          "and `--no-cfca` drops it with the block")
    run.add_argument("--sit", nargs="*", default=None, metavar="CELL",
                     help="patch these cells in conditions.yaml first (window + chat id read "
                          "off the logs), then eval; no names means the --only cells")
    _add_sit_args(run, with_paths=False)
    run.add_argument("--strict-conditions", action="store_true",
                     help="unknown top-level keys in conditions.yaml are errors")
    _add_verify_args(run, with_switches=True)
    run.set_defaults(func=cmd_run)

    ver = sub.add_parser("verify", help="post-run math check on a written report.json")
    ver.add_argument("--report", default=os.path.join(settings.output_dir, "report.json"),
                     help="report.json to check (default ./out/report.json)")
    ver.add_argument("--all", action="store_true", help="list passing checks too")
    ver.add_argument("--json", action="store_true", help="print the full result as JSON")
    _add_verify_args(ver, with_switches=False)
    ver.set_defaults(func=cmd_verify)

    vw = sub.add_parser("view", help="open the snapshot viewer on out/cells/*.json (optional UI)")
    vw.add_argument("--app", default=None, help="app file (default veridic_eval/cells_app.py)")
    vw.add_argument("--streamlit", default=None,
                    help="streamlit executable (default: PATH, else `python -m streamlit`)")
    vw.add_argument("--port", type=int, default=None, help="server port (default: streamlit's own)")
    vw.add_argument("--address", default=None, help="bind address (default: streamlit's own)")
    vw.add_argument("--headless", dest="headless", action="store_true",
                    help="force headless (default: whatever .streamlit/config.toml says)")
    vw.add_argument("--no-headless", dest="headless", action="store_false",
                    help="force the browser to open")
    vw.add_argument("--print-only", action="store_true",
                    help="print the command instead of running it")
    vw.set_defaults(func=cmd_view, headless=None)

    mtr = sub.add_parser("meter", help="log this machine's watts for the whole experiment; every cell is priced off this one file")
    mtr.add_argument("--device", default=None,
                     help="machine profile: a name (omen, spark), else read from conditions.yaml `device:`")
    mtr.add_argument("--conditions", default="conditions.yaml",
                     help="file the device and power settings come from when --device is absent")
    mtr.add_argument("--out", default=None, help="power log CSV (default out/power/power.csv)")
    mtr.add_argument("--tag", default="", help="label written on every row, to split one file later")
    mtr.add_argument("--interval-s", type=float, default=None, help="poll cadence (default: the device's own)")
    mtr.add_argument("--duration-s", type=float, default=None, help="stop after N seconds (default: until Ctrl+C)")
    mtr.add_argument("--truncate", action="store_true",
                     help="start the log empty, discarding every cell already measured")
    mtr.add_argument("--probe", action="store_true", help="check the sampler works and exit")
    mtr.set_defaults(func=cmd_meter)

    rp = sub.add_parser("reprice", help="price the dumped cells at the end, off the finished power log "
                                       "(no database, no re-ingest)")
    rp.add_argument("cells", nargs="*", metavar="CELL",
                    help="cells to re-price; none takes every dumped cell")
    rp.add_argument("--conditions", default="conditions.yaml",
                    help="file the `device:` and `power:` settings come from")
    rp.add_argument("--device", default=None,
                    help="machine profile: a name (omen, spark), when the conditions file has none")
    rp.add_argument("--cell-dir", default=None,
                    help="directory holding the dumps (default <out>/cells)")
    rp.add_argument("--log", default=None,
                    help="power CSV to price from (default the `power:` block, else out/power/power.csv)")
    rp.add_argument("--basis", choices=["busy", "window"], default=None,
                    help="price the answer spans (busy) or the whole run window; default: the conditions file's")
    rp.add_argument("--tag", default=None, help="price only the log rows carrying this meter tag")
    rp.add_argument("--min-coverage", type=float, default=None,
                    help="coverage floor (default: the `power:` block's); 0.0 prices whatever the "
                         "finished log covers and reports the fraction instead of gating on it")
    rp.add_argument("--fallback-tdp", dest="fallback_tdp", action="store_true", default=None,
                    help="replace a thin log's measured watts with the nameplate upper bound")
    rp.add_argument("--no-fallback-tdp", dest="fallback_tdp", action="store_false",
                    help="report a thin log's measured watts instead of the nameplate")
    rp.add_argument("--require-measured", action="store_true",
                    help="leave a file untouched unless the new price is measured")
    rp.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    rp.add_argument("--skip-missing", action="store_true",
                    help="a named cell with no dump is skipped instead of an error")
    rp.set_defaults(func=cmd_reprice)

    wu = sub.add_parser("warmup", help="warm-only sensitivity: re-price the written cells without "
                                       "each sitting's first answer (no re-score, no re-run)")
    wu.add_argument("--report", default=os.path.join(settings.output_dir, "report.json"),
                    help="report.json to add the block to (default ./out/report.json)")
    wu.add_argument("--cell-dir", default=None,
                    help="dumps the report was scored from (default <report dir>/cells)")
    wu.add_argument("--drop", type=int, default=1,
                    help="answers dropped from the head of each sitting (default 1)")
    wu.add_argument("--md", default=None,
                    help="report.md to write the section into (default: beside the report.json)")
    wu.add_argument("--no-md", action="store_true", help="write the JSON block only")
    wu.add_argument("--allow-stale-cells", action="store_true",
                    help="warn instead of refusing when the dumps do not re-sum to the "
                         "report's own serving seconds")
    wu.add_argument("--dry-run", action="store_true", help="print the table, write nothing")
    wu.set_defaults(func=cmd_warmup)

    # Listed for --help only; real dispatch happens above.
    sub.add_parser("cost", help="CFCA cost calculator, GBP; `veridic-eval cost --help`")
    sub.add_parser("power", help="GPU power poller for electricity pricing; `veridic-eval power --help`")

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
