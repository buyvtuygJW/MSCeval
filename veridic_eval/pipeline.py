"""
The whole evaluation behind one call, so `init` + `run` is the entire flow.

`run_experiment` reads conditions.yaml, snapshots each declared cell to
``out/cells/<cell>.json`` while its ingest is still live, scores whatever is
present (snapshot first, live Postgres for anything not snapshotted yet),
computes the GBP CFCA from the declared cost quantities, writes report.json and
report.md, then re-derives every statistic with the post-run math check.

The snapshot step is what makes the before-window measurable at all: a re-ingest
under the other chunker deletes ``message_evidence`` rows through the cascade on
``documents``, so `prebaseline` has to leave the database as JSON before the
chunking correction lands. Running the same command after each cell is
therefore safe by design: an existing snapshot is reused untouched, never
re-dumped from a database that no longer holds those rows.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .benchmark import load_benchmark
from .cells import CELL_SUBDIR, CELL_SUFFIX, cell_dir_for, cell_file_path
from .conditions import (
    ConditionsFile,
    DeclaredCell,
    apply_measured_power,
    cfca_for_cells,
    load_conditions_file,
    p_hat_from_report,
    render_cfca_markdown,
)
from .cfca_cost import per_answer_cost_gbp
from .cfca_metric import apply_measured_tokens
from .config import Condition, settings
from .extract import QueryRecord, extract_condition
from .ingest_cost import apply_measured_ingest, check_ingest_policies, ingest_policy_block
from .report import evaluate_experiment_split, evaluate_partial_split, render_markdown
from .verify import render_verify_text, verify_report

JSON_NAME: str = "report.json"
MD_NAME: str = "report.md"
CFCA_FIELD: str = "cfca_gbp"

#: Dumper meta fields the CFCA reads back off a snapshot.
COST_META_KEY: str = "cost_inputs_measured"
INGEST_META_KEY: str = "ingest_measured"
#: The whole ingest measurement, span and token block included. The cost block
#: carries hours and Mtok only, so this is where the chunking policy label the
#: build ran under survives, and `check_ingest_policies` reads it from here.
INGEST_DETAIL_META_KEY: str = "ingest"
#: The per-answer serving token counts the app logged, for the query terms.
TOKENS_META_KEY: str = "tokens_measured"
#: Every latency sum measured for a cell, priced and unpriced, for the report.
SERVING_META_KEY: str = "serving_seconds"

#: `snapshot_cells` statuses.
WRITTEN, REUSED, EMPTY, SKIPPED, FAILED = "written", "reused", "empty", "skipped", "failed"

#: `snapshot_verdict` rejection tags.
NO_RECORDS, NO_LINKED = "no_records", "no_linked"

#: rejection tag -> the line `snapshot_cells` prints. Fields available to it:
#: ``{cell}``, ``{path}``, ``{n_records}``, ``{n_linked}``.
EMPTY_NOTES: Dict[str, str] = {
    NO_RECORDS: "0 rows, no snapshot kept (nothing logged in that window yet)",
    NO_LINKED: ("{n_linked}/{n_records} queries linked, no snapshot kept "
                "(ask this cell's benchmark questions in the app first, or --keep-empty)"),
}


class _Blanks(dict):
    """`{unknown}` in a caller's note prints as itself, never as a KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def snapshot_path(name: str, cell_dir: str, *, suffix: str = CELL_SUFFIX) -> str:
    """``<cell_dir>/<name>.json``, through `cells.cell_file_path`."""
    return cell_file_path(name, cell_dir=cell_dir, suffix=suffix)


# --------------------------------------------------------- 0. power wiring
#: `power_config_verdict` outcome tags.
POWER_OK, POWER_NO_DEVICE, POWER_NO_BLOCK = "ok", "no_device", "no_power_block"

#: The lines a conditions file needs before one watt can reach a cell.
POWER_KEYS_HINT: str = (
    "device: omen-system               # whole machine off the CPU utilisation counter\n"
    "power: {min_samples: 1}           # slices out/power/power.csv per answer's serving span"
)


def power_config_verdict(
    device: Any,
    *,
    power: Optional[Dict[str, Any]] = None,
    require_measured_power: Any = "scored",
    metered_cells: Sequence[str] = (),
    conditions_path: str = "conditions.yaml",
    device_key: str = "device",
    power_key: str = "power",
    keys_hint: str = POWER_KEYS_HINT,
    require_power_block: bool = False,
    ok_tag: str = POWER_OK,
    no_device_tag: str = POWER_NO_DEVICE,
    no_block_tag: str = POWER_NO_BLOCK,
) -> Tuple[bool, str, str]:
    """Say whether this conditions file can measure watts at all, before a run.

    A file that names no ``device:`` skips the whole meter path: the dump
    records no ``meta.power``, `conditions.apply_measured_power` is never
    called, and ``--require-measured-power`` silently enforces nothing, so a
    cell prices on the typed ``cost_defaults.watts`` fallback while the meter
    log fills up beside it. The sitting cannot be retaken, so the refusal has
    to happen before the dump rather than at scoring time.

    Args:
        device: the file's ``device:`` value, None when the key is absent.
        power: the file's ``power:`` block, None or empty when absent.
        require_measured_power: the caller's own setting, read the same way
            `run_experiment` reads it. ``none`` means no cell needs measured
            watts, so a missing device is the user's declared choice and
            passes. ``scored`` or an explicit list of cells demands watts.
        metered_cells: the cells ``scored`` expands to, named in the message.
        conditions_path: the file named in the message.
        device_key / power_key: key names, for a caller that renamed them.
        keys_hint: the exact lines the message tells the user to add.
        require_power_block: True also rejects a device with no ``power:``
            block, for a caller that refuses to rely on the block's defaults.
        ok_tag / no_device_tag / no_block_tag: returned tags.

    Returns:
        ``(ok, tag, message)``, with ``message`` empty when ``ok``.
    """
    if isinstance(require_measured_power, str):
        required: List[str] = [] if require_measured_power == "none" else list(metered_cells)
    else:
        required = list(require_measured_power)

    if device is not None:
        if require_power_block and not power:
            return False, no_block_tag, (
                f"{conditions_path} names {device_key}: {device!r} but no {power_key}: block. "
                f"Add it:\n{keys_hint}"
            )
        return True, ok_tag, ""

    if not required:
        return True, ok_tag, ""

    which = ", ".join(required) if required else "every scored cell"
    return False, no_device_tag, (
        f"{conditions_path} names no {device_key}:, so no watt can reach a cell and "
        f"require_measured_power={require_measured_power!r} would enforce nothing on {which}. "
        f"Add these two lines at the top level, then re-dump the affected cells with "
        f"`--only <cell> --refresh`:\n{keys_hint}"
    )


# ----------------------------------------------- 0b. which cells to re-dump
#: `resolve_refresh` outcome tags.
REFRESH_OFF, REFRESH_NAMED, REFRESH_SELECTED, REFRESH_ALL = (
    "off", "named", "selected", "all")


def resolve_refresh(
    refresh: Optional[Sequence[str]],
    *,
    only: Optional[Sequence[str]] = None,
    known_cells: Sequence[str] = (),
    bare_without_only: str = "raise",
    outside_only: str = "raise",
    unknown_cell: str = "raise",
    flag: str = "--refresh",
    only_flag: str = "--only",
    off_tag: str = REFRESH_OFF,
    named_tag: str = REFRESH_NAMED,
    selected_tag: str = REFRESH_SELECTED,
    all_tag: str = REFRESH_ALL,
) -> Tuple[Tuple[str, ...], str, str]:
    """Turn a refresh request into the exact cells `snapshot_cells` may re-dump.

    ``--only`` filters which cells enter the loop; reuse is a separate decision
    (an existing file with enough linked queries is kept untouched), so forcing
    one cell used to mean naming it twice. Here a bare request inherits
    ``only``, which is the whole point: one intent, one cell name.

    A bare request with no ``only`` is refused rather than expanded, because a
    re-dump reads today's database and a cell whose ingest has since been
    replaced comes back stripped of the evidence rows it was scored on.

    Args:
        refresh: None when the flag was never given, an empty sequence for the
            bare flag, otherwise the cells named on it. A caller that cannot
            tell absent from bare should pass None for both.
        only: the run's ``--only`` filter, None when the run covers every cell.
        known_cells: every cell the conditions file declares. Empty disables
            the ``unknown_cell`` check and the list inside its message.
        bare_without_only: ``raise``, or ``all`` to expand to `known_cells`, or
            ``off`` to treat the bare flag as no refresh at all.
        outside_only: ``raise``, or ``drop``, when a named cell is one that
            ``only`` filters out and so could never be dumped anyway.
        unknown_cell: ``raise``, or ``drop``, or ``keep``, for a name that is
            not in `known_cells` (a typo, which otherwise refreshes nothing).
        flag / only_flag: the names used in the messages.
        off_tag / named_tag / selected_tag / all_tag: returned tags.

    Returns:
        ``(cells, tag, note)``, deduplicated in the order given, with ``note``
        a one-line summary for the caller to print, empty when nothing is
        being refreshed.

    Raises:
        ValueError: on a refused request, or an unrecognised setting.
    """
    if refresh is None:
        return (), off_tag, ""

    named = [str(n) for n in refresh]
    picked = None if only is None else [str(n) for n in only]
    declared = [str(n) for n in known_cells]

    if named:
        cells, tag = named, named_tag
    elif picked:
        cells, tag = list(picked), selected_tag
    elif bare_without_only == "off":
        return (), off_tag, ""
    elif bare_without_only == "all":
        cells, tag = list(declared), all_tag
    elif bare_without_only == "raise":
        which = ", ".join(declared) if declared else "every declared cell"
        raise ValueError(
            f"bare {flag} with no {only_flag} would re-dump {which}. A re-dump reads "
            f"the database as it stands now, so any cell whose ingest has already been "
            f"replaced comes back without the evidence rows it was scored on. Say which "
            f"run this is ({only_flag} <cell>), or name the cells ({flag} <cell> ...)."
        )
    else:
        raise ValueError("bare_without_only must be 'raise', 'all' or 'off', got "
                         f"{bare_without_only!r}")

    if tag == named_tag and picked is not None:
        stray = [c for c in cells if c not in picked]
        if stray:
            if outside_only == "raise":
                raise ValueError(
                    f"{flag} {' '.join(stray)} but {only_flag} {' '.join(picked)} leaves "
                    f"those cells out of the run, so nothing would be re-dumped. Drop them "
                    f"from {flag}, add them to {only_flag}, or use a bare {flag}."
                )
            if outside_only == "drop":
                cells = [c for c in cells if c in picked]
            else:
                raise ValueError(f"outside_only must be 'raise' or 'drop', got {outside_only!r}")

    if declared and unknown_cell != "keep":
        unknown = [c for c in cells if c not in declared]
        if unknown:
            if unknown_cell == "raise":
                raise ValueError(
                    f"{flag} names {', '.join(unknown)}, which the conditions file does not "
                    f"declare. Its cells are: {', '.join(declared)}."
                )
            if unknown_cell == "drop":
                cells = [c for c in cells if c in declared]
            else:
                raise ValueError("unknown_cell must be 'raise', 'drop' or 'keep', got "
                                 f"{unknown_cell!r}")

    seen: Dict[str, None] = {}
    for c in cells:
        seen.setdefault(c, None)
    out = tuple(seen)
    if not out:
        return (), off_tag, ""
    inherited = " (inherited from " + only_flag + ")" if tag == selected_tag else ""
    return out, tag, f"{flag}: re-dumping {', '.join(out)}{inherited}"


# ------------------------------------------------------------- 1. snapshot
def snapshot_verdict(
    n_records: int,
    n_linked: int,
    *,
    min_records: int = 1,
    min_linked: int = 1,
    keep_empty: bool = False,
    keep_unlinked: bool = False,
    reason_no_records: str = NO_RECORDS,
    reason_no_linked: str = NO_LINKED,
) -> Tuple[bool, str]:
    """Keep a freshly dumped cell, or say which rule rejected it.

    A dump writes one record per benchmark query whether or not it reached a
    logged answer (either `cells_app` dumper), so the record count never detects a
    window that matched nothing. Only ``n_linked`` does, which is why a cell
    whose questions were never asked in the app must be rejected here instead of
    becoming a 0-linked file that later runs reuse untouched.

    Args:
        n_records / n_linked: the dumper's counts for this cell.
        min_records: records the file must hold to be kept.
        min_linked: queries that must reach a logged answer to be kept. 0 keeps
            a file with no linkage at all.
        keep_empty: keep the file whatever the counts, the caller's
            ``--keep-empty``, so an archive dump can be forced through.
        keep_unlinked: keep a file that holds records but clears no linkage
            threshold, when only ``min_records`` should decide.
        reason_no_records / reason_no_linked: tags returned on rejection, keys
            into ``EMPTY_NOTES``.

    Returns:
        ``(keep, reason)``, with ``reason`` empty when the file is kept.
    """
    if keep_empty:
        return True, ""
    if n_records < min_records:
        return False, reason_no_records
    if n_linked < min_linked and not keep_unlinked:
        return False, reason_no_linked
    return True, ""


def snapshot_counts(
    path: str,
    *,
    records_key: str = "n_records",
    linked_key: str = "n_linked",
    cell_key: str = "cell",
    encoding: str = "utf-8",
    on_error: str = "skip",
) -> Optional[Dict[str, Any]]:
    """What a written snapshot says it holds, without parsing its records.

    `write_cell` puts the counts in the file header, so deciding whether a file
    on disk is worth reusing costs one read and no record rebuild.

    Args:
        path: the snapshot file.
        records_key / linked_key / cell_key: header fields to lift.
        encoding: file encoding.
        on_error: ``skip`` returns None for a missing or unreadable file, which
            callers read as "leave that file alone"; ``raise`` propagates.

    Returns:
        ``{cell, n_records, n_linked}``, or None when the file cannot be read.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error must be 'skip' or 'raise'")
    try:
        with open(path, "r", encoding=encoding) as fh:
            payload = json.load(fh)
        return {
            "cell": payload.get(cell_key),
            "n_records": int(payload.get(records_key) or 0),
            "n_linked": int(payload.get(linked_key) or 0),
        }
    except Exception:
        if on_error == "raise":
            raise
        return None


def snapshot_dumper(
    cell: str,
    *,
    runs: Sequence[Condition] = (),
    runs_mode: str = "auto",
    device: Any = None,
    power: Optional[Dict[str, Any]] = None,
    measure_power: bool = True,
    power_per_run: bool = True,
    power_on_error: str = "warn",
    power_printer: Optional[Callable[[str], None]] = print,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    conversation_ids: Sequence[str] = (),
    single_fn: Optional[Callable[..., dict]] = None,
    runs_fn: Optional[Callable[..., dict]] = None,
    **dump_kwargs: Any,
) -> dict:
    """Dump one cell through whichever `cells_app` dumper fits it, watts and all.

    The default `snapshot_cells` dumper, and the one place that knows the two
    dumpers take different shapes. `dump_cell` takes a single window
    (``start`` / ``end`` / ``conversation_ids``) and
    nothing about a meter. `dump_cell_runs` reads the window off each
    `Condition` in ``runs``, keeps every sitting apart, and is the only dumper
    that writes ``cost_inputs_measured`` for `measured_cost_inputs` to read
    back into the GBP cost blocks. Routing here means a caller hands over
    everything it knows and no argument is dropped or mis-sent.

    Args:
        cell: cell name, the ``cell`` field of the dumped file.
        runs: every sitting of this cell, ``DeclaredCell.runs``, in the order
            asked. One element is the cell asked once.
        runs_mode: "auto" sends repeats and any metered cell to
            `dump_cell_runs`, a plain single sitting to `dump_cell`; "always"
            and "never" pin the choice.
        device / power / measure_power / power_per_run / power_on_error /
            power_printer: the meter, honoured by `dump_cell_runs` only, and
            named here so a metering argument never reaches `dump_cell`, which
            takes none of them. ``device=None`` measures nothing, and then a
            single sitting needs no runs dumper.
        start / end / conversation_ids: the one window,
            for `dump_cell` only; ``runs`` already carries it per sitting.
        single_fn / runs_fn: the two dumpers; None imports
            ``cells_app.dump_cell`` / ``cells_app.dump_cell_runs`` on the spot,
            which keeps streamlit off the CLI path.
        dump_kwargs: what both take: ``benchmark_path``, ``path``,
            ``query_ids``, ``overwrite``, ``cell_dir``, the gold and match
            settings, ``include_chunk_text``, ``indent``, ``meta``.

    Returns:
        The chosen dumper's own dict, which shares ``{path, cell, n_records,
        n_linked, n_judged, gold, records}`` either way.
    """
    if runs_mode not in ("auto", "always", "never"):
        raise ValueError("runs_mode must be 'auto', 'always' or 'never'")
    metered = measure_power and device is not None
    if runs_mode == "auto":
        use_runs = bool(runs) and (len(runs) > 1 or metered)
    else:
        use_runs = runs_mode == "always"

    if use_runs:
        if not runs:
            raise ValueError(f"cell {cell!r}: no runs to dump "
                             f"(runs_mode={runs_mode!r} needs at least one sitting)")
        if runs_fn is None:
            from .cells_app import dump_cell_runs as runs_fn  # local: streamlit stays off the CLI
        return runs_fn(
            cell,
            runs,
            device=device,
            measure_power=measure_power,
            power=power,
            power_per_run=power_per_run,
            power_on_error=power_on_error,
            power_printer=power_printer,
            **dump_kwargs,
        )
    if single_fn is None:
        from .cells_app import dump_cell as single_fn  # local: streamlit stays off the CLI
    return single_fn(
        cell=cell,
        start=start,
        end=end,
        conversation_ids=conversation_ids,
        **dump_kwargs,
    )


def snapshot_cells(
    cells: Sequence[DeclaredCell],
    *,
    cell_dir: str,
    benchmark_path: Optional[str] = None,
    reuse_existing: bool = True,
    reuse_min_linked: Optional[int] = None,
    counts_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    refresh: Sequence[str] = (),
    only: Optional[Sequence[str]] = None,
    skip: Sequence[str] = (),
    enabled_only: bool = True,
    keep_empty: bool = False,
    min_records: int = 1,
    min_linked: int = 1,
    keep_unlinked: bool = False,
    verdict_fn: Optional[Callable[..., Tuple[bool, str]]] = None,
    empty_notes: Optional[Dict[str, str]] = None,
    refresh_via_temp: bool = True,
    temp_suffix: str = ".new",
    on_error: str = "warn",
    dump_fn: Optional[Callable[..., dict]] = None,
    dump_kwargs: Optional[Dict[str, Any]] = None,
    path_for: Optional[Callable[[str], str]] = None,
    make_dirs: bool = True,
    printer: Optional[Callable[[str], None]] = print,
) -> Dict[str, Dict[str, Any]]:
    """Write one JSON file per cell, reusing whatever is already on disk.

    Returns ``{cell: {status, path, n_records, n_linked, error}}`` with status
    one of ``written`` / ``reused`` / ``empty`` / ``skipped`` / ``failed``.

    Args:
        cell_dir: directory the files live in.
        benchmark_path: benchmark YAML/JSON; None uses ``settings``.
        reuse_existing: an existing file is left exactly as it is. Keep this on:
            it is the only reason a `prebaseline` snapshot survives the
            re-ingest that deletes its evidence rows.
        reuse_min_linked: linked queries an existing file must already hold to
            be reused. `snapshot_verdict` guards a fresh dump only, so without
            this a file written before that guard, or by a run whose questions
            were never asked, is reused for ever. A file below the threshold is
            re-dumped through the same temp-file path, so a database that no
            longer holds those rows leaves the old file untouched. None reuses
            whatever is on disk, and an unreadable file is always reused.
        counts_fn: reads ``{n_records, n_linked}`` off an existing file for that
            check; None uses `snapshot_counts`.
        refresh: cells re-dumped even when a file exists.
        only / skip: restrict the set; None in ``only`` means all of them.
        enabled_only: honour each cell's ``snapshot:`` flag from the yaml.
        keep_empty: keep a file whatever it captured. Off deletes a rejected
            dump so a later run tries again instead of reusing a measurement of
            nothing.
        min_records / min_linked / keep_unlinked / verdict_fn: the keep-or-delete
            rule, see `snapshot_verdict`. ``min_linked=1`` is what stops a cell
            whose questions were never asked from being written and then reused.
        empty_notes: rejection tag -> printed line, formatted with ``{cell}``,
            ``{path}``, ``{n_records}``, ``{n_linked}``; None uses ``EMPTY_NOTES``.
        refresh_via_temp: a refresh dumps beside the old file and replaces it
            only once the new dump has rows, so a refresh against a database
            that no longer holds them cannot destroy the good snapshot.
        temp_suffix: suffix of that scratch file.
        on_error: ``warn`` records the failure and carries on, ``raise``
            propagates, ``skip`` stays silent.
        dump_fn: the dumper; None uses `snapshot_dumper`, which routes repeats
            and metered cells to ``cells_app.dump_cell_runs`` and a plain single
            sitting to ``cells_app.dump_cell``. It is called with ``cell``,
            ``runs`` and the window, so an injected dumper takes ``**kwargs``
            or names all three. Injectable so the policy above is testable
            without a database.
        dump_kwargs: extra keyword arguments for every dump (``device`` and
            ``power`` for the meter, gold evidence matching, ``document_match``,
            ``include_chunk_text`` and the rest keep the dumper's own defaults
            unless named here).
        path_for: name -> file path; None uses ``<cell_dir>/<name>.json``.
        make_dirs: create ``cell_dir`` when missing.
        printer: one line per cell; None stays quiet.
    """
    if on_error not in ("warn", "raise", "skip"):
        raise ValueError("on_error must be 'warn', 'raise' or 'skip'")
    if dump_fn is None:
        dump_fn = snapshot_dumper
    decide = verdict_fn or snapshot_verdict
    notes = dict(EMPTY_NOTES if empty_notes is None else empty_notes)
    if make_dirs:
        os.makedirs(cell_dir, exist_ok=True)

    resolve_path = path_for or (lambda name: snapshot_path(name, cell_dir))
    wanted = None if only is None else {str(n) for n in only}
    skipped = {str(n) for n in skip}
    forced = {str(n) for n in refresh}
    extra = dict(dump_kwargs or {})

    out: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        name = cell.name
        path = resolve_path(name)
        if (wanted is not None and name not in wanted) or name in skipped or (enabled_only and not cell.snapshot):
            out[name] = {"status": SKIPPED, "path": path, "n_records": None, "n_linked": None}
            continue

        existed = os.path.exists(path)
        held: Optional[Dict[str, Any]] = None
        if existed and reuse_min_linked is not None and name not in forced:
            held = (counts_fn or snapshot_counts)(path)
        thin = held is not None and int(held.get("n_linked") or 0) < int(reuse_min_linked or 0)
        if existed and reuse_existing and name not in forced and not thin:
            out[name] = {"status": REUSED, "path": path, "n_records": None, "n_linked": None}
            if printer:
                printer(f"  {name}: snapshot reused, untouched ({path})")
            continue
        if thin and printer:
            printer(f"  {name}: on-disk snapshot holds {held['n_linked']}/{held['n_records']} "
                    f"linked, below {reuse_min_linked}, re-dumping")

        target = path + temp_suffix if (existed and refresh_via_temp) else path
        try:
            result = dump_fn(
                cell=name,
                runs=cell.runs,
                benchmark_path=benchmark_path,
                path=target,
                start=cell.condition.start,
                end=cell.condition.end,
                conversation_ids=cell.condition.conversation_ids,
                query_ids=cell.query_ids,
                overwrite=True,
                **extra,
            )
        except Exception as exc:  # DB down, gold that matched nothing, bad benchmark
            if on_error == "raise":
                raise
            if target != path and os.path.exists(target):
                os.remove(target)
            out[name] = {
                "status": FAILED, "path": path, "n_records": None, "n_linked": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if printer and on_error == "warn":
                printer(f"  {name}: snapshot failed ({type(exc).__name__}: {exc})")
            continue

        n_records = int(result.get("n_records") or 0)
        n_linked = int(result.get("n_linked") or 0)
        keep, reason = decide(
            n_records, n_linked,
            min_records=min_records, min_linked=min_linked,
            keep_empty=keep_empty, keep_unlinked=keep_unlinked,
        )
        if not keep:
            if os.path.exists(target):
                os.remove(target)
            out[name] = {
                "status": EMPTY, "path": path, "n_records": n_records, "n_linked": n_linked,
                "reason": reason, "kept_previous": existed and target != path,
            }
            if printer:
                note = notes.get(reason, reason)
                fields = {"cell": name, "path": path,
                          "n_records": n_records, "n_linked": n_linked}
                printer(f"  {name}: {note.format_map(_Blanks(fields))}")
            continue

        if target != path:
            os.replace(target, path)
        out[name] = {"status": WRITTEN, "path": path, "n_records": n_records, "n_linked": n_linked}
        if printer:
            printer(f"  {name}: snapshot written, {n_linked}/{n_records} linked ({path})")
    return out


def measured_cost_inputs(
    cell_dir: str,
    names: Sequence[str],
    *,
    meta_key: str = COST_META_KEY,
    path_for: Optional[Callable[[str], str]] = None,
    on_error: str = "skip",
    printer: Optional[Callable[[str], None]] = None,
    label: str = "power measurement",
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    One measured block per cell dump, ``{cell: cost_inputs or None}``.

    Read off the snapshot, not re-measured, so a cell keeps the number taken
    while it was live even after the power log rotates or the ingest is
    overwritten. ``meta_key`` picks which measurement: watts under
    ``cost_inputs_measured``, ingest hours under ``ingest_measured``, counted
    serving tokens under ``tokens_measured``.

    Args:
        meta_key: field the dumper writes the measurement under.
        path_for: name -> snapshot path; None uses ``<cell_dir>/<name>.json``.
        on_error: ``skip`` treats an unreadable file as unmeasured.
        printer: one line per unreadable file; None stays silent.
        label: what an unreadable file is called in that line.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error must be 'skip' or 'raise'")
    resolve = path_for or (lambda name: snapshot_path(name, cell_dir))
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in names:
        path = resolve(name)
        if not os.path.exists(path):
            out[name] = None
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            out[name] = (payload.get("meta") or {}).get(meta_key)
        except Exception as exc:
            if on_error == "raise":
                raise
            out[name] = None
            if printer:
                printer(f"  {name}: {label} unreadable ({type(exc).__name__}: {exc})")
    return out


# ----------------------------------------------------------------- 2. load
def load_cells(
    conditions: ConditionsFile,
    *,
    cell_dir: str,
    benchmark_path: Optional[str] = None,
    source: str = "snapshot",
    live_fallback: bool = True,
    strict_schema: bool = True,
    expect_cell_name: bool = True,
    drop_empty: bool = True,
    min_linked: int = 0,
    drop_unlisted: bool = True,
    only: Optional[Sequence[str]] = None,
    skip: Sequence[str] = (),
    queries: Optional[Sequence[Any]] = None,
    read_fn: Optional[Callable[..., List[QueryRecord]]] = None,
    extract_fn: Callable[..., List[QueryRecord]] = extract_condition,
    path_for: Optional[Callable[[str], str]] = None,
    printer: Optional[Callable[[str], None]] = print,
) -> Tuple[Dict[str, List[QueryRecord]], Dict[str, str]]:
    """Records per cell, from the snapshots first and the live logs otherwise.

    Returns ``({cell: records}, {cell: "snapshot"|"live"})``.

    Args:
        source: ``snapshot`` prefers the file on disk, ``live`` always reads
            Postgres and treats the files as archive only.
        live_fallback: a cell with no snapshot is extracted from the DB. False
            scores only what was snapshotted.
        strict_schema / expect_cell_name: reject a file from another schema
            version or a file whose ``cell`` is not the cell it was loaded as.
        drop_empty: a cell with no records is left out of the report instead of
            printing a table of nothing.
        min_linked: linked answers a cell must reach to be scored. A dump holds
            one record per benchmark query whether or not it reached an answer,
            so records alone do not prove a cell ran; 1 keeps a stale or
            never-asked file out of the report rather than scoring a row whose
            every metric is empty. 0 scores whatever loaded.
        drop_unlisted: keep only the names the conditions file placed in the
            grid or the off-grid list (a ``drop`` policy name goes nowhere).
        only / skip: restrict the set.
        queries: preloaded benchmark; None loads ``benchmark_path``.
        read_fn / extract_fn: snapshot reader and live extractor, injectable.
        path_for: name -> snapshot path.
        printer: one line per cell; None stays quiet.
    """
    if source not in ("snapshot", "live"):
        raise ValueError("source must be 'snapshot' or 'live'")
    if read_fn is None:
        from .cells_app import read_cell as read_fn

    resolve_path = path_for or (lambda name: snapshot_path(name, cell_dir))
    listed = set(conditions.grid) | set(conditions.offgrid)
    wanted = None if only is None else {str(n) for n in only}
    skipped = {str(n) for n in skip}

    todo: List[DeclaredCell] = [
        c for c in conditions.cells
        if (wanted is None or c.name in wanted) and c.name not in skipped
        and (not drop_unlisted or c.name in listed)
    ]
    needs_live = [
        c for c in todo
        if source == "live" or not os.path.exists(resolve_path(c.name))
    ]
    all_queries = None
    if needs_live and live_fallback:
        all_queries = list(queries) if queries is not None else load_benchmark(
            benchmark_path or settings.benchmark_path
        )

    cells: Dict[str, List[QueryRecord]] = {}
    sources: Dict[str, str] = {}
    for cell in todo:
        name = cell.name
        path = resolve_path(name)
        records: Optional[List[QueryRecord]] = None
        if source == "snapshot" and os.path.exists(path):
            records = read_fn(
                path,
                expect_cell=name if expect_cell_name else None,
                strict_schema=strict_schema,
            )
            origin = "snapshot"
        elif live_fallback:
            picked = all_queries or []
            if cell.query_ids:
                keep = {str(q) for q in cell.query_ids}
                picked = [q for q in picked if q.id in keep]
            records = extract_fn(picked, cell.condition)
            origin = "live"
        else:
            if printer:
                printer(f"  {name}: no snapshot and live reads are off, skipped")
            continue

        linked = sum(1 for r in records if r.linked)
        if not records and drop_empty:
            if printer:
                printer(f"  {name}: 0 rows ({origin}), not scored")
            continue
        if linked < min_linked:
            if printer:
                printer(f"  {name}: {linked}/{len(records)} queries linked ({origin}), "
                        f"below {min_linked}, not scored")
            continue
        cells[name] = records
        sources[name] = origin
        if printer:
            printer(f"  {name}: {linked}/{len(records)} queries linked to logged answers ({origin})")
    return cells, sources


# ------------------------------------------------------------------ 3. run
def run_experiment(
    conditions_path: str = "conditions.yaml",
    *,
    benchmark_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    cell_dir: Optional[str] = None,
    cell_subdir: str = CELL_SUBDIR,
    # --- snapshots ---
    snapshot: bool = True,
    reuse_existing: bool = True,
    reuse_min_linked: Optional[int] = 1,
    refresh: Sequence[str] = (),
    refresh_unknown: str = "raise",
    only: Optional[Sequence[str]] = None,
    skip: Sequence[str] = (),
    source: str = "snapshot",
    live_fallback: bool = True,
    score_min_linked: int = 1,
    keep_empty: bool = False,
    min_records: int = 1,
    min_linked: int = 1,
    keep_unlinked: bool = False,
    verdict_fn: Optional[Callable[..., Tuple[bool, str]]] = None,
    empty_notes: Optional[Dict[str, str]] = None,
    on_snapshot_error: str = "warn",
    dump_kwargs: Optional[Dict[str, Any]] = None,
    # --- conditions file ---
    reference: Optional[str] = None,
    conditions_kwargs: Optional[Dict[str, Any]] = None,
    # --- scoring ---
    score: bool = True,
    unknown: str = "grid",
    metrics: Optional[Sequence[str]] = None,
    n_contrasts: Optional[int] = None,
    offgrid_reference: Optional[str] = None,
    offgrid_n_contrasts: int = 1,
    min_offgrid_linked: int = 1,
    include_offgrid_cells: bool = True,
    require_offgrid: bool = False,
    # --- CFCA ---
    with_cfca: bool = True,
    cfca_field: str = CFCA_FIELD,
    cfca_kwargs: Optional[Dict[str, Any]] = None,
    per_answer_cost_kwargs: Optional[Dict[str, Any]] = None,
    measured_power_kwargs: Optional[Dict[str, Any]] = None,
    require_measured_power: Any = "scored",
    power_config_on_missing: str = "raise",
    power_config_kwargs: Optional[Dict[str, Any]] = None,
    with_measured_ingest: bool = True,
    measured_ingest_kwargs: Optional[Dict[str, Any]] = None,
    with_measured_tokens: bool = True,
    measured_tokens_kwargs: Optional[Dict[str, Any]] = None,
    cost_meta_key: str = COST_META_KEY,
    ingest_meta_key: str = INGEST_META_KEY,
    tokens_meta_key: str = TOKENS_META_KEY,
    serving_meta_key: str = SERVING_META_KEY,
    # --- chunking policy check ---
    with_policy_check: bool = True,
    policy_check_kwargs: Optional[Dict[str, Any]] = None,
    ingest_detail_meta_key: str = INGEST_DETAIL_META_KEY,
    # --- output ---
    write_json: bool = True,
    write_md: bool = True,
    json_name: str = JSON_NAME,
    md_name: str = MD_NAME,
    indent: int = 2,
    encoding: str = "utf-8",
    # --- post-run check ---
    verify: bool = True,
    verify_kwargs: Optional[Dict[str, Any]] = None,
    verify_max_rows: Optional[int] = 40,
    printer: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """conditions.yaml -> snapshots -> report.json + report.md -> math check.

    Returns ``{conditions, snapshots, sources, report, cfca, paths, verify,
    ok}``; ``ok`` is False only when the math check failed.

    Args:
        conditions_path: the one file that declares the experiment.
        benchmark_path / out_dir: None uses ``settings``.
        cell_dir: snapshot directory; None uses ``<out_dir>/cells``.
        snapshot: write the per-cell JSON files before scoring. Off scores the
            live database only, which cannot see a window whose evidence rows
            are already deleted.
        score: False stops after the snapshots and returns them, which is the
            "I just finished a cell in the app, save it" call.
        reuse_existing / refresh / keep_empty / on_snapshot_error / dump_kwargs:
            passed to `snapshot_cells`. ``refresh`` is a list of cell names, and
            an empty one refreshes nothing: the bare-flag rule that makes a
            refresh inherit ``only`` lives in `resolve_refresh`, which the CLI
            calls before it gets here.
        refresh_unknown: what a refreshed name the conditions file never declared
            does. ``raise`` (default), or ``drop``, or ``keep`` to allow a name
            resolved elsewhere. A typo would otherwise refresh nothing in
            silence, see `resolve_refresh`.
        reuse_min_linked: an existing snapshot holding fewer linked queries than
            this is re-dumped instead of reused, which is what retires a
            0-linked file written before that rule existed. The old file stays
            until the re-dump has rows. None reuses whatever is on disk, and
            ``keep_empty`` forces None, since it means "keep what was captured".
        min_records / min_linked / keep_unlinked / verdict_fn / empty_notes: the
            keep-or-delete rule for a fresh dump, see `snapshot_verdict`.
            ``min_linked=1`` refuses to write a cell whose benchmark questions
            were never asked in the app, so no run reuses a 0-linked file.
        only / skip: restrict both the snapshot and the scoring set.
        source / live_fallback: passed to `load_cells`.
        score_min_linked: linked answers a loaded cell must hold to be scored,
            passed to `load_cells` as ``min_linked``. 1 keeps a cell whose
            questions were never asked out of the report, including one whose
            snapshot predates that rule; 0 scores whatever loaded, and
            ``keep_empty`` forces 0 so that flag means one thing everywhere.
        reference: override the reference cell the yaml names.
        conditions_kwargs: forwarded to `conditions.load_conditions_file`
            (strictness, ``on_unbounded``, ``assume_tz`` and the rest).
        unknown / metrics / n_contrasts / offgrid_reference /
        offgrid_n_contrasts / min_offgrid_linked / include_offgrid_cells /
        require_offgrid: forwarded to `report.evaluate_experiment_split`;
            ``metrics=None`` keeps its default column set.
        measured_power_kwargs: overrides for `conditions.apply_measured_power`,
            used when the file names a ``device:``.
        require_measured_power: which cells must carry measured watts, since a
            dump keeps a cell whose meter failed rather than lose its evidence
            and the refusal to price belongs here instead. ``scored`` demands a
            measurement for every scored cell that has a cost block, ``none``
            prices whatever the yaml declared, and a list names the cells.
        power_config_on_missing: what to do when `power_config_verdict` finds
            the file cannot measure watts at all. ``raise`` stops before the
            dump, since the sitting cannot be retaken and a silent fallback to
            the typed watts is the failure this guards. ``warn`` prints and
            carries on, ``ignore`` is the old silent behaviour.
        power_config_kwargs: overrides for that check (``require_power_block``,
            the key names, the hint text).
        with_measured_ingest / measured_ingest_kwargs: fold each cell's measured
            ingest hours into its cost block, `ingest_cost.apply_measured_ingest`.
            The hours come off the snapshot written while that ingest was live,
            so the chunking correction can overwrite the database afterwards.
            Off prices whatever ``onetime_gpu_hours`` the yaml declared.
        with_measured_tokens / measured_tokens_kwargs: fold each cell's counted
            serving tokens into its cost block, `cfca_metric.apply_measured_tokens`,
            so ``query_in_mtok`` and ``query_out_mtok`` come off the app's own
            ``completion_logs`` rows rather than a typed estimate. Off prices
            whatever the yaml declared.
        with_policy_check / policy_check_kwargs: read the chunking policy label
            each cell's ingest actually ran under, off the block its dump
            already carries, and say so per cell. It edits nothing and prices
            nothing: a cell whose ``CHUNKING_MODE`` was never flipped ingests
            under another cell's builder, lands in that cell's cost bucket, and
            this is the line that names it. Cells declare the label they expect
            with ``expect_policy:``; a cell that declares none is not judged.
            `ingest_cost.check_ingest_policies` owns every knob, ``match`` and
            the four ``on_*`` settings among them.
        ingest_detail_meta_key: snapshot ``meta`` field holding the whole ingest
            measurement, which is where the label survives.
        cost_meta_key / ingest_meta_key / tokens_meta_key / serving_meta_key: snapshot ``meta``
            fields the measurements are read back from. The third carries every
            latency sum measured for a cell, priced and unpriced, so the report
            shows the full serving cost and the generation-only cost side by
            side under ``serving_seconds``.
        with_cfca / cfca_field / cfca_kwargs: GBP CFCA from the ``cost:`` blocks,
            stored in the report under ``cfca_field`` and appended to the
            markdown. Off leaves the report exactly as an older run wrote it.
        per_answer_cost_kwargs: `per_answer_cost_gbp` overrides, the FX rate
            ``gbp_usd`` among them. The cost blocks are measured before the
            cells are scored, so the pound figure per answer is what the
            per-cell CFCA column and its bootstrap interval are built from.
        write_json / write_md / json_name / md_name / indent / encoding: output.
        verify / verify_kwargs / verify_max_rows: post-run math check; rows None
            or 0 lists every assertion.
        printer: progress sink; None runs silent.
    """
    say = printer or (lambda _msg: None)
    out_dir = out_dir or settings.output_dir
    cells_dir = cell_dir or cell_dir_for(out_dir, subdir=cell_subdir)
    bench = benchmark_path or settings.benchmark_path

    parse_kwargs: Dict[str, Any] = {"printer": printer}
    parse_kwargs.update(conditions_kwargs or {})
    cf = load_conditions_file(conditions_path, **parse_kwargs)
    ref = reference or cf.reference
    if ref not in cf.grid:
        raise ValueError(f"reference {ref!r} is not one of the grid cells {list(cf.grid)}")
    say(f"Loaded {conditions_path}: {len(cf.cells)} conditions "
        f"(grid={list(cf.grid)}, off-grid={list(cf.offgrid) or 'none'}), reference cell={ref}")

    refresh, _refresh_tag, refresh_note = resolve_refresh(
        list(refresh) or None,
        known_cells=[c.name for c in cf.cells],
        bare_without_only="off",
        unknown_cell=refresh_unknown,
    )
    if refresh_note:
        say(refresh_note)

    dump_kwargs = dict(dump_kwargs or {})
    power_ok, _power_tag, power_note = power_config_verdict(
        cf.device,
        power=cf.power,
        require_measured_power=require_measured_power,
        metered_cells=list(cf.grid) + list(cf.offgrid),
        conditions_path=conditions_path,
        **(power_config_kwargs or {}),
    )
    if not power_ok:
        if power_config_on_missing == "raise":
            raise ValueError(power_note)
        if power_config_on_missing == "warn":
            say(power_note)
        elif power_config_on_missing != "ignore":
            raise ValueError("power_config_on_missing must be 'raise', 'warn' or "
                             f"'ignore', got {power_config_on_missing!r}")
    if cf.device is not None:
        dump_kwargs.setdefault("device", cf.device)
        dump_kwargs.setdefault("power_printer", printer)
        if cf.power:
            dump_kwargs.setdefault("power", dict(cf.power))
        say(f"Power: device={cf.device!r}, measured per cell from "
            f"{(cf.power or {}).get('log_path', 'out/power/power.csv')}")
    dump_kwargs.setdefault("ingest_printer", printer)
    if cf.ingest:
        dump_kwargs.setdefault("ingest", dict(cf.ingest))
    if dump_kwargs.get("measure_ingest", True):
        ing = dict(cf.ingest or {})
        say(f"Ingest: chunking measured per cell off the database, "
            f"scope={ing.get('scope', 'corpus')}, "
            f"basis={ing.get('basis', 'lifecycle')}, "
            f"tokens={ing.get('tokens_total', 'chars/4.0 estimate')}, "
            f"hours={ing.get('hours_source', 'span')}")
        if ing.get("basis", "lifecycle") == "lifecycle":
            say("  note: the lifecycle basis spans documents.created_at -> "
                "documents.updated_at, and a re-ingest writes chunk rows only, so "
                "a rebuild leaves it unchanged. Use basis: chunks with "
                "tokens_total: index_build_logs to price a re-chunking")

    snapshots: Dict[str, Dict[str, Any]] = {}
    if snapshot:
        say(f"Snapshots in {cells_dir}:")
        snapshots = snapshot_cells(
            cf.cells,
            cell_dir=cells_dir,
            benchmark_path=bench,
            reuse_existing=reuse_existing,
            reuse_min_linked=None if keep_empty else reuse_min_linked,
            refresh=refresh,
            only=only,
            skip=skip,
            keep_empty=keep_empty,
            min_records=min_records,
            min_linked=min_linked,
            keep_unlinked=keep_unlinked,
            verdict_fn=verdict_fn,
            empty_notes=empty_notes,
            on_error=on_snapshot_error,
            dump_kwargs=dump_kwargs,
            printer=printer,
        )

    policy_check: Optional[Dict[str, Any]] = None
    if with_policy_check and cf.expect_policy:
        judged = sorted(cf.expect_policy)
        if only:
            judged = [n for n in judged if n in set(only)]
        if judged:
            details = measured_cost_inputs(
                cells_dir, judged, meta_key=ingest_detail_meta_key,
                printer=printer, label="ingest detail",
            )
            blocks = {
                name: ingest_policy_block(detail)
                for name, detail in details.items() if detail is not None
            }
            unread = [n for n in judged if n not in blocks]
            say("Chunking policy per cell, the bucket every cost query groups by:")
            policy_ok, verdicts, faults = check_ingest_policies(
                blocks, cf.expect_policy, printer=printer,
                **(policy_check_kwargs or {}),
            )
            for name in unread:
                say(f"  ingest policy: {name} carries no ingest measurement, "
                    f"expected {cf.expect_policy[name]}. Dump it against a reachable "
                    f"database, with tokens_total: index_build_logs in the ingest block")
            policy_check = {
                "ok": bool(policy_ok and not unread),
                "verdicts": verdicts,
                "faults": faults,
                "unread": unread,
                "expected": dict(cf.expect_policy),
                "blocks": blocks,
            }

    if not score:
        return {
            "conditions": cf, "snapshots": snapshots, "sources": {}, "report": None,
            "cfca": None, "paths": {"cell_dir": cells_dir}, "verify": None,
            "policy": policy_check, "ok": True,
        }

    say("Cells:")
    extracted, sources = load_cells(
        cf,
        cell_dir=cells_dir,
        benchmark_path=bench,
        source=source,
        live_fallback=live_fallback,
        min_linked=0 if keep_empty else score_min_linked,
        only=only,
        skip=skip,
        printer=printer,
    )
    partial = ref not in extracted
    if partial and not extracted:
        raise ValueError(
            f"reference cell {ref!r} reached no scored answers; loaded {sorted(extracted)}. "
            "Ask its benchmark questions in the app, or point the cell at the chat "
            "that already holds them, then re-run this command."
        )

    cost_inputs: Dict[str, Dict[str, float]] = {}
    cost_provenance: Dict[str, Any] = {}
    per_answer_cost: Dict[str, float] = dict(cf.per_answer_cost)
    if with_cfca and cf.cost_inputs:
        cost_inputs = cf.cost_inputs
        if cf.device is not None:
            say("Measured power into the cost blocks:")
            if isinstance(require_measured_power, str):
                if require_measured_power == "scored":
                    required = [n for n in sorted(cost_inputs) if n in extracted]
                elif require_measured_power == "none":
                    required = []
                else:
                    raise ValueError("require_measured_power must be 'scored', 'none' "
                                     f"or a list of cells, got {require_measured_power!r}")
            else:
                required = list(require_measured_power)
            power_kwargs: Dict[str, Any] = {"require_measured": required}
            power_kwargs.update(measured_power_kwargs or {})
            cost_inputs, power_provenance = apply_measured_power(
                cost_inputs,
                measured_cost_inputs(cells_dir, sorted(cost_inputs), meta_key=cost_meta_key),
                cost_defaults=cf.cost_defaults,
                printer=printer,
                **power_kwargs,
            )
            cost_provenance["power_provenance"] = power_provenance
            cost_provenance["serving_seconds"] = measured_cost_inputs(
                cells_dir, sorted(cost_inputs), meta_key=serving_meta_key,
                label="serving seconds",
            )
        if with_measured_ingest:
            measured_ingest = measured_cost_inputs(
                cells_dir,
                sorted(cost_inputs),
                meta_key=ingest_meta_key,
                printer=printer,
                label="ingest measurement",
            )
            if any(measured_ingest.values()):
                say("Measured ingest into the cost blocks:")
                cost_inputs, ingest_provenance = apply_measured_ingest(
                    cost_inputs,
                    measured_ingest,
                    printer=printer,
                    **(measured_ingest_kwargs or {}),
                )
                cost_provenance["ingest_provenance"] = ingest_provenance
        if with_measured_tokens:
            measured_tokens = measured_cost_inputs(
                cells_dir,
                sorted(cost_inputs),
                meta_key=tokens_meta_key,
                printer=printer,
                label="serving token count",
            )
            if any(measured_tokens.values()):
                say("Counted serving tokens into the cost blocks:")
                cost_inputs, token_provenance = apply_measured_tokens(
                    cost_inputs,
                    measured_tokens,
                    printer=printer,
                    **(measured_tokens_kwargs or {}),
                )
                cost_provenance["token_provenance"] = token_provenance
        per_answer_cost = per_answer_cost_gbp(
            cost_inputs,
            usd_cost=cf.per_answer_cost,
            **(per_answer_cost_kwargs or {}),
        )
        say("Cost per answer scored in GBP: "
            + ", ".join(f"{n}={per_answer_cost[n]:.6f}" for n in sorted(per_answer_cost)))

    if partial:
        say(f"  reference cell {ref!r} has no scored answers yet: writing a "
            f"partial report over {', '.join(sorted(extracted))}. Score {ref!r} "
            "and re-run this command for the delta tables.")
        report = evaluate_partial_split(
            extracted,
            reference=ref,
            per_answer_cost=per_answer_cost,
            grid_names=cf.grid,
            offgrid_names=cf.offgrid,
            unknown=unknown,
            printer=printer,
        )
    else:
        split_kwargs: Dict[str, Any] = dict(
            reference=ref,
            per_answer_cost=per_answer_cost,
            grid_names=cf.grid,
            offgrid_names=cf.offgrid,
            unknown=unknown,
            n_contrasts=n_contrasts,
            offgrid_reference=offgrid_reference,
            offgrid_n_contrasts=offgrid_n_contrasts,
            min_offgrid_linked=min_offgrid_linked,
            include_offgrid_cells=include_offgrid_cells,
            require_offgrid=require_offgrid,
        )
        if metrics is not None:
            split_kwargs["metrics"] = tuple(metrics)
        report = evaluate_experiment_split(extracted, **split_kwargs)
    report["sources"] = sources
    report.update(cost_provenance)
    if snapshots:
        report["snapshots"] = {k: v.get("status") for k, v in snapshots.items()}
    if report.get("offgrid"):
        say(f"  off-grid (not scored in the 2x2): {', '.join(report['offgrid'])}")
    if partial:
        say(f"  no grid contrast yet: the Bonferroni family starts when {ref!r} is scored")
    else:
        say(f"  grid contrasts in the Bonferroni family: {report['n_contrasts']}")

    cfca_block: Optional[Dict[str, Any]] = None
    if with_cfca and cost_inputs:
        cfca_block = cfca_for_cells(
            cost_inputs,
            p_hat_from_report(report),
            reference=ref,
            order=[n for n in report.get("conditions") or [] if n in cost_inputs],
            **(cfca_kwargs or {}),
        )
        report[cfca_field] = cfca_block

    if policy_check is not None:
        report["ingest_policy"] = policy_check
    paths: Dict[str, str] = {"cell_dir": cells_dir}
    if write_json or write_md:
        os.makedirs(out_dir, exist_ok=True)
    if write_json:
        paths["json"] = os.path.join(out_dir, json_name)
        with open(paths["json"], "w", encoding=encoding) as fh:
            json.dump(report, fh, indent=indent, default=str)
        say(f"\nWrote {paths['json']}")
    if write_md:
        paths["md"] = os.path.join(out_dir, md_name)
        text = render_markdown(report)
        if cfca_block:
            text = text.rstrip("\n") + "\n\n" + render_cfca_markdown(cfca_block)
        with open(paths["md"], "w", encoding=encoding) as fh:
            fh.write(text)
        say(f"Wrote {paths['md']}")

    checked = None
    if verify:
        checked = verify_report(report, **(verify_kwargs or {}))
        say("\n" + render_verify_text(checked, max_rows=verify_max_rows))
    return {
        "conditions": cf,
        "snapshots": snapshots,
        "sources": sources,
        "report": report,
        "cfca": cfca_block,
        "paths": paths,
        "verify": checked,
        "policy": policy_check,
        "ok": True if checked is None else bool(checked.get("ok")),
    }
