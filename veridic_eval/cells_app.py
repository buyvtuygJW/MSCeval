"""
Cell dump, read back, and result rendering in one file. Launch it with
``veridic-eval view`` (which owns the streamlit invocation and its flags).

1. ``dump_cell``    pulls one cell's records out of the app Postgres and writes
                    them to a JSON file, with the gold evidence matched against the
                    ingest that is live at the moment of the dump.
2. ``read_cell``    reads that JSON back into ``QueryRecord`` objects.
3. ``render_cells`` computes metrics per cell and deltas against a chosen
                    reference cell, from the JSON files alone.

The dump is needed because ``message_evidence.chunk_id`` is ``ON DELETE
CASCADE`` to ``chunks``, which cascades from ``documents``: deleting a document
to re-ingest it under the other chunker deletes the served evidence rows of
every cell already run. Dump a cell as soon as its conversations are done.

``prebaseline`` is the default cell to dump: the app as it stands, RCTS chunking
not yet corrected and the base model unadapted, so it sits below the grid rather
than in it. ``control`` is the 2x2 reference cell every delta is measured
against, and it needs the chunking correction first. Names live in cells.py.

In ``full`` mode ``render_cells`` scores the four grid cells and reports
``prebaseline`` against the reference in a separate uncorrected block, so loading
the before-window sizes the baseline construction without touching the 2x2
numbers or their Bonferroni family. Render ``{prebaseline, control}`` alone with
``reference=prebaseline`` and the before-window becomes the grid reference.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from veridic_eval import provenance  # column -> file:line, function, equation
from veridic_eval.benchmark import load_benchmark
from veridic_eval.cells import (
    CELL_SUBDIR,
    DEFAULT_DUMP_CELL,
    DEFAULT_REFERENCE,
    GRID_ORDER,
    OFFGRID_ORDER,
    cell_dir_for,
    cell_file_path,
    cell_names,
    contrast_key,
    deltas_key,
    deltas_of,
    offgrid_deltas_of,
    reference_from_mapping,
    resolve_reference,
)
from veridic_eval.cfca_metric import serving_token_totals
from veridic_eval.charts import cost_chart, delta_ci_chart, levels_chart
from veridic_eval.conditions import (  # yaml parsing lives there
    load_conditions_file,
    parse_timestamp,
)
from veridic_eval.config import DEFAULT_RUN_ID, Condition, settings
from veridic_eval.extract import (
    QueryRecord,
    ServedEvidence,
    extract_cell_runs,
    extract_condition,
)
from veridic_eval.gold_evidence import apply_gold_to_records
from veridic_eval.report import evaluate_experiment_split, render_markdown
from veridic_eval.retrieval_eval import evaluate_retrieval
from veridic_eval.warmup import (
    BASE_SERIES,
    WARM_COLUMN,
    cost_columns,
    warm_only_caption,
    warm_only_columns,
    warm_only_rows,
)

SCHEMA = "veridic-eval/cell-dump/1"
DEFAULT_CELL = DEFAULT_DUMP_CELL
DEFAULT_CELL_DIR = cell_dir_for()


def eval_view_options(
    *,
    out_dir: Optional[str] = None,
    benchmark_path: Optional[str] = None,
    cell_subdir: str = CELL_SUBDIR,
    v2_suffix: str = "-v2",
    v2_benchmark: str = "./benchmarkv2.yaml",
    report_name: str = "report.json",
    labels: Tuple[str, str] = ("v1", "v2"),
    require_power: str = "scored",
    strict_conditions: bool = True,
) -> Dict[str, Dict[str, str]]:
    """The v1 and v2 markings as one pick: dump dir, benchmark, report, command.

    v1 is the ten-query marking; v2 is the same ten plus ``q021`` and ``q022``,
    scored from its own dump directory and its own out directory because scoring
    reads a dump verbatim.

    Args:
        out_dir: v1 report directory; None reads ``settings.output_dir``.
        benchmark_path: v1 benchmark; None reads ``settings.benchmark_path``.
        cell_subdir: dump leaf inside an out directory.
        v2_suffix: what separates the v2 dump dir and out dir from v1's.
        v2_benchmark: the twelve-query benchmark YAML.
        report_name: file a scoring run writes into an out directory.
        labels: dropdown labels, v1 first, so a selectbox opens on v1.
        require_power: value for ``--require-measured-power`` in the printed
            command; "" drops the flag for a view that has no meter log.
        strict_conditions: print ``--strict-conditions`` in that command.

    Returns:
        ``{label: {"cell_dir", "benchmark_path", "report_path", "run_hint"}}``.
    """
    base = out_dir or settings.output_dir
    v1, v2 = labels
    out2 = f"{base}{v2_suffix}"
    cells2 = os.path.join(base, f"{cell_subdir}{v2_suffix}")
    gate = " ".join(
        p
        for p in (
            f"--require-measured-power {require_power}" if require_power else "",
            "--strict-conditions" if strict_conditions else "",
        )
        if p
    )
    return {
        v1: {
            "cell_dir": cell_dir_for(base, subdir=cell_subdir),
            "benchmark_path": benchmark_path or settings.benchmark_path,
            "report_path": os.path.join(base, report_name),
            "run_hint": f"veridic-eval run {gate}".rstrip(),
        },
        v2: {
            "cell_dir": cells2,
            "benchmark_path": v2_benchmark,
            "report_path": os.path.join(out2, report_name),
            "run_hint": (
                f"veridic-eval run --benchmark {v2_benchmark} "
                f"--cell-dir {cells2} --out {out2} {gate}"
            ).rstrip(),
        },
    }


# --------------------------------------------------------------------------
# 1. dump
# --------------------------------------------------------------------------


def _plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    return obj


def write_cell(
    records: Sequence[QueryRecord],
    path: str,
    *,
    cell: str,
    meta: Optional[Dict[str, Any]] = None,
    include_chunk_text: bool = True,
    overwrite: bool = True,
    indent: int = 2,
) -> str:
    """Write one cell's records to ``path`` as JSON and return the path.

    Args:
        cell: cell name stored in the file, checked on read.
        meta: whatever is needed to reproduce the cell (conversation ids,
            chunker settings, adapter path, gold resolution summary). Written
            as given.
        include_chunk_text: keep the full chunk text of each served evidence
            row. Faithfulness scoring needs it; False gives a small IR-only
            file and any context-based metric then sees empty text.
        overwrite: False raises instead of replacing an existing file.
        indent: JSON indent; None writes one line.
    """
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    rows: List[dict] = []
    for r in records:
        row = _plain(r)
        if not include_chunk_text:
            for ev in row.get("served_evidence", []):
                ev["chunk_text"] = ""
        rows.append(row)

    payload = {
        "schema": SCHEMA,
        "cell": cell,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "include_chunk_text": include_chunk_text,
        "n_records": len(rows),
        "n_linked": sum(1 for r in rows if r.get("linked")),
        "n_judged": sum(1 for r in rows if r.get("judged_chunk_ids")),
        "meta": meta or {},
        "records": rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=str)
    return path


def measured_meta(
    cell: str,
    records: Sequence[QueryRecord],
    *,
    kind: str,
    runner: Callable[..., Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    records_first: bool = True,
    pass_cell: bool = True,
    pass_printer: bool = True,
    measure: bool = True,
    on_error: str = "warn",
    printer: Optional[Callable[[str], None]] = print,
    require_records: bool = True,
    error_key: str = "error",
) -> Optional[Dict[str, Any]]:
    """
    One measurement of one cell, guarded, for the dumped file's ``meta``.

    The single place a per-cell measurement is called from, so every
    measurement fails the same way: a captured cell is written whatever the
    meter or the database did, because the evidence in a snapshot cannot be
    recovered once the next ingest cascades it away, while a measurement can
    always be retaken off the log.

    Args:
        kind: name in the warning line and nothing else, e.g. ``power``.
        runner: the measuring function, `measure_cell_serving_power` or
            `measure_cell_ingest`.
        options: the caller's yaml block, forwarded as keyword arguments.
        extra: keywords the dumper owns rather than the yaml, e.g. ``device``.
        records_first: True calls ``runner(records, ...)``, False calls
            ``runner(cell, records, ...)``, matching the two signatures.
        pass_cell: name the cell in the keywords, for the runner's own printing.
        pass_printer: hand `printer` to the runner as well.
        measure: False returns None, which prices the declared yaml number.
        on_error: ``raise`` propagates; ``warn`` and ``skip`` keep the dump and
            record the reason in place of the numbers.
        printer: one line on failure, and the runner's own sink; None silences.
        require_records: True measures nothing for a cell with no answers.
        error_key: key the reason is recorded under.
    """
    if on_error not in ("warn", "raise", "skip"):
        raise ValueError("on_error must be 'warn', 'raise' or 'skip'")
    if not measure or (require_records and not records):
        return None
    kwargs: Dict[str, Any] = dict(options or {})
    kwargs.update(extra or {})
    if pass_cell and records_first:
        kwargs.setdefault("cell", cell)
    if pass_printer:
        kwargs.setdefault("printer", printer)
    args: Tuple[Any, ...] = (records,) if records_first else (cell, records)
    try:
        return runner(*args, **kwargs)
    except Exception as exc:
        if on_error == "raise":
            raise
        out = {"cell": cell, error_key: f"{type(exc).__name__}: {exc}"}
        if on_error == "warn" and printer:
            printer(f"  {cell}: {kind} not measured ({out[error_key]})")
        return out


def measured_ingest_meta(
    cell: str,
    records: Sequence[QueryRecord],
    *,
    measure: bool = True,
    ingest: Optional[Dict[str, Any]] = None,
    on_error: str = "warn",
    printer: Optional[Callable[[str], None]] = print,
    require_records: bool = True,
) -> Optional[Dict[str, Any]]:
    """This cell's ingest priced off the live database, through `measured_meta`."""
    from veridic_eval.ingest_cost import measure_cell_ingest  # keep import local

    return measured_meta(
        cell,
        records,
        kind="ingest",
        runner=measure_cell_ingest,
        options=ingest,
        records_first=False,
        measure=measure,
        on_error=on_error,
        printer=printer,
        require_records=require_records,
    )


def cell_measurements(
    cell: str,
    records: Sequence[QueryRecord],
    *,
    device: Any = None,
    power: Optional[Dict[str, Any]] = None,
    measure_power: bool = True,
    power_per_run: bool = True,
    power_on_error: str = "warn",
    power_printer: Optional[Callable[[str], None]] = print,
    runs: Sequence[Any] = (),
    run_id_field: str = "run_id",
    ingest: Optional[Dict[str, Any]] = None,
    measure_ingest: bool = True,
    ingest_on_error: str = "warn",
    ingest_printer: Optional[Callable[[str], None]] = print,
    tokens: Optional[Dict[str, Any]] = None,
    measure_tokens: bool = True,
    tokens_printer: Optional[Callable[[str], None]] = print,
    power_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ingest_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    require_records: bool = True,
    per_run_key: str = "per_run",
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Both of a cell's measurements, watts and chunking, in one guarded call.

    Every dumper goes through here, so neither measurement can be wired one way
    in one dumper and another way in the other, and neither can cost a captured
    cell.

    Args:
        device / power / measure_power: the meter and its yaml block;
            ``device=None`` measures no watts and leaves the cost inputs typed.
        power_per_run / runs / run_id_field: also price each sitting separately
            when a cell was asked more than once.
        power_on_error / ingest_on_error: `measured_meta` error handling, per
            measurement, so a dead meter and a moved schema are separable. Both
            are checked here, before either measurement runs, so a typo raises
            on a cell that measures nothing too.
        power_printer / ingest_printer / tokens_printer: the three sinks; None
            silences one.
        measure_tokens / tokens: count this cell's serving tokens off the
            ``completion_logs`` rows already extracted, with
            `cfca_metric.serving_token_totals` overrides. No database read and
            no failure mode of its own, so it takes no ``on_error``.
        power_runner / ingest_runner: injected measuring functions, for a test
            without a log or a database.
        require_records: True measures nothing for a cell with no answers.
        per_run_key: key the per-sitting measurements are recorded under.

    Returns:
        ``{"power": meta or None, "ingest": meta or None, "tokens": counts or None}``.
    """
    for name, mode in (
        ("power_on_error", power_on_error),
        ("ingest_on_error", ingest_on_error),
    ):
        if mode not in ("warn", "raise", "skip"):
            raise ValueError(f"{name} must be 'warn', 'raise' or 'skip', got {mode!r}")
    runner = power_runner
    if runner is None and measure_power and device is not None:
        from veridic_eval.power_meter import (
            measure_cell_serving_power,  # local: streamlit stays off
        )

        runner = measure_cell_serving_power
    power_meta: Optional[Dict[str, Any]] = None
    if runner is not None and device is not None:
        power_meta = measured_meta(
            cell,
            records,
            kind="power",
            runner=runner,
            options=power,
            extra={"device": device},
            measure=measure_power,
            on_error=power_on_error,
            printer=power_printer,
            require_records=require_records,
        )
        if (
            power_meta is not None
            and not power_meta.get("error")
            and power_per_run
            and len(runs) > 1
        ):
            by_run: Dict[str, Any] = {}
            for cond in runs:
                run_id = str(getattr(cond, run_id_field, "") or "")
                rows = [
                    r for r in records if str(getattr(r, run_id_field, "")) == run_id
                ]
                if not rows:
                    continue
                one = measured_meta(
                    cell,
                    rows,
                    kind=f"power[{run_id}]",
                    runner=runner,
                    options=power,
                    extra={"device": device, "cell": f"{cell}#{run_id}"},
                    measure=measure_power,
                    on_error=power_on_error,
                    printer=power_printer,
                    require_records=require_records,
                )
                if one is not None:
                    by_run[run_id] = one
            if by_run:
                power_meta[per_run_key] = by_run

    ingest_meta = measured_ingest_meta(
        cell,
        records,
        measure=measure_ingest,
        ingest=ingest,
        on_error=ingest_on_error,
        printer=ingest_printer,
        require_records=require_records,
    )
    tokens_meta = (
        serving_token_totals(records, **(tokens or {})) if measure_tokens else None
    )
    if tokens_meta and tokens_printer:
        tokens_printer(f"  {cell} serving tokens: {tokens_meta['source']}")
    return {"power": power_meta, "ingest": ingest_meta, "tokens": tokens_meta}


def dump_cell(
    *,
    cell: str = DEFAULT_CELL,
    benchmark_path: Optional[str] = None,
    path: Optional[str] = None,
    cell_dir: str = DEFAULT_CELL_DIR,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    conversation_ids: Sequence[str] = (),
    include_deleted_conversations: bool = False,
    query_ids: Optional[Sequence[str]] = None,
    resolve_gold: bool = True,
    gold_default_document: Optional[str] = None,
    gold_require_all: bool = False,
    document_match: str = "exact",
    document_status: Optional[Sequence[str]] = ("completed",),
    match_mode: str = "containment_then_shingle",
    shingle_n: int = 5,
    min_shingle_coverage: float = 0.6,
    min_token_coverage: float = 0.9,
    page_window: Optional[int] = None,
    page_required: bool = False,
    max_chunks_per_item: int = 0,
    min_score: float = 0.0,
    tie_margin: float = 0.0,
    measure_ingest: bool = True,
    ingest: Optional[Dict[str, Any]] = None,
    ingest_on_error: str = "warn",
    ingest_printer: Optional[Callable[[str], None]] = print,
    measure_tokens: bool = True,
    tokens: Optional[Dict[str, Any]] = None,
    tokens_printer: Optional[Callable[[str], None]] = print,
    include_chunk_text: bool = True,
    overwrite: bool = True,
    indent: int = 2,
    meta: Optional[Dict[str, Any]] = None,
) -> dict:
    """Extract one cell from Postgres, resolve its gold, write the JSON file.

    Returns ``{path, cell, n_records, n_linked, n_judged, gold, ingest,
    records}``.

    Args:
        cell: cell name, e.g. ``prebaseline`` / ``control`` / ``chunk_opt`` /
            ``qdora`` / ``combined``.
        benchmark_path: benchmark YAML/JSON; defaults to ``settings``.
        path: output file; defaults to ``<cell_dir>/<cell>.json``.
        start, end, conversation_ids: the slice of the logs that is this cell,
            same meaning as in conditions.yaml.
        include_deleted_conversations: True also reads chats the app soft-deleted.
        query_ids: run only these benchmark ids (a partial cell, e.g. the 3
            documents ingested so far); None takes the whole benchmark.
        resolve_gold: map ``gold_evidence`` to this ingest's chunk ids. Off dumps
            every record unjudged, which skips IR and keeps only the answer-side
            metrics; there is no other source of qrels to fall back to.
        gold_default_document: document filename for gold entries that omit one.
        gold_require_all: raise if any gold text matches no chunk. Use it for the
            final run so gold that matched nothing is not scored as a miss.
        document_match: ``exact`` | ``casefold`` | ``basename`` | ``contains``.
        document_status: ``documents.status`` values to accept; None takes any.
        match_mode: ``containment`` | ``shingle`` | ``containment_then_shingle``
            | ``token``.
        shingle_n, min_shingle_coverage, min_token_coverage, page_window,
        page_required, max_chunks_per_item, min_score, tie_margin: passed
            straight to ``gold_evidence.match_gold_evidence``.
        measure_ingest / ingest / ingest_on_error / ingest_printer: price this
            cell's chunking off the live database; see ``measured_ingest_meta``.
        measure_tokens / tokens / tokens_printer: count the serving tokens the
            app logged for these answers into ``meta.tokens_measured``, which
            the run reads back for the two per-answer query terms; see
            ``cfca_metric.serving_token_totals``.
        include_chunk_text, overwrite, indent: see ``write_cell``.
        meta: extra keys merged into the file's ``meta`` block.
    """
    bench = benchmark_path or settings.benchmark_path
    queries = load_benchmark(bench)
    if query_ids:
        wanted = {str(q) for q in query_ids}
        queries = [q for q in queries if q.id in wanted]
        if not queries:
            raise ValueError(f"none of query_ids={sorted(wanted)} are in {bench}")

    cond = Condition(
        name=cell,
        start=start,
        end=end,
        conversation_ids=[str(c) for c in conversation_ids],
    )
    records = extract_condition(
        queries,
        cond,
        include_deleted_conversations=include_deleted_conversations,
    )

    gold_summary = None
    if resolve_gold and any(getattr(q, "gold_evidence", None) for q in queries):
        reports = apply_gold_to_records(
            {cell: records},
            queries,
            default_document=gold_default_document,
            require_all=gold_require_all,
            document_match=document_match,
            document_status=document_status,
            match_mode=match_mode,
            shingle_n=shingle_n,
            min_shingle_coverage=min_shingle_coverage,
            min_token_coverage=min_token_coverage,
            page_window=page_window,
            page_required=page_required,
            max_chunks_per_item=max_chunks_per_item,
            min_score=min_score,
            tie_margin=tie_margin,
        )
        gold_summary = reports[cell].summary()

    measured = cell_measurements(
        cell,
        records,
        device=None,
        measure_ingest=measure_ingest,
        ingest=ingest,
        ingest_on_error=ingest_on_error,
        ingest_printer=ingest_printer,
        measure_tokens=measure_tokens,
        tokens=tokens,
        tokens_printer=tokens_printer,
    )
    ingest_meta = measured["ingest"]
    tokens_meta = measured["tokens"]

    out_path = cell_file_path(cell, path=path, cell_dir=cell_dir)
    file_meta: Dict[str, Any] = {
        "benchmark_path": os.path.abspath(bench),
        "postgres_url": settings.postgres_url,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "conversation_ids": [str(c) for c in conversation_ids],
        "ingest": ingest_meta,
        "ingest_measured": (ingest_meta or {}).get("cost_inputs"),
        "tokens_measured": tokens_meta,
        "query_ids": sorted({q.id for q in queries}),
        "gold_match": gold_summary,
        "gold_settings": {
            "resolve_gold": resolve_gold,
            "document_match": document_match,
            "document_status": list(document_status) if document_status else None,
            "match_mode": match_mode,
            "shingle_n": shingle_n,
            "min_shingle_coverage": min_shingle_coverage,
            "min_token_coverage": min_token_coverage,
            "page_window": page_window,
            "page_required": page_required,
            "max_chunks_per_item": max_chunks_per_item,
            "min_score": min_score,
            "tie_margin": tie_margin,
        },
    }
    file_meta.update(meta or {})
    written = write_cell(
        records,
        out_path,
        cell=cell,
        meta=file_meta,
        include_chunk_text=include_chunk_text,
        overwrite=overwrite,
        indent=indent,
    )
    return {
        "path": written,
        "cell": cell,
        "n_records": len(records),
        "n_linked": sum(1 for r in records if r.linked),
        "n_judged": sum(1 for r in records if r.judged_chunk_ids),
        "gold": gold_summary,
        "ingest": ingest_meta,
        "tokens": tokens_meta,
        "records": records,
    }


def dump_cell_runs(
    cell: str,
    runs: Sequence[Condition],
    *,
    benchmark_path: Optional[str] = None,
    path: Optional[str] = None,
    cell_dir: str = DEFAULT_CELL_DIR,
    include_deleted_conversations: bool = False,
    query_ids: Optional[Sequence[str]] = None,
    run_ids: Optional[Sequence[str]] = None,
    only_runs: Optional[Sequence[str]] = None,
    skip_runs: Sequence[str] = (),
    reuse_existing_runs: bool = True,
    require_distinct: bool = True,
    require_all_runs: bool = False,
    keep_undeclared_runs: bool = False,
    resolve_gold: bool = True,
    gold_scope: str = "new",
    gold_default_document: Optional[str] = None,
    gold_require_all: bool = False,
    document_match: str = "exact",
    document_status: Optional[Sequence[str]] = ("completed",),
    match_mode: str = "containment_then_shingle",
    shingle_n: int = 5,
    min_shingle_coverage: float = 0.6,
    min_token_coverage: float = 0.9,
    page_window: Optional[int] = None,
    page_required: bool = False,
    max_chunks_per_item: int = 0,
    min_score: float = 0.0,
    tie_margin: float = 0.0,
    include_chunk_text: bool = True,
    strict_schema: bool = True,
    device: Any = None,
    measure_power: bool = True,
    power: Optional[Dict[str, Any]] = None,
    power_per_run: bool = True,
    power_on_error: str = "warn",
    power_printer: Optional[Callable[[str], None]] = print,
    measure_ingest: bool = True,
    ingest: Optional[Dict[str, Any]] = None,
    ingest_on_error: str = "warn",
    ingest_printer: Optional[Callable[[str], None]] = print,
    measure_tokens: bool = True,
    tokens: Optional[Dict[str, Any]] = None,
    tokens_printer: Optional[Callable[[str], None]] = print,
    overwrite: bool = True,
    indent: int = 2,
    meta: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Extract every sitting of one cell into one JSON file, runs inside.

    A re-dump queries only the runs the file does not already hold: rows already
    dumped were served by a database the next ingest has rewritten, so
    re-querying them comes back empty.

    Args:
        runs: the cell's windows, in the order asked, each with its ``run_id``.
        run_ids: override those ids, position by position.
        only_runs / skip_runs: restrict the ids this call may query.
        reuse_existing_runs: off re-queries every declared run.
        require_distinct: raise when two windows claim one run id.
        require_all_runs: raise when a declared run reached no logged answer.
        keep_undeclared_runs: keep records whose run id left the yaml.
        resolve_gold / gold_scope: ``new`` regolds only the runs queried here
            and leaves reused judgements frozen; ``all`` regolds everything.
        gold_default_document ... tie_margin: see `match_gold_evidence`.
        strict_schema: passed to `read_cell` when reusing a file.
        device: ``omen``, ``spark``, an inline profile, or None to skip
            measurement and leave the cost inputs to the yaml.
        measure_power / power / power_per_run: slice the power log to this cell,
            with `measure_cell_serving_power` overrides, and per sitting as well.
        power_on_error: ``warn`` keeps the dump when the log is absent or short
            and records the reason, so an unmetered sitting never costs the
            captured evidence; ``raise`` stops; ``skip`` stays silent. The run
            still refuses to price an unmeasured cell, in `apply_measured_power`.
        power_printer: the watts sink, the low-coverage warning and the failure
            line; None stays silent.
        measure_ingest / ingest: read this cell's ingest stamps off the database
            and price the chunking, with `measure_cell_ingest` overrides. The
            database only holds the newest ingest, so the number is captured
            here, while the cell is live.
        ingest_on_error: ``warn`` keeps the dump when the stamps cannot be read
            (absent columns, moved schema) and records the reason; ``raise``
            stops; ``skip`` stays silent.
        ingest_printer: one line naming the measured hours; None stays silent.
        measure_tokens / tokens / tokens_printer: count the serving tokens the
            app logged for these answers into ``meta.tokens_measured``, which
            the run reads back for the two per-answer query terms; see
            `cfca_metric.serving_token_totals`.
        include_chunk_text, overwrite, indent, meta: see `write_cell`.

    Returns:
        ``{path, cell, n_runs, run_ids, n_records, n_linked, n_judged, gold,
        runs, power, ingest, tokens, records}``.
    """
    if gold_scope not in ("new", "all"):
        raise ValueError("gold_scope must be 'new' or 'all'")
    if not runs:
        raise ValueError(f"cell '{cell}': no runs to dump")
    if run_ids is not None and len(run_ids) != len(runs):
        raise ValueError(f"run_ids has {len(run_ids)} entries for {len(runs)} runs")

    declared: List[Condition] = []
    for i, cond in enumerate(runs):
        rid = str(cond.run_id if run_ids is None else run_ids[i])
        if require_distinct and any(c.run_id == rid for c in declared):
            raise ValueError(f"cell '{cell}': run id {rid!r} declared twice")
        declared.append(replace(cond, name=cell, run_id=rid))

    wanted_runs = None if only_runs is None else {str(r) for r in only_runs}
    skipped_runs = {str(r) for r in skip_runs}

    bench = benchmark_path or settings.benchmark_path
    queries = load_benchmark(bench)
    if query_ids:
        wanted = {str(q) for q in query_ids}
        queries = [q for q in queries if q.id in wanted]
        if not queries:
            raise ValueError(f"none of query_ids={sorted(wanted)} are in {bench}")

    out_path = cell_file_path(cell, path=path, cell_dir=cell_dir)
    held: Dict[str, List[QueryRecord]] = {}
    header: Dict[str, Any] = {}
    if reuse_existing_runs and os.path.exists(out_path):
        prior, header = read_cell(
            out_path,
            expect_cell=cell,
            strict_schema=strict_schema,
            with_meta=True,
        )
        for rec in prior:
            held.setdefault(str(rec.run_id or DEFAULT_RUN_ID), []).append(rec)

    todo = [
        c
        for c in declared
        if c.run_id not in held
        and c.run_id not in skipped_runs
        and (wanted_runs is None or c.run_id in wanted_runs)
    ]
    fresh = (
        extract_cell_runs(
            queries,
            todo,
            include_deleted_conversations=include_deleted_conversations,
            require_distinct=require_distinct,
        )
        if todo
        else {}
    )

    records: List[QueryRecord] = []
    per_run: List[Dict[str, Any]] = []
    for cond in declared:
        rid = cond.run_id
        if rid in fresh:
            recs, status = fresh[rid], "written"
        elif rid in held:
            recs, status = held[rid], "reused"
        else:
            recs, status = [], "skipped"
        if require_all_runs and status != "skipped" and not recs:
            raise ValueError(
                f"cell '{cell}' run {rid!r}: window matched no logged answer; "
                "ask this run's questions in the app, or drop it from the yaml"
            )
        records.extend(recs)
        per_run.append(
            {
                "run_id": rid,
                "status": status,
                "n_records": len(recs),
                "n_linked": sum(1 for r in recs if r.linked),
                "start": cond.start.isoformat() if cond.start else None,
                "end": cond.end.isoformat() if cond.end else None,
                "conversation_ids": [str(c) for c in cond.conversation_ids],
                "note": cond.note,
            }
        )

    if keep_undeclared_runs:
        for rid, recs in held.items():
            if any(c.run_id == rid for c in declared):
                continue
            records.extend(recs)
            per_run.append(
                {
                    "run_id": rid,
                    "status": "undeclared",
                    "n_records": len(recs),
                    "n_linked": sum(1 for r in recs if r.linked),
                    "start": None,
                    "end": None,
                    "conversation_ids": [],
                    "note": None,
                }
            )

    gold_summary = (header.get("meta") or {}).get("gold_match") if header else None
    scored = (
        records
        if gold_scope == "all"
        else [r for rid, recs in fresh.items() for r in recs]
    )
    if (
        resolve_gold
        and scored
        and any(getattr(q, "gold_evidence", None) for q in queries)
    ):
        reports = apply_gold_to_records(
            {cell: scored},
            queries,
            default_document=gold_default_document,
            require_all=gold_require_all,
            document_match=document_match,
            document_status=document_status,
            match_mode=match_mode,
            shingle_n=shingle_n,
            min_shingle_coverage=min_shingle_coverage,
            min_token_coverage=min_token_coverage,
            page_window=page_window,
            page_required=page_required,
            max_chunks_per_item=max_chunks_per_item,
            min_score=min_score,
            tie_margin=tie_margin,
        )
        gold_summary = reports[cell].summary()

    measured = cell_measurements(
        cell,
        records,
        device=device,
        power=power,
        measure_power=measure_power,
        power_per_run=power_per_run,
        power_on_error=power_on_error,
        power_printer=power_printer,
        runs=declared,
        measure_ingest=measure_ingest,
        ingest=ingest,
        ingest_on_error=ingest_on_error,
        ingest_printer=ingest_printer,
        measure_tokens=measure_tokens,
        tokens=tokens,
        tokens_printer=tokens_printer,
    )
    power_meta = measured["power"]
    ingest_meta = measured["ingest"]
    tokens_meta = measured["tokens"]

    file_meta: Dict[str, Any] = {
        "benchmark_path": os.path.abspath(bench),
        "postgres_url": settings.postgres_url,
        "runs": per_run,
        "run_ids": [c.run_id for c in declared],
        "power": power_meta,
        "cost_inputs_measured": (power_meta or {}).get("cost_inputs"),
        "serving_seconds": (power_meta or {}).get("serving_seconds"),
        "ingest": ingest_meta,
        "ingest_measured": (ingest_meta or {}).get("cost_inputs"),
        "tokens_measured": tokens_meta,
        "query_ids": sorted({q.id for q in queries}),
        "gold_match": gold_summary,
        "gold_settings": {
            "resolve_gold": resolve_gold,
            "gold_scope": gold_scope,
            "document_match": document_match,
            "document_status": list(document_status) if document_status else None,
            "match_mode": match_mode,
            "shingle_n": shingle_n,
            "min_shingle_coverage": min_shingle_coverage,
            "min_token_coverage": min_token_coverage,
            "page_window": page_window,
            "page_required": page_required,
            "max_chunks_per_item": max_chunks_per_item,
            "min_score": min_score,
            "tie_margin": tie_margin,
        },
    }
    file_meta.update(meta or {})
    written = write_cell(
        records,
        out_path,
        cell=cell,
        meta=file_meta,
        include_chunk_text=include_chunk_text,
        overwrite=overwrite,
        indent=indent,
    )
    return {
        "path": written,
        "cell": cell,
        "n_runs": len(declared),
        "run_ids": [c.run_id for c in declared],
        "n_records": len(records),
        "n_linked": sum(1 for r in records if r.linked),
        "n_judged": sum(1 for r in records if r.judged_chunk_ids),
        "gold": gold_summary,
        "runs": per_run,
        "power": power_meta,
        "ingest": ingest_meta,
        "tokens": tokens_meta,
        "records": records,
    }


# --------------------------------------------------------------------------
# 2. read back
# --------------------------------------------------------------------------


def read_cell(
    path: str,
    *,
    expect_cell: Optional[str] = None,
    strict_schema: bool = True,
    with_meta: bool = False,
):
    """Read a dumped cell file into ``QueryRecord`` objects.

    Unknown keys are dropped and missing keys fall back to the dataclass
    defaults, so a file written before a field was added still reads.

    Args:
        expect_cell: raise if the file's ``cell`` differs.
        strict_schema: raise on an unrecognised ``schema`` value.
        with_meta: return ``(records, header_without_records)``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if strict_schema and payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema is {payload.get('schema')!r}, want {SCHEMA}")
    if expect_cell is not None and payload.get("cell") != expect_cell:
        raise ValueError(
            f"{path}: cell is {payload.get('cell')!r}, want {expect_cell!r}"
        )

    ev_names = {f.name for f in fields(ServedEvidence)}
    rec_names = {f.name for f in fields(QueryRecord)}
    records: List[QueryRecord] = []
    for raw in payload.get("records", []):
        kwargs = {k: v for k, v in raw.items() if k in rec_names}
        kwargs["served_evidence"] = [
            ServedEvidence(**{k: v for k, v in ev.items() if k in ev_names})
            for ev in (raw.get("served_evidence") or [])
        ]
        records.append(QueryRecord(**kwargs))

    if with_meta:
        return records, {k: v for k, v in payload.items() if k != "records"}
    return records


def read_cells(
    paths: Sequence[str],
    *,
    strict_schema: bool = True,
    name_from: str = "file",
) -> Tuple[Dict[str, List[QueryRecord]], Dict[str, dict]]:
    """Read several cell files. Returns ``({cell: records}, {cell: header})``.

    Args:
        name_from: ``file`` takes the cell name stored in the file, ``stem``
            takes the filename without extension (use it when two runs of the
            same cell must sit side by side).
    """
    cells: Dict[str, List[QueryRecord]] = {}
    headers: Dict[str, dict] = {}
    for p in paths:
        records, header = read_cell(p, strict_schema=strict_schema, with_meta=True)
        stem = os.path.splitext(os.path.basename(p))[0]
        name = stem if name_from == "stem" else str(header.get("cell") or stem)
        if name in cells:
            raise ValueError(f"two files claim cell {name!r}; use name_from='stem'")
        cells[name] = records
        headers[name] = dict(header, path=os.path.abspath(p))
    return cells, headers


def list_cell_files(
    cell_dir: str = DEFAULT_CELL_DIR, *, pattern: str = "*.json"
) -> List[str]:
    return sorted(glob.glob(os.path.join(cell_dir, pattern)))


# --------------------------------------------------------------------------
# 2b. re-price a dump off the log, after the run
# --------------------------------------------------------------------------


def power_summary(block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The watts a priced power block claims, and how it reached them."""
    block = block or {}
    ci = block.get("cost_inputs") or {}
    chosen = block.get(str(block.get("basis"))) or {}
    return {
        "watts": ci.get("watts"),
        "coverage": chosen.get("coverage"),
        "upper_bound": bool(ci.get("upper_bound")),
        "ok": bool((block.get("quality") or {}).get("ok")),
        "source": ci.get("source"),
    }


def _priced_line(summary: Mapping[str, Any]) -> str:
    w = summary.get("watts")
    kind = (
        "upper bound"
        if summary.get("upper_bound")
        else "measured"
        if summary.get("ok")
        else "unmeasured"
    )
    return (
        f"{'?' if w is None else round(float(w), 3)} W {kind} "
        f"(coverage {summary.get('coverage')})"
    )


def reprice_cell_file(
    path: str,
    *,
    device: Any,
    power: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    runner: Optional[Callable[..., Dict[str, Any]]] = None,
    min_coverage: Optional[float] = None,
    fallback_tdp: Optional[bool] = None,
    power_meta_key: str = "power",
    cost_meta_key: str = "cost_inputs_measured",
    serving_meta_key: str = "serving_seconds",
    previous_key: Optional[str] = "power_at_dump",
    write: bool = True,
    require_records: bool = True,
    require_measured: bool = False,
    strict_schema: bool = False,
    on_error: str = "warn",
    indent: int = 2,
    encoding: str = "utf-8",
    printer: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """Price one dumped cell at the end, off the finished log, and rewrite it.

    ``dump_cell`` prices a cell the moment it is captured, under whatever the
    conditions ``power:`` block said at that moment, and the result is frozen
    into the file: a block edited afterwards never reaches a cell already
    dumped, so a nameplate upper bound written at 01:02 is what scoring refuses
    at 09:00. This re-reads the finished log over the same answer spans under
    the block as it stands now, and rewrites the three power keys in place.
    Records and their evidence rows are copied through as they were dumped, so
    a re-price needs no database and no re-ingest.

    Args:
        device: the machine, a registry name or an inline profile mapping; what
            the conditions file declares under ``device:``.
        power: the conditions ``power:`` block, forwarded as keywords. The knob
            that recovers a span the meter never fired inside is
            ``min_samples: 1``, which prices it from the two samples bracketing
            it instead of dropping it.
        overrides: keywords applied after ``power``, e.g. ``{"basis": "window"}``.
        runner: the measuring function; None imports
            ``power_meter.measure_cell_serving_power``.
        min_coverage: None keeps the ``power`` block's own floor. A number
            overrides it, and 0.0 prices whatever the log covers and reports
            the fraction in ``source`` instead of gating on it.
        fallback_tdp: None keeps the block's value. False reports a thin log's
            measured watts instead of replacing them with the nameplate.
        power_meta_key / cost_meta_key / serving_meta_key: the three ``meta``
            keys rewritten, named as ``dump_cell`` writes them.
        previous_key: where the dump-time block is kept, written once so a
            second re-price cannot bury the original. None discards it.
        write: False prices and prints, leaving the file alone.
        require_records: a file holding no answers is skipped, not measured.
        require_measured: True refuses to replace a block unless the new one is
            measured, so a failed re-price cannot destroy a good price.
        strict_schema: raise on a file written under another schema version.
        on_error: ``raise`` | ``warn`` | ``skip`` when the measurement itself
            fails.
        indent / encoding: how the file is rewritten.

    Returns:
        ``{cell, path, status, before, after}``, status one of ``repriced``,
        ``priced`` (nothing written), ``skipped``, ``unmeasured``, ``failed``,
        and the two summaries from ``power_summary``.
    """
    if on_error not in ("raise", "warn", "skip"):
        raise ValueError("on_error must be 'raise', 'warn' or 'skip'")

    with open(path, "r", encoding=encoding) as fh:
        data = json.load(fh)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    cell = str(data.get("cell") or os.path.splitext(os.path.basename(path))[0])
    before = power_summary(meta.get(power_meta_key))
    out: Dict[str, Any] = {
        "cell": cell,
        "path": path,
        "status": "skipped",
        "before": before,
        "after": before,
    }

    records = read_cell(path, strict_schema=strict_schema)
    if require_records and not records:
        if printer:
            printer(f"  {cell}: no answers in the dump, not re-priced")
        return out

    if runner is None:
        from veridic_eval.power_meter import measure_cell_serving_power

        runner = measure_cell_serving_power
    kwargs: Dict[str, Any] = dict(power or {})
    if min_coverage is not None:
        kwargs["min_coverage"] = float(min_coverage)
    if fallback_tdp is not None:
        kwargs["fallback_tdp"] = bool(fallback_tdp)
    kwargs.update(overrides or {})
    kwargs.setdefault("cell", cell)
    kwargs.setdefault("printer", printer)
    try:
        block = runner(records, device=device, **kwargs)
    except Exception as exc:
        if on_error == "raise":
            raise
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        if on_error == "warn" and printer:
            printer(f"  {cell}: not re-priced ({out['error']})")
        return out

    after = power_summary(block)
    out["after"] = after
    if require_measured and not after["ok"]:
        out["status"] = "unmeasured"
        if printer:
            printer(f"  {cell}: {_priced_line(after)}, file left as it was")
        return out

    if previous_key and meta.get(power_meta_key) is not None:
        meta.setdefault(previous_key, meta[power_meta_key])
    meta[power_meta_key] = _plain(block)
    meta[cost_meta_key] = _plain(block.get("cost_inputs"))
    meta[serving_meta_key] = _plain(block.get("serving_seconds"))
    out["status"] = "repriced" if write else "priced"
    if write:
        tmp = f"{path}.repriced"
        with open(tmp, "w", encoding=encoding) as fh:
            json.dump(data, fh, indent=indent, default=str)
        os.replace(tmp, path)
    if printer:
        printer(
            f"  {cell}: {_priced_line(before)} -> {_priced_line(after)}"
            f"{'' if write else '  (not written)'}"
        )
    return out


def reprice_cells(
    cells: Sequence[str] = (),
    *,
    device: Any,
    cell_dir: str = DEFAULT_CELL_DIR,
    pattern: str = "*.json",
    paths: Sequence[str] = (),
    missing: str = "raise",
    printer: Optional[Callable[[str], None]] = print,
    **file_kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Re-price named cells, or every dumped cell, off the finished log.

    Args:
        cells: cell names; empty takes every file in ``cell_dir`` matching
            ``pattern``.
        paths: explicit files, used instead of ``cells`` when given.
        missing: a named cell with no dumped file ``raise``s or is ``skip``ped.
        file_kwargs: every ``reprice_cell_file`` knob, per file.

    Returns:
        ``{cell: result}``, one ``reprice_cell_file`` result each.
    """
    if missing not in ("raise", "skip"):
        raise ValueError("missing must be 'raise' or 'skip'")
    todo = (
        list(paths)
        if paths
        else [cell_file_path(n, cell_dir=cell_dir) for n in cells]
        if cells
        else list_cell_files(cell_dir, pattern=pattern)
    )
    out: Dict[str, Dict[str, Any]] = {}
    for p in todo:
        name = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(p):
            if missing == "raise":
                raise FileNotFoundError(f"no dumped cell at {p}")
            if printer:
                printer(f"  {name}: no dump at {p}, skipped")
            continue
        got = reprice_cell_file(p, device=device, printer=printer, **file_kwargs)
        out[str(got.get("cell") or name)] = got
    return out


# --------------------------------------------------------------------------
# 3. render
# --------------------------------------------------------------------------


def _sub(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 6)


def render_cells(
    cells: Dict[str, List[QueryRecord]],
    *,
    reference: str = DEFAULT_REFERENCE,
    metrics: str = "ir",
    k: Optional[int] = None,
    per_answer_cost: Optional[Dict[str, float]] = None,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    unknown: str = "grid",
    offgrid_reference: Optional[str] = None,
    offgrid_n_contrasts: int = 1,
    n_contrasts: Optional[int] = None,
) -> dict:
    """Metrics per cell plus deltas against ``reference``.

    Args:
        metrics: ``ir`` computes retrieval metrics only (ranx, no model loads).
            ``full`` runs the whole per-cell report including faithfulness,
            abstention, CFCA and bootstrap CIs, which loads the faithfulness
            model and needs every cell to hold chunk text.
        k: IR cut-off; defaults to ``settings.ir_k`` and cannot exceed the
            served top_n that was logged.
        per_answer_cost: ``{cell: GBP per answer}``, the CFCA numerator in
            ``full`` mode.
        grid_names / offgrid_names: which loaded cells are scored inside the 2x2
            and which get the separate uncorrected before/after block, in
            ``full`` mode. ``unknown`` places any other name ("grid",
            "offgrid", "drop" or "raise").
        offgrid_reference: cell the off-grid windows are differenced against;
            defaults to ``reference``.
        offgrid_n_contrasts: Bonferroni divisor for the off-grid block; 1 leaves
            it uncorrected.
        n_contrasts: overrides the grid family size, which otherwise counts the
            loaded grid cells minus the reference.
    """
    if metrics not in ("ir", "full"):
        raise ValueError("metrics must be 'ir' or 'full'")
    if not cells:
        raise ValueError("no cells given")
    if reference not in cells:
        raise ValueError(f"reference {reference!r} not among {list(cells)}")

    if metrics == "full":
        report = evaluate_experiment_split(
            cells,
            reference=reference,
            per_answer_cost=per_answer_cost or {},
            grid_names=grid_names,
            offgrid_names=offgrid_names,
            unknown=unknown,
            n_contrasts=n_contrasts,
            offgrid_reference=offgrid_reference,
            offgrid_n_contrasts=offgrid_n_contrasts,
        )
        report["mode"] = "full"
        return report

    k = k or settings.ir_k
    out: Dict[str, dict] = {}
    for name, recs in cells.items():
        ir = evaluate_retrieval(recs, k)
        out[name] = {
            "n_queries": len(recs),
            "n_linked": sum(1 for r in recs if r.linked),
            "n_judged": ir.get("n_judged", 0),
            "n_linked_no_evidence": sum(
                1 for r in recs if r.linked and not r.served_evidence
            ),
            "aggregate": ir["aggregate"],
            "per_query": ir["per_query"],
            "note": ir.get("note"),
        }
    deltas = {
        contrast_key(name, reference): {
            m: _sub(cell["aggregate"].get(m), out[reference]["aggregate"].get(m))
            for m in cell["aggregate"]
        }
        for name, cell in out.items()
        if name != reference
    }
    return {
        "mode": "ir",
        "k": k,
        "reference": reference,
        "conditions": list(cells),
        "cells": out,
        deltas_key(reference): deltas,
    }


def cells_table(report: dict) -> List[dict]:
    """One row per cell for display, for either ``metrics`` mode."""
    rows: List[dict] = []
    for name, cell in report["cells"].items():
        if report.get("mode") == "ir":
            row = {
                "cell": name,
                "linked": f"{cell['n_linked']}/{cell['n_queries']}",
                "judged": cell["n_judged"],
            }
            row.update(cell["aggregate"])
        else:
            fa = cell["faithfulness"].get("aggregate", {})
            ab = cell["abstention"]["aggregate"]
            row = {
                "cell": name,
                "linked": f"{cell['n_linked']}/{cell['n_queries']}",
                "judged": cell["retrieval"]["n_judged"],
            }
            row.update(cell["retrieval"]["aggregate"])
            row["faithful_rate"] = fa.get("faithful_rate")
            row["abstention_recall"] = ab.get("abstention_recall")
            row["over_abstention_rate"] = ab.get("over_abstention_rate")
            row["cfca"] = cell["cfca"].get("cfca")
        rows.append(row)
    return rows


def read_report_file(
    path: str,
    *,
    encoding: str = "utf-8",
    cells_key: str = "cells",
    on_error: str = "note",
) -> Tuple[Optional[dict], str]:
    """The report `run` wrote, as ``(report, reason)``. Nothing is scored here.

    Scoring belongs to `run`, which commits `out/report.json`; the view reads that
    file so a figure cannot disagree with the run that produced it.

    Args:
        path: the json to read.
        cells_key: the key that makes a json file a report rather than a dump.
        on_error: ``note`` returns the reason, ``raise`` propagates.
    """
    if not path:
        return None, "no report path"
    if not os.path.exists(path):
        return None, f"no file at {path}"
    try:
        with open(path, "r", encoding=encoding) as fh:
            data = json.load(fh)
    except Exception as exc:
        if on_error == "raise":
            raise
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict) or cells_key not in data:
        return None, f"{path} carries no `{cells_key}` mapping, so it is not a report"
    return data, ""


def deltas_table(report: dict) -> List[dict]:
    """One row per (contrast, metric)."""
    rows: List[dict] = []
    if report.get("mode") == "ir":
        for contrast, dmap in deltas_of(report).items():
            for m, d in dmap.items():
                rows.append({"contrast": contrast, "metric": m, "delta": d})
        return rows
    for contrast, dmap in deltas_of(report).items():
        for m, d in dmap.items():
            rows.append(
                {
                    "contrast": contrast,
                    "metric": m,
                    "delta": d.get("delta"),
                    "ci_low": d.get("ci_low"),
                    "ci_high": d.get("ci_high"),
                    "perm_p_bonferroni": d.get("perm_p_bonferroni"),
                    "significant": d.get("significant"),
                }
            )
    return rows


def offgrid_table(report: dict, *, cfca_key: str = "cfca") -> List[dict]:
    """One row per (off-grid contrast, metric): the before/after check, uncorrected.

    Empty for ``ir`` mode, for a report written before the off-grid split, and
    for any run whose conditions held no off-grid window.
    """
    rows: List[dict] = []
    for contrast, dmap in offgrid_deltas_of(report).items():
        for m, d in dmap.items():
            if m == cfca_key:
                rows.append(
                    {
                        "contrast": contrast,
                        "metric": m,
                        "delta": d.get("delta"),
                        "ci_low": None,
                        "ci_high": None,
                        "perm_p": None,
                        "n": None,
                    }
                )
                continue
            rows.append(
                {
                    "contrast": contrast,
                    "metric": m,
                    "delta": d.get("delta"),
                    "ci_low": d.get("ci_low"),
                    "ci_high": d.get("ci_high"),
                    "perm_p": d.get("perm_p"),
                    "n": d.get("n"),
                }
            )
    return rows


def report_markdown(report: dict) -> str:
    """Markdown for either mode; ``full`` reuses ``report.render_markdown``."""
    if report.get("mode") != "ir":
        return render_markdown(report)
    k = report["k"]
    lines = [
        "# VERIDIC eval, retrieval only\n",
        f"- reference cell: **{report['reference']}**",
        f"- cells: {', '.join(report['conditions'])}",
        f"- IR cap: k = {k} (served evidence only)\n",
        "## Per-cell\n",
        f"| cell | linked | judged | recall@{k} | mrr@{k} | ndcg@{k} | hit_rate@{k} |",
        "|" + "---|" * 7,
    ]

    def f(v):
        return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    for row in cells_table(report):
        lines.append(
            f"| {row['cell']} | {row['linked']} | {row['judged']} "
            f"| {f(row.get(f'recall@{k}'))} | {f(row.get(f'mrr@{k}'))} "
            f"| {f(row.get(f'ndcg@{k}'))} | {f(row.get(f'hit_rate@{k}'))} |"
        )
    if deltas_of(report):
        lines += [
            f"\n## Deltas vs {report['reference']} (aggregate difference, no CI)\n",
            "| contrast | metric | delta |",
            "|---|---|---|",
        ]
        for row in deltas_table(report):
            lines.append(f"| {row['contrast']} | {row['metric']} | {f(row['delta'])} |")
    return "\n".join(lines)


def query_table(records: Sequence[QueryRecord]) -> List[dict]:
    """One row per query: linkage, gold count, served count, hit at rank 1."""
    rows: List[dict] = []
    for r in records:
        served = r.served_chunk_ids
        gold = set(r.judged_chunk_ids)
        rows.append(
            {
                "query_id": r.query_id,
                "answerable": r.answerable,
                "linked": r.linked,
                "gold": len(gold),
                "served": len(served),
                "hits": sum(1 for c in served if c in gold),
                "first_hit_rank": next(
                    (i + 1 for i, c in enumerate(served) if c in gold), None
                ),
                "question": r.question,
            }
        )
    return rows


# --------------------------------------------------------------------------
# streamlit ui
# --------------------------------------------------------------------------


def result_panel(
    st,
    title: str,
    rows: Sequence[dict],
    chart: Any = None,
    *,
    caption: str = "",
    table: bool = True,
    table_width: str = "stretch",
    empty_note: str = "",
    column_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """One results panel: heading, its table, then its chart. Returns drawn or not.

    Every panel in the results view goes through this, so a panel cannot pick up
    a different layout, a chart without its table, or a heading over nothing.

    Args:
        st: the streamlit module, passed in so this stays importable headless.
        rows: table rows; empty draws ``empty_note`` and nothing else.
        chart: an altair chart from `charts`, or None for a table-only panel.
        caption: one line under the heading, for what the panel does not claim.
        table / table_width: draw the table, and how wide.
        empty_note: shown when there are no rows, "" for silence.
        column_config: streamlit column config, normally
            `provenance.column_config` so each column header carries the
            file:line, function and equation behind it. None draws it bare.
    """
    if not rows:
        if empty_note:
            st.caption(empty_note)
        return False
    st.subheader(title)
    if caption:
        st.caption(caption)
    if table:
        st.dataframe(rows, width=table_width, column_config=column_config)
    if chart is not None:
        st.altair_chart(chart)
    return True


def render_results_view(
    st,
    report: dict,
    *,
    reference: Optional[str] = None,
    with_charts: bool = True,
    with_downloads: bool = True,
    cfca_key: str = "cfca",
    with_provenance: bool = True,
    provenance_kwargs: Optional[Mapping[str, Any]] = None,
    section: Optional[str] = None,
    picker_label: str = "section",
) -> None:
    """The results tab: a picker, then ONE section drawn, nothing else built.

    Reads a report `run` already wrote and scores nothing, so the figures and the
    tables are two views of one committed number. Sections: ``per cell``,
    ``deltas``, ``cost``, ``off-grid``, ``downloads``; the unpicked ones are
    never computed, so ten runs cost one section of scroll, not five stacked.

    Args:
        reference: name shown in the deltas heading; None reads it off the report.
        with_charts: draw the chart beside the picked section's table.
        with_downloads: offer the ``downloads`` section at all.
        cfca_key: the per-cell cost column the cost section plots.
        with_provenance: hover every column header for the file:line, function
            and equation that computed it, plus this run's inputs for it.
        provenance_kwargs: passed to `provenance.column_config`, e.g.
            ``{"unknown": "label"}`` to mark a column no entry covers, or
            ``{"chain_limit": 3}`` for a shorter tooltip.
        section: draw this section with no picker; None shows the picker.
        picker_label: the picker's label.
    """
    ref = reference or reference_from_mapping(report) or DEFAULT_REFERENCE

    def config_for(rows: Sequence[dict]) -> Optional[Dict[str, Any]]:
        if not with_provenance:
            return None
        return provenance.column_config(
            st, rows, report=report, **(provenance_kwargs or {})
        )

    def cost_pair(
        rows: Sequence[dict],
    ) -> Tuple[List[dict], Any, str, List[str]]:
        """Every cell's reported price and its warm-up sensitivity, one bar each.

        Both sections that price a cell build the picture here, so the pair is
        the same picture in each. The chart shape puts the warm-only number in
        its own column of the cell's own row, which is what makes `cost_chart`
        group two bars under one cell label instead of one merged bar.

        Returns:
            ``(rows, chart, note, paired)``. ``paired`` is the cells that gained
            the second bar, empty for a report written before the block existed;
            ``note`` names the rule when they are there and the command that
            writes it when they are not, so a one-bar chart says why it is one.
        """
        pair_rows, paired = warm_only_columns(rows, report, value_key=cfca_key)
        chart = (
            cost_chart(
                pair_rows,
                value_key=cfca_key,
                extra_keys=(WARM_COLUMN,),
                labels={cfca_key: BASE_SERIES},
            )
            if (with_charts and pair_rows)
            else None
        )
        if paired:
            return pair_rows, chart, warm_only_caption(report), list(paired)
        return (
            pair_rows,
            chart,
            (
                "This report carries no warm-only block, so each cell is drawn with "
                "its reported price alone. `veridic-eval warmup --report "
                "<report.json>` re-prices the dumps that run wrote and fills the "
                "second bar; it re-scores nothing."
            ),
            [],
        )

    def per_cell() -> None:
        cell_rows = cells_table(report)
        result_panel(
            st,
            "per cell",
            cell_rows,
            levels_chart(cell_rows) if with_charts else None,
            caption="Levels behind every delta.",
            column_config=config_for(cell_rows),
        )
        # The warm-up pair is drawn here as well as under `cost`. These are the
        # cells the reader already has on screen, and a sensitivity that lives
        # one picker click away is a sensitivity nobody looks at. Chart only:
        # the numbers stay under `cost`, so this is the picture, not a copy.
        # A report with no warm block has no pair to show and is left alone;
        # `cost` is where that absence is already named.
        pair_rows, pair_chart, pair_note, paired = cost_pair(cell_rows)
        if pair_chart is not None and paired:
            result_panel(
                st,
                "CFCA cost with and without warm-up",
                pair_rows,
                pair_chart,
                table=False,
                caption=(
                    "Two bars for every cell above: the price this run reported, "
                    "and the same cell re-priced with its first answer dropped. "
                    f"{pair_note}"
                ),
            )

    def deltas() -> None:
        delta_rows = deltas_table(report)
        result_panel(
            st,
            f"deltas vs {ref}",
            delta_rows,
            delta_ci_chart(delta_rows) if with_charts else None,
            column_config=config_for(delta_rows),
            caption=(
                f"95% CI as the black rule, Bonferroni over "
                f"{report.get('n_contrasts')} grid contrasts."
            ),
        )

    def cost() -> None:
        cell_rows, _ = warm_only_rows(cells_table(report), report)
        # Table and chart both read across: one line per cell carrying the
        # reported price and its warm-up sensitivity side by side, so the pair
        # is compared at the cell instead of hunted for two slots apart.
        _, pair_chart, note, _paired = cost_pair(cells_table(report))
        money_rows = cost_columns(report) or cell_rows
        caption = "Recurring GBP per faithfully-cited answer; index build is capex, reported apart."
        result_panel(
            st,
            "CFCA cost",
            money_rows,
            pair_chart,
            caption=f"{caption} {note}".strip(),
        )

    def offgrid() -> None:
        off_rows = offgrid_table(report)
        off_ref = report.get("offgrid_reference") or DEFAULT_REFERENCE
        result_panel(
            st,
            f"off-grid before/after vs {off_ref}",
            off_rows,
            delta_ci_chart(off_rows, flag_key=None) if with_charts else None,
            column_config=config_for(off_rows),
            caption=(
                "Sizes the one-time baseline construction: "
                f"{', '.join(report.get('offgrid') or [])}. Outside the "
                f"{report.get('n_contrasts')} grid contrasts, uncorrected, so it "
                "carries no research claim."
            ),
        )

    def downloads() -> None:
        d1, d2 = st.columns(2)
        d1.download_button(
            "report.json",
            json.dumps(report, indent=2, default=str),
            file_name="report.json",
            mime="application/json",
        )
        d2.download_button(
            "report.md",
            report_markdown(report),
            file_name="report.md",
            mime="text/markdown",
        )
        with st.expander("markdown"):
            st.markdown(report_markdown(report))
        with st.expander("raw json"):
            st.json(report, expanded=False)

    sections: Dict[str, Callable[[], None]] = {
        "per cell": per_cell,
        "deltas": deltas,
        "cost": cost,
        "off-grid": offgrid,
    }
    if with_downloads:
        sections["downloads"] = downloads
    picked = (
        section if section is not None else st.selectbox(picker_label, list(sections))
    )
    draw = sections.get(picked)
    if draw is None:
        st.caption(f"no section `{picked}`; sections: {', '.join(sections)}")
        return
    draw()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="evals", layout="wide")
    # st.title("eval cells")

    views = eval_view_options()
    with st.sidebar:
        st.caption(f"postgres: {settings.postgres_url}")
        view_name = st.selectbox(
            "eval",
            list(views),
            help="v1 is the ten-query marking, v2 the same ten plus q021 and "
            "q022. The pick moves the cell dir, the benchmark and the "
            "report json below, and each one stays editable.",
        )
        view = views[view_name]
        cell_dir = st.text_input("cell dir", view["cell_dir"])
        benchmark_path = st.text_input("benchmark", view["benchmark_path"])
        conditions_path = st.text_input("conditions", "./conditions.yaml")
        if st.button("ping postgres"):
            from veridic_eval.db import ping

            st.write("reachable" if ping() else "unreachable")

    conds: Dict[str, Condition] = {}
    costs: Dict[str, float] = {}
    yaml_reference = DEFAULT_REFERENCE
    if conditions_path and os.path.exists(conditions_path):
        try:
            parsed = load_conditions_file(conditions_path, printer=None)
            yaml_reference, costs = parsed.reference, parsed.per_answer_cost
            conds = {c.name: c for c in parsed.conditions}
        except Exception as exc:
            st.sidebar.warning(f"conditions: {exc}")

    tab_results, tab_dump, tab_inspect = st.tabs(["results", "dump", "inspect"])

    with tab_dump:
        names = cell_names(extra=tuple(conds))
        pick = st.selectbox(
            "cell from conditions.yaml",
            ["(type my own)"] + names,
            index=(names.index(DEFAULT_CELL) + 1) if DEFAULT_CELL in names else 0,
        )
        picked = conds.get(pick)
        c1, c2, c3 = st.columns(3)
        cell = c1.text_input("cell name", picked.name if picked else DEFAULT_CELL)
        start_s = c2.text_input(
            "start (ISO, blank = no bound)",
            picked.start.isoformat() if picked and picked.start else "",
        )
        end_s = c3.text_input(
            "end (ISO, blank = no bound)",
            picked.end.isoformat() if picked and picked.end else "",
        )
        conv_s = st.text_area(
            "conversation ids (one per line, blank = window only)",
            "\n".join(picked.conversation_ids) if picked else "",
            height=80,
        )
        qids_s = st.text_area(
            "query ids (one per line, blank = whole benchmark)", "", height=68
        )

        c4, c5, c6 = st.columns(3)
        include_chunk_text = c4.checkbox("include chunk text", True)
        overwrite = c5.checkbox("overwrite", True)
        out_path = c6.text_input("out file (blank = <cell dir>/<cell>.json)", "")

        with st.expander("gold evidence"):
            g1, g3 = st.columns(2)
            resolve_gold = g1.checkbox("match gold text to chunk ids", True)
            gold_require_all = g3.checkbox("require every gold text to match", False)
            g4, g5, g6 = st.columns(3)
            document_match = g4.selectbox(
                "document match", ["exact", "casefold", "basename", "contains"]
            )
            match_mode = g5.selectbox(
                "match mode",
                ["containment_then_shingle", "containment", "shingle", "token"],
            )
            gold_default_document = g6.text_input("default document (blank = none)", "")
            g7, g8, g9 = st.columns(3)
            shingle_n = g7.number_input("shingle n", 2, 20, 5)
            min_shingle_coverage = g8.number_input(
                "min shingle coverage", 0.0, 1.0, 0.6, 0.05
            )
            min_token_coverage = g9.number_input(
                "min token coverage", 0.0, 1.0, 0.9, 0.05
            )
            g10, g11, g12 = st.columns(3)
            page_window_raw = g10.number_input("page window (-1 = off)", -1, 50, -1)
            page_required = g11.checkbox("page required", False)
            max_chunks_per_item = g12.number_input(
                "max chunks per gold text (0 = all)", 0, 20, 0
            )
            g13, g14 = st.columns(2)
            min_score = g13.number_input("min score", 0.0, 1.0, 0.0, 0.05)
            tie_margin = g14.number_input("tie margin", 0.0, 1.0, 0.0, 0.05)

        if st.button("dump cell", type="primary"):
            try:
                result = dump_cell(
                    cell=cell,
                    benchmark_path=benchmark_path,
                    path=out_path or None,
                    cell_dir=cell_dir,
                    start=parse_timestamp(start_s.strip() or None),
                    end=parse_timestamp(end_s.strip() or None),
                    conversation_ids=[
                        x.strip() for x in conv_s.splitlines() if x.strip()
                    ],
                    query_ids=[x.strip() for x in qids_s.splitlines() if x.strip()]
                    or None,
                    resolve_gold=resolve_gold,
                    gold_default_document=gold_default_document.strip() or None,
                    gold_require_all=gold_require_all,
                    document_match=document_match,
                    match_mode=match_mode,
                    shingle_n=int(shingle_n),
                    min_shingle_coverage=float(min_shingle_coverage),
                    min_token_coverage=float(min_token_coverage),
                    page_window=None if page_window_raw < 0 else int(page_window_raw),
                    page_required=page_required,
                    max_chunks_per_item=int(max_chunks_per_item),
                    min_score=float(min_score),
                    tie_margin=float(tie_margin),
                    include_chunk_text=include_chunk_text,
                    overwrite=overwrite,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(
                    f"{result['path']}: {result['n_records']} queries, "
                    f"{result['n_linked']} linked, {result['n_judged']} judged"
                )
                if result["gold"] is not None:
                    st.json(result["gold"], expanded=False)
                dump_rows = query_table(result["records"])
                st.dataframe(
                    dump_rows,
                    width="stretch",
                    column_config=provenance.column_config(st, dump_rows),
                )

    with tab_results:
        rp1, rp2 = st.columns([5, 1])
        report_path = rp1.text_input(
            f"{view_name} report json, written by `veridic-eval run`",
            view["report_path"],
        )
        rp2.button("reload")
        report, load_note = read_report_file(report_path)
        if not report:
            st.info(
                f"{load_note}. Score {view_name} first:\n\n"
                f"`{view['run_hint']}`\n\n"
                "This tab scores nothing, so a figure here is always the number "
                "that run committed."
            )
        else:
            render_results_view(st, report)

    with tab_inspect:
        files = list_cell_files(cell_dir)
        path = st.selectbox("file", files) if files else None
        if path:
            records, header = read_cell(path, strict_schema=False, with_meta=True)
            st.caption(
                f"cell={header.get('cell')} written={header.get('written_at')} "
                f"records={header.get('n_records')} linked={header.get('n_linked')} "
                f"judged={header.get('n_judged')}"
            )
            cell_query_rows = query_table(records)
            st.dataframe(
                cell_query_rows,
                width="stretch",
                column_config=provenance.column_config(st, cell_query_rows),
            )
            ids = [r.query_id for r in records]
            if ids:
                qid = st.selectbox("query", ids)
                rec = next(r for r in records if r.query_id == qid)
                st.write(f"**question** {rec.question}")
                st.write(f"**gold answer** {rec.gold_answer or '(none)'}")
                st.write(f"**answer** {rec.answer_text or '(not linked)'}")
                st.write(f"**judged chunk ids** {rec.judged_chunk_ids or '(none)'}")
                st.dataframe(provenance.record_gates(rec), width="stretch")
                st.caption(
                    "Every gate this answer passes through, with the line that runs it "
                    "and the fields that line is handed. Nothing is scored here, so a row "
                    "cannot disagree with the run: read the line, not a verdict."
                )
                gold = set(rec.judged_chunk_ids)
                for ev in rec.served_evidence:
                    mark = "HIT" if ev.chunk_id in gold else "miss"
                    with st.expander(
                        f"rank {ev.served_rank + 1} [{mark}] {ev.document_name} p{ev.page} "
                        f"retr={ev.retrieval_score} rerank={ev.rerank_score}"
                    ):
                        st.text(ev.chunk_text or "(chunk text not in this file)")


if __name__ == "__main__":
    main()
