"""
Warm-up sensitivity: the same cells priced without each sitting's first answer.

The priced second on this rig is wall clock (`power_meter.serving_intervals`
sums retrieval + rerank + generation and bills the watts under it), so the
first answer of a sitting carries the model load into the cell's mean. In the
v2 grid `combined` answered its first question in 73.7 s against a 7.9 s median
for the other eleven: one span, 47 percent of that cell's entire serving time,
and the reason `combined` reads as roughly twice `control`'s cost per answer.
`control` paid the same toll (10.0 s against a 3.4 s median) but a smaller one,
because a load is a fixed number of seconds and its cell's answers are shorter.

This module re-prices every cell with that one answer dropped and writes the
result **beside** the reported block, never over it. Nothing is re-scored:
faithfulness, citations, evidence links and therefore P_hat are read from the
report exactly as `run` wrote them, and the only quantity that moves is
``query_gpu_seconds``. The arithmetic is `conditions.cfca_for_cells` over
`cfca_cost.cost_per_answer` -- the same two functions that produced the
headline block -- so the two tables are subtractable rather than merely
comparable, and `verify.check_cfca_gbp_block` reads the new block with
``block_key="cfca_warm_only"`` and no other change.

    veridic-eval warmup --report out-v2/report.json --cell-dir out/cells-v2

What the warm-only row does NOT do: re-integrate the power log. ``watts``,
and so ``p_gpu_hour``, stay the cell's measured mean over the whole sitting,
load included; tokens, storage and the one-time index build are untouched.
Seconds are the term a cold model inflates, so seconds are the term that moves.
The figure is a sensitivity on a measurement, not a second measurement, and the
report says so in the section this writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from veridic_eval.conditions import cfca_for_cells
from veridic_eval.power_meter import (
    SERVING_LATENCY_FIELDS,
    Interval,
    serving_intervals,
)

#: report field this block is written under, next to ``cfca_gbp``.
WARM_FIELD: str = "cfca_warm_only"
#: report field holding the priced block this one is derived from.
COST_FIELD: str = "cfca_gbp"
#: report field holding the per-cell serving-second measurement.
SERVING_FIELD: str = "serving_seconds"
#: the field set inside ``serving_seconds`` that the cost block was priced off.
PRICED_SET: str = "serving"
#: the one cost input this module moves.
SECONDS_INPUT: str = "query_gpu_seconds"
#: heading of the money table this adds its rows to, in report.md.
SECTION_TITLE: str = "## CFCA (GBP, declared cost inputs)"
#: suffix on the added row's cell name, in the table and in the viewer.
ROW_LABEL: str = "(wo warmup)"
#: series name the viewer paints those rows with.
ROW_SERIES: str = "without warm-up"
#: series name the rows the run reported keep.
BASE_SERIES: str = "reported"
#: column the warm-only cost is written to for the chart, and its legend text.
#: One row per cell keeps the pair under a single cell label instead of two.
WARM_COLUMN: str = "without warm-up"
#: heading an earlier build wrote a separate table under; removed on sight.
LEGACY_TITLE: str = "## CFCA without the sitting's first answer (warm-only)"


# --------------------------------------------------------------------------
# 1. spans: one sitting's answers, in the order they were served
# --------------------------------------------------------------------------

def sitting_spans(
    records: Sequence[Any],
    *,
    latency_fields: Sequence[str] = SERVING_LATENCY_FIELDS,
    run_field: str = "run_id",
    **interval_kwargs: Any,
) -> List[Tuple[str, List[Interval]]]:
    """One entry per sitting: ``(run_id, intervals sorted by start)``.

    `serving_intervals` decides what a priced answer is -- a record with a
    timestamp and at least a millisecond of summed latency -- so a null or
    unserved row never becomes a warm-up candidate and never enters the count
    here, exactly as it never entered the reported mean.

    Args:
        latency_fields: the fields the cell was priced on; take them from the
            report's ``serving_seconds[cell]["serving"]["fields"]`` so this can
            not silently price a different sum.
        run_field: record attribute naming the sitting; runs come back in
            first-seen order, and a record without one falls in a single group.
        interval_kwargs: passed through to `serving_intervals`.
    """
    order: List[str] = []
    by_run: Dict[str, List[Any]] = {}
    for rec in records:
        run = str(getattr(rec, run_field, "") or "")
        if run not in by_run:
            by_run[run] = []
            order.append(run)
        by_run[run].append(rec)
    return [
        (run, serving_intervals(by_run[run], latency_fields=latency_fields, **interval_kwargs))
        for run in order
    ]


def warmup_split(
    spans: Sequence[Tuple[str, Sequence[Interval]]],
    *,
    drop: int = 1,
) -> Tuple[List[Tuple[str, Interval]], List[Tuple[str, Interval]]]:
    """Split every sitting into ``(dropped, kept)`` at its ``drop``-th answer.

    Args:
        drop: answers discarded from the head of each sitting. 0 disables the
            rule and makes the warm row a copy of the reported one.
    """
    if drop < 0:
        raise ValueError(f"drop must be 0 or more, got {drop}")
    dropped: List[Tuple[str, Interval]] = []
    kept: List[Tuple[str, Interval]] = []
    for run, ivs in spans:
        dropped += [(run, iv) for iv in list(ivs)[:drop]]
        kept += [(run, iv) for iv in list(ivs)[drop:]]
    return dropped, kept


def warm_seconds(
    reported: Mapping[str, Any],
    spans: Sequence[Tuple[str, Sequence[Interval]]],
    *,
    drop: int = 1,
    on_mismatch: str = "raise",
    seconds_tol: float = 0.05,
    round_s: int = 4,
) -> Dict[str, Any]:
    """The warm-only per-answer seconds for one cell, and the audit behind it.

    The base is the report's own ``total_s`` and ``n_answers``, with the dropped
    spans subtracted from both, so the warm figure inherits the run's window
    arithmetic instead of re-deriving it. Re-summing the dump is used only to
    prove the dump under this report is the dump that produced it.

    Args:
        reported: the cell's ``serving_seconds[cell]["serving"]`` block.
        spans: `sitting_spans` output for the same cell.
        drop: answers discarded per sitting.
        on_mismatch: ``raise`` | ``warn`` | ``ignore`` when the re-summed dump
            disagrees with the report's total by more than ``seconds_tol``,
            which means the cells directory is not the one that was scored.
        seconds_tol: slack on that comparison, in seconds.
        round_s: decimals on the emitted seconds.

    Returns:
        ``{n_answers, total_s, per_answer_s, warm_*, dropped, resummed_s, note}``.
        ``warm_per_answer_s`` is None when the rule would empty the cell.
    """
    if on_mismatch not in ("raise", "warn", "ignore"):
        raise ValueError("on_mismatch must be 'raise', 'warn' or 'ignore'")

    n_reported = int(reported.get("n_answers") or 0)
    total_reported = float(reported.get("total_s") or 0.0)
    dropped, kept = warmup_split(spans, drop=drop)
    resummed = sum(iv.seconds for _, ivs in spans for iv in ivs)
    n_resummed = sum(len(list(ivs)) for _, ivs in spans)

    note = ""
    gap = abs(resummed - total_reported)
    if n_resummed != n_reported or gap > seconds_tol:
        message = (
            f"the dump re-sums to {n_resummed} answers / {resummed:.3f} s against the report's "
            f"{n_reported} / {total_reported:.3f} s: this cells directory is not the one this "
            f"report was scored from, so the warm-only figure would not be its sensitivity"
        )
        if on_mismatch == "raise":
            raise ValueError(message)
        if on_mismatch == "warn":
            note = message

    dropped_s = sum(iv.seconds for _, iv in dropped)
    warm_n = n_reported - len(dropped)
    warm_total = total_reported - dropped_s
    out: Dict[str, Any] = {
        "n_answers": n_reported,
        "total_s": round(total_reported, round_s),
        "per_answer_s": reported.get("per_answer_s"),
        "dropped": [
            {"run_id": run, "query_id": iv.label, "seconds": round(iv.seconds, round_s),
             "answered_at": iv.end.isoformat()}
            for run, iv in dropped
        ],
        "dropped_s": round(dropped_s, round_s),
        "dropped_share": round(dropped_s / total_reported, 4) if total_reported else None,
        "warm_n_answers": warm_n,
        "warm_total_s": round(warm_total, round_s),
        "warm_per_answer_s": round(warm_total / warm_n, round_s) if warm_n > 0 else None,
        "kept_max_s": round(max((iv.seconds for _, iv in kept), default=0.0), round_s),
        "resummed_s": round(resummed, round_s),
        "fields": list(reported.get("fields") or SERVING_LATENCY_FIELDS),
    }
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------
# 2. the block: the priced table, re-priced on those seconds
# --------------------------------------------------------------------------

def warm_only_block(
    report: Mapping[str, Any],
    spans: Mapping[str, Sequence[Tuple[str, Sequence[Interval]]]],
    *,
    drop: int = 1,
    cost_field: str = COST_FIELD,
    serving_field: str = SERVING_FIELD,
    priced_set: str = PRICED_SET,
    seconds_input: str = SECONDS_INPUT,
    round_to: Optional[int] = 8,
    cost_tol: float = 5e-9,
    on_mismatch: str = "raise",
    seconds_tol: float = 0.05,
    source: str = "",
) -> Dict[str, Any]:
    """Re-price the report's own cost block on warm-only seconds.

    Every cell keeps its declared inputs, its measured watts and its scored
    P_hat; ``query_gpu_seconds`` alone is replaced, and the block comes back in
    the shape `conditions.cfca_for_cells` writes, deltas included, so the
    existing renderer, the existing checker and the existing viewer read it.

    A cell whose stored inputs do not reproduce its stored ``cost_per_answer``
    raises: that means the report's cost was not the arithmetic of the inputs
    printed beside it, and a sensitivity on those inputs would be fiction.

    Args:
        spans: ``{cell: sitting_spans(...)}``, cells absent from it are dropped
            from the block rather than priced on the reported seconds.
        drop: answers discarded per sitting.
        round_to: decimals for the emitted numbers; 8 matches ``cfca_gbp``.
        cost_tol: slack when re-deriving each cell's reported cost.
        on_mismatch / seconds_tol: passed to `warm_seconds`.
        source: free text recorded in the block's provenance, e.g. the cells
            directory the spans were read from.
    """
    from veridic_eval.cfca_cost import cost_per_answer

    cost_block = dict((report or {}).get(cost_field) or {})
    cells = cost_block.get("cells") or {}
    if not cells:
        raise ValueError(
            f"report has no {cost_field}.cells to re-price; run `veridic-eval run` with a "
            f"`cost:` block in conditions.yaml first"
        )
    serving = (report or {}).get(serving_field) or {}

    inputs: Dict[str, Dict[str, float]] = {}
    p_hat: Dict[str, Optional[float]] = {}
    audit: Dict[str, Any] = {}
    for name, entry in cells.items():
        declared = dict(entry.get("inputs") or {})
        if not declared:
            raise ValueError(f"{cost_field}.cells.{name} has no `inputs`; re-run `run` with "
                             f"include_inputs on, the sensitivity has nothing to move")
        if name not in spans:
            continue
        measured = ((serving.get(name) or {}).get(priced_set)) or {}
        if not measured:
            raise ValueError(f"{serving_field}.{name}.{priced_set} is missing; the cell was not "
                             f"priced on measured seconds, so there is no warm-up to remove")
        reported_cost = entry.get("cost_per_answer")
        rebuilt = float(cost_per_answer(**declared)["Cost_per_answer"])
        if reported_cost is not None and abs(rebuilt - float(reported_cost)) > cost_tol:
            raise ValueError(
                f"{cost_field}.cells.{name}: the stored inputs re-price to {rebuilt:.10g}, not the "
                f"stored {float(reported_cost):.10g}; the block and its inputs disagree"
            )
        seconds = warm_seconds(
            measured, spans[name], drop=drop,
            on_mismatch=on_mismatch, seconds_tol=seconds_tol,
        )
        if seconds["warm_per_answer_s"] is None:
            continue
        declared_s = declared.get(seconds_input)
        if declared_s is not None and abs(float(declared_s) - float(measured.get("per_answer_s") or 0.0)) > 1e-6:
            raise ValueError(
                f"{cost_field}.cells.{name}: priced {seconds_input}={declared_s} but "
                f"{serving_field} measured {measured.get('per_answer_s')}; the cost block was not "
                f"priced on this measurement"
            )
        inputs[name] = {**declared, seconds_input: seconds["warm_per_answer_s"]}
        p_hat[name] = entry.get("p_hat")
        audit[name] = seconds

    if not inputs:
        raise ValueError("no cell carried both a cost block and a priced answer to drop")

    block = cfca_for_cells(
        inputs,
        p_hat,
        reference=cost_block.get("reference") if cost_block.get("reference") in inputs else None,
        order=[n for n in cells if n in inputs],
        round_to=round_to,
        currency=str(cost_block.get("currency") or "GBP"),
    )
    block["warmup"] = audit
    block["drop"] = drop
    block["rule"] = (
        f"the first {drop} priced answer(s) of each sitting are dropped from "
        f"{seconds_input}; P_hat, watts, tokens and the one-time build are the report's own"
    )
    block["derived_from"] = cost_field
    block["source"] = source or "cell dumps beside this report"
    block["written_at"] = datetime.now(timezone.utc).isoformat()
    block["vs_reported"] = {
        name: {
            "cost_per_answer": _sub(entry.get("cost_per_answer"),
                                    (cells.get(name) or {}).get("cost_per_answer"), round_to),
            "cfca": _sub(entry.get("cfca"), (cells.get(name) or {}).get("cfca"), round_to),
        }
        for name, entry in (block.get("cells") or {}).items()
    }
    return block


def warm_only_rows(
    rows: Sequence[Dict[str, Any]],
    report: Mapping[str, Any],
    *,
    field: str = WARM_FIELD,
    cell_key: str = "cell",
    value_key: str = "cfca",
    cost_key: str = "cost_per_answer",
    label: str = ROW_LABEL,
    series_key: str = "series",
    series: str = ROW_SERIES,
    base_series: str = BASE_SERIES,
    only: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:
    """Give each cell a second display row: the same cell without its warm-up.

    The warm row carries the cost numbers and nothing else. Copying the whole
    reported row would repeat the retrieval and faithfulness figures beside a
    cost that was re-derived, and a reader would have no way to see that only
    one of the two moved; nothing here is re-scored, so those cells stay empty.

    Returns ``(rows, added)`` with ``added`` the cells that gained a row, empty
    for a report written before this block existed, so one call serves both.
    """
    cells = ((report or {}).get(field) or {}).get("cells") or {}
    wanted = None if only is None else {str(c) for c in only}
    out: List[Dict[str, Any]] = []
    added: List[str] = []
    for row in rows:
        base = dict(row)
        name = str(base.get(cell_key))
        entry = cells.get(name) or {}
        value = entry.get("cfca")
        pair = value is not None and (wanted is None or name in wanted)
        if cells:
            # Every row is named once the block exists, so a cell that gained no
            # partner still reads as the run's own number and not as an unlabelled
            # third thing.
            base[series_key] = base_series
        out.append(base)
        if not pair:
            continue
        warm_row = {cell_key: f"{name} {label}", value_key: value, series_key: series}
        if entry.get("cost_per_answer") is not None and cost_key in base:
            warm_row[cost_key] = entry.get("cost_per_answer")
        out.append(warm_row)
        added.append(name)
    return out, tuple(added)


def warm_only_columns(
    rows: Sequence[Dict[str, Any]],
    report: Mapping[str, Any],
    *,
    field: str = WARM_FIELD,
    cell_key: str = "cell",
    value_key: str = "cfca",
    warm_key: str = WARM_COLUMN,
    entry_key: str = "cfca",
    keep_keys: Optional[Sequence[str]] = None,
    only: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:
    """One row per cell carrying both costs, reported and warm-only.

    `warm_only_rows` is the table shape: a second row per cell, read down the
    page. This is the chart shape: the warm-only number sits in its own column
    of the cell's own row, so `charts.cost_chart` groups the two bars under one
    cell label instead of spending two axis slots on one cell.

    Args:
        field: report block holding the warm-only cells.
        value_key: the reported cost column already on each row.
        warm_key: column the warm-only cost is written to, which is also the
            legend text the chart prints for that series.
        entry_key: the number read out of the warm-only block.
        keep_keys: columns carried through from the input row. None keeps the
            cell name and the reported cost, which is all the chart reads, and
            leaves out the retrieval and faithfulness columns that the warm-only
            pass never re-derived.
        only: pair just these cells; None pairs every cell the block names.

    Returns:
        ``(rows, added)`` with ``added`` the cells that gained the column, empty
        for a report written before the block existed, so one call serves both.
    """
    cells = ((report or {}).get(field) or {}).get("cells") or {}
    wanted = None if only is None else {str(c) for c in only}
    keep = (cell_key, value_key) if keep_keys is None else tuple(keep_keys)
    out: List[Dict[str, Any]] = []
    added: List[str] = []
    for row in rows:
        name = str(row.get(cell_key))
        new: Dict[str, Any] = {cell_key: row.get(cell_key)}
        new.update({k: row.get(k) for k in keep if k in row})
        value = (cells.get(name) or {}).get(entry_key)
        if value is not None and (wanted is None or name in wanted):
            new[warm_key] = value
            added.append(name)
        out.append(new)
    return out, tuple(added)


def cost_table(
    report: Mapping[str, Any],
    *,
    cost_field: str = COST_FIELD,
    field: str = WARM_FIELD,
    serving_field: str = SERVING_FIELD,
    priced_set: str = PRICED_SET,
    cell_key: str = "cell",
    series_key: str = "series",
    label: str = ROW_LABEL,
    base_series: str = BASE_SERIES,
    series: str = ROW_SERIES,
    order: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """The money rows the viewer prints above the bars: each cell, then it warm.

    A bar carries one number and a cost of 2.2e-05 against 2.2e-05 is two bars
    a reader cannot tell apart, so the pair is printed as well as drawn, at the
    precision the report stores rather than the two significant figures a money
    column rounds to. Every figure is read off the written report; nothing here
    computes a metric.

    Returns [] for a report with no ``cfca_gbp`` block, so the caller can fall
    back to whatever it drew before this existed.
    """
    priced = (report or {}).get(cost_field) or {}
    cells = priced.get("cells") or {}
    if not cells:
        return []
    deltas = priced.get("deltas") or {}
    reference = priced.get("reference")
    warm_block = (report or {}).get(field) or {}
    warm_cells = warm_block.get("cells") or {}
    warm_deltas = warm_block.get("deltas") or {}
    audit = warm_block.get("warmup") or {}
    serving = (report or {}).get(serving_field) or {}
    delta_col = f"dCFCA vs {reference}" if reference else None

    def row(name, tag, entry, against, seconds, dropped):
        out: Dict[str, Any] = {
            cell_key: name if tag == base_series else f"{name} {label}",
            series_key: tag,
            "s/answer": seconds,
            "cost/answer": entry.get("cost_per_answer"),
            "P_hat": entry.get("p_hat"),
            "CFCA": entry.get("cfca"),
        }
        if delta_col:
            out[delta_col] = None if name == reference else (against.get(name) or {}).get("cfca")
        out["dropped"], out["dropped s"], out["share of serving"] = dropped
        return out

    rows: List[Dict[str, Any]] = []
    for name in (order if order is not None else cells):
        entry = cells.get(name)
        if not entry:
            continue
        served = (serving.get(name) or {}).get(priced_set) or {}
        rows.append(row(name, base_series, entry, deltas, served.get("per_answer_s"), (None, None, None)))
        warm_entry = warm_cells.get(name)
        if not warm_entry:
            continue
        seen = audit.get(name) or {}
        ids = "+".join(str(d.get("query_id") or "?") for d in (seen.get("dropped") or [])) or None
        rows.append(row(
            name, series, warm_entry, warm_deltas, seen.get("warm_per_answer_s"),
            (ids, seen.get("dropped_s"), seen.get("dropped_share")),
        ))
    return rows


def cost_columns(
    report: Mapping[str, Any],
    *,
    cost_field: str = COST_FIELD,
    field: str = WARM_FIELD,
    serving_field: str = SERVING_FIELD,
    priced_set: str = PRICED_SET,
    cell_key: str = "cell",
    with_label: str = "with warm-up",
    without_label: str = "without warm-up",
    seconds_key: str = "s/answer",
    cost_key: str = "cost/answer",
    p_hat_key: str = "P_hat",
    cfca_key: str = "CFCA",
    delta_prefix: str = "dCFCA vs ",
    with_delta: bool = True,
    with_dropped: bool = True,
    dropped_keys: Sequence[str] = ("dropped", "dropped s", "share of serving"),
    order: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """One row per cell, the reported price and its warm-up sensitivity side by side.

    `cost_table` prints the same figures as two rows per cell, which reads down
    the page; this reads across, so a cell's pair sits under one label and the
    six cells stay six lines. Every figure is read off the written report and
    nothing here computes a metric.

    Args:
        cost_field / field: report blocks holding the priced and warm-only cells.
        serving_field / priced_set: where the reported per-answer seconds live.
        with_label / without_label: suffixes that name the two columns.
        seconds_key / cost_key / cfca_key: column stems the suffixes attach to.
        p_hat_key: the pass term, written once because the sensitivity cannot
            move it.
        with_delta / delta_prefix: add the reported delta against the reference.
        with_dropped / dropped_keys: add which answer was dropped, its seconds
            and its share of the cell's serving clock.
        order: cell order; None keeps the report's own.

    Returns [] for a report with no ``cfca_gbp`` block, so a caller can fall
    back to whatever it drew before this existed.
    """
    priced = (report or {}).get(cost_field) or {}
    cells = priced.get("cells") or {}
    if not cells:
        return []
    deltas = priced.get("deltas") or {}
    reference = priced.get("reference")
    warm_block = (report or {}).get(field) or {}
    warm_cells = warm_block.get("cells") or {}
    audit = warm_block.get("warmup") or {}
    serving = (report or {}).get(serving_field) or {}
    delta_col = f"{delta_prefix}{reference}" if (with_delta and reference) else None

    rows: List[Dict[str, Any]] = []
    for name in (order if order is not None else cells):
        entry = cells.get(name)
        if not entry:
            continue
        warm_entry = warm_cells.get(name) or {}
        seen = audit.get(name) or {}
        served = (serving.get(name) or {}).get(priced_set) or {}
        row: Dict[str, Any] = {
            cell_key: name,
            p_hat_key: entry.get("p_hat"),
            f"{seconds_key} ({with_label})": served.get("per_answer_s"),
            f"{seconds_key} ({without_label})": seen.get("warm_per_answer_s"),
            f"{cost_key} ({with_label})": entry.get("cost_per_answer"),
            f"{cost_key} ({without_label})": warm_entry.get("cost_per_answer"),
            f"{cfca_key} ({with_label})": entry.get("cfca"),
            f"{cfca_key} ({without_label})": warm_entry.get("cfca"),
        }
        if delta_col:
            row[delta_col] = None if name == reference else (deltas.get(name) or {}).get("cfca")
        if with_dropped:
            ids = "+".join(str(d.get("query_id") or "?") for d in (seen.get("dropped") or [])) or None
            for key, value in zip(dropped_keys, (ids, seen.get("dropped_s"), seen.get("dropped_share"))):
                row[key] = value
        rows.append(row)
    return rows


def warm_only_caption(
    report: Mapping[str, Any], *, field: str = WARM_FIELD, label: str = ROW_LABEL,
) -> str:
    """One line naming the rule and the worst wait it removed, or "" if absent."""
    block = (report or {}).get(field) or {}
    warmup = block.get("warmup") or {}
    if not warmup:
        return ""
    worst, share = max(
        ((name, float(entry.get("dropped_share") or 0.0)) for name, entry in warmup.items()),
        key=lambda pair: pair[1], default=("", 0.0),
    )
    drop = int(block.get("drop") or 1)
    answers = "answer" if drop == 1 else f"{drop} answers"
    line = (f"Each `{label}` bar is its own cell with the first {answers} of the sitting dropped and "
            f"the seconds re-priced: a cold model load is billed to whichever answer waits for it, "
            f"and it is not a property of the condition.")
    if worst and share:
        line += (f" Largest single wait: `{worst}`, {100 * share:.0f}% of that cell's serving time "
                 f"in one answer.")
    return line


def _sub(a: Optional[float], b: Optional[float], round_to: Optional[int]) -> Optional[float]:
    if a is None or b is None:
        return None
    value = float(a) - float(b)
    return value if round_to is None else round(value, round_to)


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or not b:
        return None
    return float(a) / float(b)


# --------------------------------------------------------------------------
# 3. the rows it prints
# --------------------------------------------------------------------------

def render_warmup_markdown(
    block: Mapping[str, Any],
    reported: Optional[Mapping[str, Any]] = None,
    *,
    title: str = SECTION_TITLE,
    label: str = ROW_LABEL,
    order: Optional[Sequence[str]] = None,
    only: Optional[Sequence[str]] = None,
) -> str:
    """The report's own money table, re-rendered with the paired rows added.

    The sensitivity is a row per cell, not a second table: a parallel table
    would reprint the reported cost and CFCA in a second place, and two
    printings of one number are two numbers to keep in step. Every reported row
    keeps its figures and its position, and the cell without its warm-up sits
    directly beneath it, so the pair reads down a single column.

    Args:
        block: `warm_only_block` output.
        reported: the ``cfca_gbp`` block being added to. None returns "", since
            there is no table to extend.
        title: heading of that table.
        label: suffix on the added row's cell name.
        order: row order; None keeps the reported block's own.
        only: which cells gain the extra row; None is every cell covered.
    """
    from veridic_eval.conditions import render_cfca_markdown

    if not ((reported or {}).get("cells") or {}):
        return ""
    return render_cfca_markdown(
        dict(reported), title=title, order=order, warm=dict(block),
        warm_label=label, warm_rows=only,
    )


def drop_section(text: str, title: str, *, level: str = "## ") -> str:
    """Remove a whole ``## `` section, heading included. Absent, nothing changes."""
    lines = (text or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == title.strip())
    except StopIteration:
        return text or ""
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(level)), len(lines))
    return "\n".join(lines[:start] + lines[end:]).rstrip("\n") + "\n"


def splice_section(text: str, section: str, *, title: str = SECTION_TITLE, level: str = "## ") -> str:
    """Append the section, or replace the one already under ``title``.

    Re-running the tool must not stack three copies of its own table in a
    report, so this finds the heading and swaps the body up to the next heading
    of the same level.
    """
    body = section if section.endswith("\n") else section + "\n"
    lines = (text or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == title.strip())
    except StopIteration:
        return (text or "").rstrip("\n") + "\n\n" + body
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith(level)),
        len(lines),
    )
    kept = lines[:start] + body.rstrip("\n").splitlines() + [""] + lines[end:]
    return "\n".join(kept).rstrip("\n") + "\n"


# --------------------------------------------------------------------------
# 4. the command behind `veridic-eval warmup`
# --------------------------------------------------------------------------

def cell_spans_for_report(
    report: Mapping[str, Any],
    cell_dir: str,
    *,
    serving_field: str = SERVING_FIELD,
    priced_set: str = PRICED_SET,
    cost_field: str = COST_FIELD,
    missing: str = "raise",
    printer: Optional[Any] = None,
) -> Dict[str, List[Tuple[str, List[Interval]]]]:
    """Read the dumps this report was scored from and cut them into sittings.

    Each cell is cut on the very fields its own ``serving_seconds`` block names,
    so a cell priced on generation alone is not silently re-priced on the full
    serving sum.

    Args:
        missing: ``raise`` | ``skip`` when a priced cell has no dump file.
        printer: one line per cell read; None stays silent.
    """
    from veridic_eval.cells_app import read_cell

    if missing not in ("raise", "skip"):
        raise ValueError("missing must be 'raise' or 'skip'")
    priced = list(((report or {}).get(cost_field) or {}).get("cells") or {})
    serving = (report or {}).get(serving_field) or {}
    out: Dict[str, List[Tuple[str, List[Interval]]]] = {}
    for name in priced:
        path = os.path.join(cell_dir, f"{name}.json")
        if not os.path.exists(path):
            if missing == "raise":
                raise ValueError(f"no dump for priced cell {name!r} at {path}; point --cell-dir at "
                                 f"the directory this report was scored from")
            continue
        fields = ((serving.get(name) or {}).get(priced_set) or {}).get("fields")
        records = read_cell(path, expect_cell=name)
        spans = sitting_spans(records, latency_fields=tuple(fields or SERVING_LATENCY_FIELDS))
        out[name] = spans
        if printer:
            printer(f"  {name}: {sum(len(ivs) for _, ivs in spans)} priced answers over "
                    f"{len(spans)} sitting(s), on {'+'.join(fields or SERVING_LATENCY_FIELDS)}")
    return out


def backfill_report(
    report_path: str,
    cell_dir: str,
    *,
    drop: int = 1,
    field: str = WARM_FIELD,
    md_path: Optional[str] = None,
    write: bool = True,
    title: str = SECTION_TITLE,
    indent: int = 2,
    printer: Optional[Any] = None,
    **block_kwargs: Any,
) -> Dict[str, Any]:
    """Compute the warm-only block for a written report and store it beside it.

    The report keeps every figure it had: this adds one JSON field and one row
    per cell on the money table it was derived from, and re-running replaces
    both rather than stacking them. No reported number is edited or moved.

    Args:
        report_path: the ``report.json`` `run` wrote.
        cell_dir: the dumps it was scored from.
        md_path: report.md whose CFCA table gains the rows; None derives it
            from ``report_path``, "" skips the markdown entirely.
        write: False computes and returns the block without touching disk.
        indent: JSON indent, matching the one `pipeline` wrote the report with,
            so adding a field does not reformat every other line of it.
        printer: progress lines; None stays silent.

    Returns:
        The block, as it was written under ``field``.
    """
    say = printer or (lambda *_: None)
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    say(f"Reading the dumps under {cell_dir}:")
    spans = cell_spans_for_report(report, cell_dir, printer=printer)
    block = warm_only_block(
        report, spans, drop=drop,
        source=os.path.abspath(cell_dir),
        **block_kwargs,
    )
    report[field] = block

    if write:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=indent, default=str)
            fh.write("\n")
        say(f"wrote {field} into {report_path}")

    md = md_path
    if md is None:
        md = os.path.splitext(report_path)[0] + ".md"
    if md and write:
        text = ""
        if os.path.exists(md):
            with open(md, "r", encoding="utf-8") as fh:
                text = fh.read()
        section = render_warmup_markdown(block, report.get(COST_FIELD), title=title)
        if not section:
            say(f"{md}: no CFCA table to add the rows to; left alone")
            return block
        text = drop_section(text, LEGACY_TITLE)
        with open(md, "w", encoding="utf-8") as fh:
            fh.write(splice_section(text, section, title=title))
        say(f"added the `{ROW_LABEL}` rows to {md}")
    return block
