"""
Where every dashboard number comes from: file:line, function, equation.

`column_config` hands streamlit one help string per column, so hovering a column
name in any table shows the source line that computed it, the function it sits
in, and the arithmetic that line runs. `SOURCES` is the only place a claim about
the code lives, so a wrong claim is one diff away from the line it names.

Nothing is scored here. The "this run" values are read off the report `run`
committed, which is the same file the tables are drawn from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricSource:
    """One column: the line that computes it, and what that line computes.

    file / line / func: the definition site, as `file:line func()`.
    equation: the arithmetic that line runs, in its own variable names.
    chain: (file, line, what) for each earlier line the equation depends on,
        in the order the value is built, so a reader can walk it downwards.
    live: dotted paths into the report for the inputs this run used; a path is
        tried at the report root first, then inside every cell.
    note: what the number does not claim.
    """

    file: str
    line: int
    func: str
    equation: str
    chain: Tuple[Tuple[str, int, str], ...] = ()
    live: Tuple[str, ...] = ()
    note: str = ""

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"


_IR = (
    ("retrieval_eval.py", 51, "qrels[qid] = {chunk_id: 1 for chunk_id in judged_chunk_ids}"),
    ("retrieval_eval.py", 55, "run[qid] = {chunk_id: 1/(served_rank+1)}, served order"),
    ("retrieval_eval.py", 75, "k = settings.ir_k (config.py:82, default 5)"),
    ("retrieval_eval.py", 89, "ranx_evaluate(qrels, run, metrics, return_mean=True)"),
)
_BOOT = (
    ("stats.py", 100, "scipy bootstrap(paired=True, method='BCa'), stats.py:100-108"),
    ("stats.py", 104, "n_resamples / confidence_level / random_state = settings.bootstrap_resamples / "
                      "confidence_level / random_state (config.py:124-126, defaults 9999 / 0.95 / 42)"),
)

#: column name -> its source line. ``name@k`` columns are keyed as ``name@k``.
SOURCES: Dict[str, MetricSource] = {
    # ------------------------------------------------------------ per cell
    "cell": MetricSource(
        "cells_app.py", 1252, "cells_table",
        'row["cell"] = the conditions.yaml cell name this row was scored under',
        live=("reference",),
    ),
    "linked": MetricSource(
        "cells_app.py", 1253, "cells_table",
        'linked = f"{n_linked}/{n_queries}"',
        chain=(
            ("report.py", 101, "linked = [r for r in records if r.linked]"),
            ("report.py", 117, "n_linked = len(linked)"),
            ("extract.py", 182, "rec.linked is False when no answer matched this cell"),
        ),
    ),
    "judged": MetricSource(
        "cells_app.py", 1254, "cells_table",
        'judged = cell["retrieval"]["n_judged"]',
        chain=(
            ("retrieval_eval.py", 47, "a query is judged only when judged_chunk_ids is non-empty"),
            ("retrieval_eval.py", 107, "n_judged = len(judged)"),
            ("gold_evidence.py", 585, "apply_gold_to_records writes judged_chunk_ids at dump time"),
        ),
    ),
    "recall@k": MetricSource(
        "retrieval_eval.py", 89, "evaluate_retrieval",
        "recall@k = |gold AND served[:k]| / |gold|, meaned over judged queries (ranx)",
        chain=_IR, live=("retrieval.k", "retrieval.n_judged"),
    ),
    "mrr@k": MetricSource(
        "retrieval_eval.py", 89, "evaluate_retrieval",
        "mrr@k = mean(1 / rank of the first gold chunk, 0 when none in the top k) (ranx)",
        chain=_IR, live=("retrieval.k",),
    ),
    "ndcg@k": MetricSource(
        "retrieval_eval.py", 89, "evaluate_retrieval",
        "ndcg@k = mean(DCG@k / IDCG@k), binary gain from the qrels (ranx)",
        chain=_IR, live=("retrieval.k",),
    ),
    "hit_rate@k": MetricSource(
        "retrieval_eval.py", 89, "evaluate_retrieval",
        "hit_rate@k = mean(1 if any gold chunk is in served[:k] else 0) (ranx)",
        chain=_IR, live=("retrieval.k",),
    ),
    "faithful_rate": MetricSource(
        "faithfulness.py", 528, "score_faithfulness",
        "faithful_rate = sum(faithful) / len(faithful)",
        chain=(
            ("faithfulness.py", 41, "_scorable = linked and answerable and has_evidence and answer_text.strip()"),
            ("faithfulness.py", 341, "LettuceDetect predict(context, question, answer) -> spans"),
            ("faithfulness.py", 347, "strong = spans with confidence >= settings.lettucedetect_threshold (config.py:92, default 0.5)"),
            ("faithfulness.py", 351, "faithful = len(strong) <= faithful_max_spans, which is 0 unless a caller raises it"),
        ),
        live=("faithfulness.backend", "faithfulness.n_scored"),
        note="per-answer verdicts stay in report.json for faithfulness alone (report.py:120-123); retrieval and abstention drop per_query (report.py:119, 124).",
    ),
    "faithfulness": MetricSource(
        "faithfulness.py", 527, "score_faithfulness",
        "faithfulness = mean(1 - flagged_chars / len(answer)) over scored answers",
        chain=(
            ("faithfulness.py", 349, "flagged = sum(span.end - span.start) over strong spans"),
            ("faithfulness.py", 356, "faithfulness = max(0.0, 1.0 - flagged/answer_len)"),
        ),
        live=("faithfulness.backend", "faithfulness.n_scored"),
    ),
    "abstention_recall": MetricSource(
        "abstention.py", 190, "score_abstention",
        "abstention_recall = refusals_on_unanswerable / unanswerable",
        chain=(
            ("abstention.py", 51, r"marker regex, anchored: ^\s*<settings.abstention_marker>\s*$"),
            ("abstention.py", 46, "_REFUSAL_RE = the deterministic refusal-keyword net (abstention.py:35-45)"),
            ("abstention.py", 69, "marker_match = int(anchored marker OR settings.abstention_substring in answer)"),
            ("abstention.py", 178, "refused = int(marker_match or is_refusal)"),
        ),
        live=("abstention.tool", "abstention.marker", "abstention.counts.unanswerable"),
        note="promptfoo signals it when its CLI answers (abstention.py:163-166), the python net otherwise (abstention.py:168); `tool` above says which ran.",
    ),
    "abstention_precision": MetricSource(
        "abstention.py", 191, "score_abstention",
        "abstention_precision = refusals_on_unanswerable / refusals_total",
        live=("abstention.tool", "abstention.counts.refusals_total"),
    ),
    "over_abstention_rate": MetricSource(
        "abstention.py", 192, "score_abstention",
        "over_abstention_rate = wrongly_refused_answerable / answerable",
        chain=(
            ("abstention.py", 178, "refused = int(marker_match or is_refusal), same decision as recall"),
            ("abstention.py", 187, "wrongly_refused_ans += refused, over answerable items only"),
        ),
        live=("abstention.tool", "abstention.counts.answerable"),
    ),
    "cfca": MetricSource(
        "cfca_metric.py", 349, "compute_cfca",
        "cfca = mean(cost_col) / P(faithful AND cited AND right_version), over answerable linked items",
        chain=(
            ("cfca_metric.py", 303, "answerable = [r for r in records if r.linked and r.answerable]"),
            ("cfca_metric.py", 320, "joint = int(faithful and cited and version) per question"),
            ("cfca_metric.py", 45, "cited = int(rec.has_evidence)"),
            ("cfca_metric.py", 48, "right_version = 1 unless a declared governing doc was mis-served"),
            ("cfca_metric.py", 331, "p_joint = sum(joint) / n"),
            ("cfca_metric.py", 335, "cost_num = mean of the per-answer cost column"),
            ("cfca_cost.py", 69, "Cost_per_answer = C_onetime/A + C_query + (U*C_update + M*index_gb*p_store)/Q"),
            ("cfca_cost.py", 31, "p_gpu_hour = (watts / 1000) * kwh_gbp"),
        ),
        live=("cfca.cost_source", "cfca.n_answerable", "cfca.p_faithful_cited_version", "cfca.currency"),
        note="None when the denominator is 0 (cfca_metric.py:345); index build is capex, priced apart.",
    ),
    # -------------------------------------------------------- deltas / off-grid
    "contrast": MetricSource(
        "cells.py", 347, "contrast_key",
        'contrast = f"{cell}_vs_{reference}", one row per scored cell',
        live=("reference", "offgrid_reference"),
    ),
    "metric": MetricSource(
        "report.py", 425, "_DELTA_METRICS",
        "metric = the per-question column this row's delta was taken on",
    ),
    "delta": MetricSource(
        "stats.py", 90, "paired_delta_ci",
        "delta = mean(cell) - mean(reference), over query ids scored in both cells",
        chain=(
            ("stats.py", 68, "_align keeps common query ids only, same order"),
            ("report.py", 464, "paired_deltas calls paired_delta_ci per metric column"),
            ("report.py", 544, "cfca_delta_block routes the cfca row through paired_ratio_delta_ci instead"),
            ("stats.py", 289, "that row: ra = mean(cost_a)/mean(joint_a), rb = mean(cost_b)/mean(joint_b)"),
            ("stats.py", 294, "that row: delta = ra - rb, resampled as one ratio"),
        ),
    ),
    "ci_low": MetricSource(
        "stats.py", 109, "paired_delta_ci",
        "ci_low = bootstrap(paired=True, method='BCa').confidence_interval.low",
        chain=_BOOT,
        note="a spread of 0 returns the point estimate as both ends, method='degenerate' (stats.py:91-93).",
    ),
    "ci_high": MetricSource(
        "stats.py", 110, "paired_delta_ci",
        "ci_high = bootstrap(paired=True, method='BCa').confidence_interval.high",
        chain=_BOOT,
    ),
    "perm_p": MetricSource(
        "stats.py", 138, "paired_permutation_p",
        "perm_p = permutation_test(mean(x - y), permutation_type='samples', two-sided).pvalue",
        note="uncorrected; the corrected column beside it is perm_p_bonferroni (report.py:471).",
    ),
    "perm_p_bonferroni": MetricSource(
        "stats.py", 153, "bonferroni",
        "perm_p_bonferroni = min(1.0, perm_p * n_contrasts)",
        chain=(("report.py", 471, "paired_deltas applies it with the grid family size"),),
        live=("n_contrasts",),
    ),
    "significant": MetricSource(
        "stats.py", 116, "paired_delta_ci",
        "significant = bool(ci_low > 0 or ci_high < 0)",
        note="read off the CI, not off the p-value.",
    ),
    "n": MetricSource(
        "stats.py", 112, "paired_delta_ci",
        "n = number of query ids scored in both cells",
        chain=(("stats.py", 68, "_align drops any id missing or None on either side"),),
    ),
    # ------------------------------------------------------------- per query
    "query_id": MetricSource(
        "extract.py", 134, "QueryRecord",
        "query_id = the benchmark question id this row answers",
    ),
    "answerable": MetricSource(
        "extract.py", 137, "QueryRecord",
        "answerable = the benchmark flag; False items are the abstention set",
    ),
    "gold": MetricSource(
        "cells_app.py", 1388, "query_table",
        "gold = len(set(rec.judged_chunk_ids))",
        chain=(("gold_evidence.py", 585, "apply_gold_to_records matches gold text against this ingest's chunks"),),
    ),
    "served": MetricSource(
        "cells_app.py", 1389, "query_table",
        "served = len(rec.served_chunk_ids)",
        chain=(("extract.py", 185, "served_chunk_ids = [e.chunk_id for e in served_evidence]"),),
    ),
    "hits": MetricSource(
        "cells_app.py", 1390, "query_table",
        "hits = sum(1 for c in served if c in gold)",
    ),
    "first_hit_rank": MetricSource(
        "cells_app.py", 1391, "query_table",
        "first_hit_rank = 1-based position of the first served chunk that is gold, None if none",
    ),
    "question": MetricSource(
        "extract.py", 136, "QueryRecord",
        "question = the benchmark question text served to the app",
    ),
}


# --------------------------------------------------------------- resolution
def resolve_source(
    column: str,
    *,
    registry: Mapping[str, MetricSource] = SOURCES,
    at_key: str = "@k",
) -> Optional[MetricSource]:
    """The entry for one column name, or None.

    An ``@`` column is looked up twice: verbatim, then with its cutoff replaced
    by ``at_key``, so ``recall@5`` and ``recall@10`` both find ``recall@k``.
    """
    hit = registry.get(column)
    if hit is not None or "@" not in column:
        return hit
    return registry.get(column.split("@", 1)[0] + at_key)


def _dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(obj, Mapping) or part not in obj:
            return None
        obj = obj[part]
    return obj


def live_values(
    report: Optional[Mapping[str, Any]],
    path: str,
    *,
    cells_key: str = "cells",
) -> Dict[str, Any]:
    """``{scope: value}`` for one dotted path: ``{"": v}`` at the report root,
    else one entry per cell that carries it. Empty when nothing holds it."""
    if not isinstance(report, Mapping):
        return {}
    root = _dig(report, path)
    if root is not None:
        return {"": root}
    cells = report.get(cells_key)
    if not isinstance(cells, Mapping):
        return {}
    found = {name: _dig(cell, path) for name, cell in cells.items()}
    return {k: v for k, v in found.items() if v is not None}


def live_line(
    report: Optional[Mapping[str, Any]],
    paths: Sequence[str],
    *,
    cells_key: str = "cells",
    on_disagree: str = "list",
    max_cells: int = 4,
    sep: str = ", ",
    label_of=lambda path: path.rsplit(".", 1)[-1],
) -> str:
    """This run's inputs for one column, as ``key=value`` pairs.

    Args:
        paths: dotted report paths, in the order they should read.
        on_disagree: cells holding different values for one path.
            ``list`` prints ``cell=value`` up to ``max_cells`` then ``...``,
            ``first`` prints one value, ``omit`` drops the path.
        max_cells: cap on the listed cells; 0 lists none.
        label_of: path -> the name printed before ``=``.
    """
    if on_disagree not in ("list", "first", "omit"):
        raise ValueError(f"on_disagree must be list|first|omit, got {on_disagree!r}")
    out: List[str] = []
    for path in paths:
        found = live_values(report, path, cells_key=cells_key)
        if not found:
            continue
        name = label_of(path)
        values = list(found.values())
        if len(set(map(str, values))) == 1:
            out.append(f"{name}={values[0]}")
            continue
        if on_disagree == "omit":
            continue
        if on_disagree == "first":
            out.append(f"{name}={values[0]}")
            continue
        shown = [f"{c or 'report'}={v}" for c, v in list(found.items())[:max_cells]]
        if len(found) > max_cells:
            shown.append("...")
        out.append(f"{name}: {sep.join(shown)}")
    return sep.join(out)


# ------------------------------------------------------------------ rendering
def help_text(
    source: MetricSource,
    *,
    report: Optional[Mapping[str, Any]] = None,
    with_equation: bool = True,
    with_chain: bool = True,
    with_live: bool = True,
    with_note: bool = True,
    chain_limit: Optional[int] = None,
    sep: str = "\n\n",
    bullet: str = "- ",
    live_prefix: str = "this run: ",
    max_chars: Optional[int] = None,
    ellipsis: str = " ...",
    **live_kwargs: Any,
) -> str:
    """One column's tooltip: the source line, its equation, its chain, its inputs.

    Args:
        report: the committed report, for the live inputs; None omits them.
        with_*: drop any block.
        chain_limit: keep only the first N chain lines; None keeps all.
        sep / bullet / live_prefix: block separator, chain marker, inputs label.
        max_chars: truncate the finished string; None never truncates.
        live_kwargs: passed to `live_line` (cells_key, on_disagree, max_cells).
    """
    blocks = [f"**{source.where}** `{source.func}()`"]
    if with_equation and source.equation:
        blocks.append(f"`{source.equation}`")
    if with_chain and source.chain:
        rows = source.chain if chain_limit is None else source.chain[:chain_limit]
        blocks.append("\n".join(f"{bullet}{f}:{ln} {what}" for f, ln, what in rows))
    if with_live and report is not None and source.live:
        line = live_line(report, source.live, **live_kwargs)
        if line:
            blocks.append(f"{live_prefix}{line}")
    if with_note and source.note:
        blocks.append(source.note)
    text = sep.join(blocks)
    if max_chars is not None and len(text) > max_chars:
        text = text[: max(0, max_chars - len(ellipsis))] + ellipsis
    return text


def columns_of(
    rows: Any,
    *,
    columns_from: str = "union",
) -> List[str]:
    """Column names, in first-seen order, from rows or from a plain name list.

    columns_from: ``union`` walks every row, ``first_row`` reads only row 0.
    """
    if isinstance(rows, Mapping):
        rows = [rows]
    if hasattr(rows, "columns"):  # a dataframe hands its own names over
        return [str(c) for c in rows.columns]
    if rows is None:
        return []
    if not isinstance(rows, (list, tuple)):
        return [str(c) for c in rows]
    if not rows:
        return []
    if all(isinstance(r, str) for r in rows):
        return list(rows)
    seen: Dict[str, None] = {}
    for row in (rows[:1] if columns_from == "first_row" else rows):
        if isinstance(row, Mapping):
            for key in row:
                seen.setdefault(str(key), None)
    return list(seen)


def column_help(
    rows: Any,
    *,
    report: Optional[Mapping[str, Any]] = None,
    registry: Mapping[str, MetricSource] = SOURCES,
    at_key: str = "@k",
    columns_from: str = "union",
    unknown: str = "skip",
    unknown_text: str = "no `provenance.SOURCES` entry names the line that computes this column",
    **help_kwargs: Any,
) -> Dict[str, str]:
    """``{column: tooltip}`` for the columns present, unknown ones per ``unknown``.

    Args:
        rows: the table rows, or a list of column names.
        report: passed to `help_text` for the live inputs.
        registry / at_key: the source table and its ``name@k`` fallback.
        unknown: ``skip`` leaves the column bare, ``label`` writes
            ``unknown_text``, ``raise`` refuses a column with no entry.
        help_kwargs: passed to `help_text`.
    """
    if unknown not in ("skip", "label", "raise"):
        raise ValueError(f"unknown must be skip|label|raise, got {unknown!r}")
    out: Dict[str, str] = {}
    for name in columns_of(rows, columns_from=columns_from):
        source = resolve_source(name, registry=registry, at_key=at_key)
        if source is not None:
            out[name] = help_text(source, report=report, **help_kwargs)
        elif unknown == "label":
            out[name] = unknown_text
        elif unknown == "raise":
            raise KeyError(f"no provenance entry for column {name!r}")
    return out


def column_config(
    st,
    rows: Any,
    *,
    report: Optional[Mapping[str, Any]] = None,
    on_missing_api: str = "skip",
    **help_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """`st.dataframe(column_config=...)` carrying one provenance tooltip per column.

    Only ``help`` is set, so column widths, order and number formatting stay
    whatever streamlit already picked.

    Args:
        st: the streamlit module, passed in so this stays importable headless.
        rows: the same rows handed to `st.dataframe`.
        report: the committed report, for the live inputs.
        on_missing_api: streamlit too old for ``st.column_config``. ``skip``
            returns None and the table draws as before, ``raise`` refuses.
        help_kwargs: passed to `column_help` (unknown, chain_limit, max_chars...).
    """
    config_api = getattr(st, "column_config", None)
    if config_api is None:
        if on_missing_api == "raise":
            raise AttributeError("this streamlit has no st.column_config")
        return None
    helps = column_help(rows, report=report, **help_kwargs)
    if not helps:
        return None
    return {name: config_api.Column(name, help=text) for name, text in helps.items()}


# ------------------------------------------------------- one answer's gates
@dataclass(frozen=True)
class RecordGate:
    """One decision taken about a single answer, and what it reads to take it.

    label: the cell number this decision feeds.
    file / line / func: the line that runs the rule.
    rule: the boolean or arithmetic that line runs.
    reads: QueryRecord attribute names the rule reads, printed back per record.
    """

    label: str
    file: str
    line: int
    func: str
    rule: str
    reads: Tuple[str, ...] = ()

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line} {self.func}()"


#: the gates that decide whether one answer enters each cell number.
RECORD_GATES: Tuple[RecordGate, ...] = (
    RecordGate(
        "retrieval (recall/mrr/ndcg/hit_rate)", "retrieval_eval.py", 47, "build_qrels_run",
        "judged only when judged_chunk_ids is non-empty; scored against served order",
        ("judged_chunk_ids", "served_chunk_ids"),
    ),
    RecordGate(
        "faithfulness / faithful_rate", "faithfulness.py", 41, "_scorable",
        "linked and answerable and has_evidence and answer_text.strip()",
        ("linked", "answerable", "has_evidence", "answer_text"),
    ),
    RecordGate(
        "abstention (recall / precision / over)", "abstention.py", 54, "_scorable",
        "linked and answer_text is not None; the unanswerable set is answerable == False",
        ("linked", "answerable", "answer_text"),
    ),
    RecordGate(
        "abstention refusal decision", "abstention.py", 178, "score_abstention",
        r"refused = marker_match or is_refusal, marker anchored as ^\s*<marker>\s*$ (abstention.py:51)",
        ("answer_text",),
    ),
    RecordGate(
        "cfca denominator: answerable set", "cfca_metric.py", 303, "compute_cfca",
        "linked and answerable",
        ("linked", "answerable"),
    ),
    RecordGate(
        "cfca denominator: cited", "cfca_metric.py", 45, "is_cited",
        "cited = int(rec.has_evidence)",
        ("has_evidence",),
    ),
    RecordGate(
        "cfca denominator: right_version", "cfca_metric.py", 48, "is_right_version",
        "1 with no governing_doc; 0 when governing_doc was not served or ranks below superseded_doc "
        "(cfca_metric.py:50-59)",
        ("governing_doc", "superseded_doc", "served_documents"),
    ),
    RecordGate(
        "cfca numerator: this answer's cost", "cfca_metric.py", 335, "compute_cfca",
        "cost_num = mean of the per-answer cost column; the serving span is answered_at and latency",
        ("tokens_input", "tokens_output", "latency_ms", "answered_at"),
    ),
)


def _read_value(rec: Any, name: str) -> Any:
    if name == "served_documents":
        return [e.document_name for e in getattr(rec, "served_evidence", [])]
    return getattr(rec, name, None)


def _fmt_read(name: str, value: Any, *, max_chars: int) -> str:
    if isinstance(value, str) and name.endswith("_text"):
        return f"{name}: {len(value)} chars"
    if isinstance(value, (list, tuple, set)):
        shown = ", ".join(str(v) for v in list(value)[:3])
        tail = "..." if len(value) > 3 else ""
        return f"{name}[{len(value)}]" + (f" = {shown}{tail}" if shown else "")
    text = str(value)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return f"{name}={text}"


def record_gates(
    rec: Any,
    *,
    gates: Sequence[RecordGate] = RECORD_GATES,
    labels: Optional[Sequence[str]] = None,
    read_value=_read_value,
    fmt_read=_fmt_read,
    max_chars: int = 60,
    sep: str = ", ",
    columns: Tuple[str, str, str, str] = ("number", "rule", "code", "this answer"),
) -> List[Dict[str, str]]:
    """One row per decision taken about ``rec``: the rule, its line, its inputs.

    The inputs are read off the record and printed. No rule is evaluated here, so
    a row can never disagree with the run that scored the answer; it says which
    line to read and what that line was handed.

    Args:
        gates: the decisions to print, in order.
        labels: keep only these gate labels; None keeps all.
        read_value / fmt_read: how a named input is pulled off the record and
            rendered; override to print a field this module does not know.
        max_chars: per-value truncation.
        columns: the four column names of the returned rows.
    """
    num, rule, code, reads = columns
    out: List[Dict[str, str]] = []
    for gate in gates:
        if labels is not None and gate.label not in labels:
            continue
        seen = sep.join(
            fmt_read(name, read_value(rec, name), max_chars=max_chars) for name in gate.reads
        )
        out.append({num: gate.label, rule: gate.rule, code: gate.where, reads: seen})
    return out
