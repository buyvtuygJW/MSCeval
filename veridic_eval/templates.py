"""
Starter yamls for `veridic-eval init`, plus the writer that lays them down.

`init` must always leave a fillable benchmark.yaml / conditions.yaml behind, so
the text of both *.example.yaml files is kept here verbatim as a last-resort
source: an installed wheel, a wrong working directory, or a deleted example must
not turn init into a no-op. Search order is caller-controlled and every edge
case (destination, overwrite, embedded fallback, printing) is an explicit arg.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .cells import (
    DEFAULT_REFERENCE,
    GRID_ORDER,
    OFFGRID_ORDER,
    REFERENCE_YAML_KEYS,
    label,
)
from .conditions import (
    CONDITIONS_KEY,
    COST_DEFAULTS_KEY,
    COST_KEY,
    DEVICE_KEY,
    GRID_KEY,
    INGEST_KEY,
    OFFGRID_KEY,
    POWER_KEY,
    RUN_ID_KEY,
    RUNS_KEY,
    SNAPSHOT_KEY,
)

BENCHMARK_TEMPLATE = """\
# Example benchmark (gold set). Copy to benchmark.yaml and fill in.
# Only `id` and `text` are required per query. `answerable` defaults to true.
#
# The `text` must match the user message logged by the app verbatim (after
# whitespace/case normalisation) so the eval can link it to the served answer,
# or set `message_ids: {<condition>: <assistant_message_uuid>}` to link explicitly.
#
# Judge evidence with `gold_evidence` (document + page + the exact quoted text),
# never with raw `chunks.id` UUIDs: the app writes fresh chunk rows on every ingest
# and the 2x2 flips the chunker, so a UUID list from one cell is wrong in the other
# three. `veridic_eval/gold_evidence.py` matches that text against whichever ingest
# is live when a cell is dumped. `document` must equal `documents.filename`;
# `page` is 1-based metadata, the quoted text decides the match. A `gold_chunk_ids`
# key is rejected at load, so nothing here can carry ids across a re-ingest.

queries:
  # Two pages of one fire risk assessment: the door ratings are stated in the
  # survey text, the thirty-minute floor in the recommendations.
  - id: q001
    text: "What fire door rating does the assessment record for the doors to the stairwell?"
    category: direct_lookup
    answerable: true
    gold_answer: "Nominal FD30 to the stairwell, thirty minutes minimum (FD60 to the flat lobbies)."
    gold_evidence:
      - document: "FRA2.pdf"
        page: 22
        text: "The corridor and lobby doors appear to be a mixture of nominal FD30 rated doors to the stairwell and FD60 rated doors to the flat lobbies"
      - document: "FRA2.pdf"
        page: 39
        text: "provide adequate and a minimum of thirty minutes fire protection"

  # The interval and the condition attached to it sit in one front-page block.
  - id: q002
    text: "When does the EICR say the installation must next be inspected and tested, and on what condition?"
    category: conditional_obligation
    answerable: true
    gold_answer: "In 5 years, subject to the necessary remedial action being taken."
    gold_evidence:
      - document: "EICR 2.pdf"
        page: 1
        text: "recommend that this installation is further inspected and tested in 5 Years Subject to the necessary remedial action being taken"

  # The three below are real quotes from `EICR 1_img.pdf` (20 pages, serial
  # 30237785), so they resolve against the live ingest without editing. Rename
  # `document` if your DB stores that file under another filename. That file is
  # a scan, and the OCR of its front page drops the postcode line and mangles
  # single characters (`I/We` loses its leading glyph, `Due to` comes through as
  # `PU to`), so every quote below stops short of the mangled runs.

  # Single field on the front page. Cheapest possible retrieval: one page, one quote.
  - id: q003
    text: "What is the address of the installation covered by the EICR?"
    category: direct_lookup
    answerable: true
    gold_answer: "Bryn Morlais Court, Heol Gwyr, Swansea."
    gold_evidence:
      - document: "EICR 1_img.pdf"
        page: 1
        text: "Bryn Morlais Court"

  # Date the answer depends on: PART 4 states the recommended next inspection,
  # and the reason for it sits in the adjacent field.
  - id: q004
    text: "By what date does the EICR recommend the next inspection and test?"
    category: effective_date
    answerable: true
    gold_answer: "21/09/2029, recommended because of the observations on page 2."
    gold_evidence:
      - document: "EICR 1_img.pdf"
        page: 1
        text: "inspected and tested by"
      - document: "EICR 1_img.pdf"
        page: 1
        text: "recommendations made on page 2"

  # Two hops across 18 pages: the observation and its code sit in PART 5, the
  # risk level and required action are defined only in the recipient guidance.
  - id: q005
    text: "The consumer unit is fitted behind a fixed panel. What risk does the EICR assign to that and what action does it require?"
    category: risk_and_remedy
    answerable: true
    gold_answer: "Code C3, improvement recommended: no immediate or potential danger, but remedying it gives a significant safety improvement, so it is not urgent remedial work."
    gold_evidence:
      - document: "EICR 1_img.pdf"
        page: 10
        text: "Consumer unit fitted behind fixed panel awkward to gain Access for maintenance & Testing."
      - document: "EICR 1_img.pdf"
        page: 2
        text: "Code C3 Improvement Recommended"
      - document: "EICR 1_img.pdf"
        page: 20
        text: "whilst not presenting immediate or potential danger, would result in a significant safety improvement if remedied"

  # The three below are real quotes from `HandS1_img.pdf` (46 pages, report
  # L-333958, 28 Willowcroft Close assessed 10/08/2021), same rename rule.

  # Numbers that only exist inside a table: the count and the per-priority split
  # sit in separate cells, so a chunker that shreds the table loses the answer.
  - id: q006
    text: "How many actions did the health and safety report raise in total, and how many must be completed within 3 months?"
    category: table_lookup
    answerable: true
    gold_answer: "18 in total: 0 priority 1, 9 priority 2 (within 3 months), 8 priority 3, 1 advisory."
    gold_evidence:
      - document: "HandS1_img.pdf"
        page: 7
        text: "In addition, the number of actions raised in the report is as follows:"
      - document: "HandS1_img.pdf"
        page: 7
        text: "Total number of actions identified:"

  # Three hops: the defect and its risk rating are in section 4.2, the rating to
  # timeframe mapping is in section 5.0, the priority list is in section 5.3.
  - id: q007
    text: "The fire door outside flat 7 catches on its frame. How soon must that be fixed and what work is recommended?"
    category: priority_timescale
    answerable: true
    gold_answer: "Hazard 10 is rated LOW, so it is a priority 3 action: start within 6 months and complete within 1 year, by re-hanging the door or adjusting the hinges or the self-closing mechanism."
    gold_evidence:
      - document: "HandS1_img.pdf"
        page: 33
        text: "Fire door is catching / sticking on frame and does not close fully."
      - document: "HandS1_img.pdf"
        page: 39
        text: "To assist you, the following timeframes are suggested:"
      - document: "HandS1_img.pdf"
        page: 41
        text: "5.3 PRIORITY 3 ACTIONS"

  # Scope question: the honest answer is a documented exclusion, not a refusal.
  # Separates "the corpus says no" from the abstention class below.
  - id: q008
    text: "Does the assessment cover the inside of the individual flats?"
    category: scope_limitation
    answerable: true
    gold_answer: "No. It is a Type 1 non-invasive inspection of the common parts only, and the roof and tenants' flats are listed as excluded areas."
    gold_evidence:
      - document: "HandS1_img.pdf"
        page: 6
        text: "This report is based on a Type 1 fire risk assessment, it is a non-invasive inspection of the common parts of the property only and as such does not examine the internal aspect of individual flats"
      - document: "HandS1_img.pdf"
        page: 5
        text: "Roof. Tenants flats"

  # Two hops 18 pages apart: PART 5 assigns the code to an observation, and only
  # the recipient guidance on the last page says what the code obliges.
  - id: q009
    text: "What does a C1 observation on this EICR mean, and how urgently must it be acted on?"
    category: risk_and_remedy
    answerable: true
    gold_answer: "C1 means danger is present with a risk of injury, so it requires immediate remedial action, without delay."
    gold_evidence:
      - document: "EICR 1_img.pdf"
        page: 2
        text: "Code C1 Danger present"
      - document: "EICR 1_img.pdf"
        page: 20
        text: "Classification code C1 (Danger present)"

  # Unanswerable item: no supporting evidence exists in the corpus.
  # Positive class for the abstention protocol; must be refused.
  - id: q020
    text: "What is the asbestos survey result for the neighbouring building?"
    category: unanswerable
    answerable: false
"""

CONDITIONS_HEADER = """\
# The four 2x2 cells, separated by time windows (the app has no `condition`
# column and we make no schema change). Run each cell live in its own window,
# then point the eval at these ranges. The `cost:` block prices each answer in
# GBP off the measured watts, so a local/Ollama run reaches a non-degenerate CFCA
# with no API bill; `per_answer_cost` (optional, USD) adds a provider's bill on
# top, converted at the rate in cfca_cost.py. Cell names and the reference
# cell come from veridic_eval/cells.py; `prebaseline` (the app before the
# chunking correction) can be added here as a fifth, off-grid window."""

CONDITIONS_HEADER_WITH_OFFGRID = """\
# The four 2x2 cells and the off-grid `prebaseline` window, separated by time
# windows (the app has no `condition` column and we make no schema change). Run
# each cell live in its own window, then point the eval at these ranges.
# The `cost:` block prices each answer in GBP off the measured watts, so a
# local/Ollama run reaches a non-degenerate CFCA with no API bill;
# `per_answer_cost` (optional, USD) adds a provider's bill on top, converted at
# the rate in cfca_cost.py. Cell names and the reference cell come from
# veridic_eval/cells.py."""

OFFGRID_HEADER = """\
  # Off the grid, so it is measured but never scored inside the 2x2. Run this
  # window FIRST: it is the app before the one-time baseline construction, and
  # the re-ingest that the chunking correction forces deletes the served
  # evidence rows it needs (dump it as soon as its conversations are done).
  # Its delta against the reference is the before/after check on that
  # construction, reported uncorrected and outside the Bonferroni family."""

DECLARATION_HEADER = """\
# The single declaration of the experiment: which cells are scored in the 2x2,
# which window marks each one, whether a cell is dumped to json before the next
# ingest, and the GBP cost quantities behind the CFCA. `veridic-eval init`
# writes this file, `veridic-eval run` reads it.
#
# The app has no `condition` column and we make no schema change, so a cell is
# a slice of the logs. Each sitting is one bounded entry under the cell's `runs:`
# list, and a repeat is a second entry whose scores average per question. A cell is
# selected by `conversation_ids: [<uuid>, ...]` (`veridic-eval init` lists the
# recent ones) inside the entry, or by its window. Whatever an entry declares
# must ALL match, so leave the others commented out.
# `per_answer_cost` (USD) is a provider's API bill per answer, converted and
# added to the GBP the `cost:` block prices from the measured watts. Cell names
# come from veridic_eval/cells.py.
#
# `device:` decides what the electricity numbers cover. `omen-system` prices the
# whole machine, which is the only way an embedder running on the CPU is billed
# at all; `omen` samples the GPU board and would bill it nothing. `ingest:` reads
# the chunking wall clock off the document and chunk stamps while the cell is
# live, so `onetime_gpu_hours` is measured rather than typed. Both write over the
# matching `cost:` numbers below and print their provenance in the run."""

CONDITIONS_START = datetime(2026, 7, 1, tzinfo=timezone.utc)


#: Trailing comment on the commented-out window.
WINDOW_COMMENT = "the slice of the logs that is this cell"

#: Trailing comment on the live first sitting.
RUN_COMMENT = "edit to the window this cell was asked in"

#: Trailing comment on each commented-out repeat.
REPEAT_COMMENT = "a repeat is its own sitting; scores average per question"


def render_run_entries(
    opened: datetime,
    closing: datetime,
    *,
    count: int = 2,
    live: int = 1,
    run_ids: Optional[Sequence[str]] = None,
    run_id_template: str = "r{n}",
    emit_run_id: bool = True,
    stride: Optional[timedelta] = None,
    runs_key: str = RUNS_KEY,
    run_id_key: str = RUN_ID_KEY,
    start_key: str = "start",
    end_key: str = "end",
    ts_format: str = "%Y-%m-%dT%H:%M:%SZ",
    comment_col: int = 34,
    prefix: str = "",
    indent: str = "    ",
    comment_prefix: str = "# ",
    live_comment: Optional[str] = RUN_COMMENT,
    repeat_comment: Optional[str] = REPEAT_COMMENT,
    entry_comments: Optional[Sequence[Optional[str]]] = None,
) -> list:
    """One cell's ``runs:`` list, each sitting its own bounded entry.

    A repeat is its own entry because `parse_cell_runs` tells sittings apart by
    window, not by id: two entries carrying the same bounds and different ids
    both match the same answers and double-count them. So the entries emitted
    here subdivide the cell's own window and never reach a neighbouring cell.

    Args:
        opened / closing: the cell's whole slice; entries tile it in order.
        count: entries written, ids included.
        live: how many lead entries are written live; the rest are commented
            out, so an unedited file declares exactly one sitting per cell.
        run_ids: explicit ids, one per entry; None numbers them from
            ``run_id_template``.
        run_id_template: ``{n}`` is the 1-based entry number, matching the
            parser's own default so an omitted id reads the same.
        emit_run_id: False leaves the ids out, and the parser numbers them.
        stride: length of each entry's window, tiled forward from ``opened``.
            None splits ``opened``-``closing`` into ``count`` equal slices,
            which is the only placement that cannot claim another cell's logs.
        runs_key / run_id_key / start_key / end_key: yaml vocabulary, defaulted
            from `conditions.py` so the template cannot drift from the parser.
        ts_format: strftime pattern for the bounds.
        comment_col: column the ``#`` comments are padded to.
        prefix: commented out wholesale by the caller (e.g. ``# ``).
        indent: leading whitespace of the ``runs:`` key itself.
        comment_prefix: what a non-live entry is prepended with.
        live_comment / repeat_comment: trailing comment on the first live entry
            and on each commented one; None omits it.
        entry_comments: per-entry override, positional; short lists fall back to
            the two comments above.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    step = (closing - opened) / count if stride is None else stride
    ids = [str(r) for r in run_ids] if run_ids is not None else [
        run_id_template.format(n=i + 1) for i in range(count)
    ]
    if len(ids) != count:
        raise ValueError(f"run_ids names {len(ids)} entries for count={count}")

    out = [f"{prefix}{indent}{runs_key}:"]
    end_label = f"{end_key}:".ljust(len(start_key) + 1)
    for i, rid in enumerate(ids):
        pad = "" if i < live else comment_prefix
        head = f"{prefix}{indent}  {pad}- "
        body = f"{prefix}{indent}  {pad}  "
        if entry_comments is not None and i < len(entry_comments):
            note = entry_comments[i]
        else:
            note = live_comment if i < live else repeat_comment
        entry_open = opened + step * i
        start_line = f"{body if emit_run_id else head}{start_key}: " \
                     f"{entry_open.strftime(ts_format)}"
        if emit_run_id:
            out.append(_commented(f"{head}{run_id_key}: {rid}", note, comment_col))
            out.append(start_line)
        else:
            out.append(_commented(start_line, note, comment_col))
        out.append(f"{body}{end_label} {(entry_open + step).strftime(ts_format)}")
    return out


def _condition_lines(
    name: str,
    opened: datetime,
    closing: datetime,
    *,
    per_answer_cost: Optional[float],
    with_factors: bool,
    comment_col: int,
    ts_format: str,
    prefix: str = "",
    emit_window: bool = True,
    window_commented: bool = False,
    window_comment: Optional[str] = WINDOW_COMMENT,
    runs_block: Optional[Sequence[str]] = None,
) -> list:
    """One cell's yaml block. ``prefix`` (e.g. ``# ``) comments the whole block out.

    Args:
        runs_block: pre-rendered ``runs:`` lines from `render_run_entries`. Given
            one, it replaces the flat window, which the parser rejects beside a
            ``runs:`` list.
        emit_window: False drops ``start``/``end`` entirely.
        window_commented: emit the window commented out, for a cell pinned by
            ``conversation_ids`` alone. Ids and window live at once means both
            must match, which silently costs you every answer asked outside it.
    """
    note = label(name, with_factors=with_factors)
    key = f"{prefix}  {name}:".ljust(comment_col + len(prefix))
    out = [f"{key}# {note}" if note else key.rstrip()]
    if runs_block:
        out += list(runs_block)
    elif emit_window:
        wp = f"{prefix}    # " if window_commented else f"{prefix}    "
        out.append(_commented(f"{wp}start: {opened.strftime(ts_format)}",
                              window_comment if window_commented else None, comment_col))
        out.append(f"{wp}end:   {closing.strftime(ts_format)}")
    if per_answer_cost is not None:
        out.append(f"{prefix}    per_answer_cost: {per_answer_cost}")
    return out


def render_conditions_yaml(
    *,
    names: Sequence[str] = GRID_ORDER,
    reference: str = DEFAULT_REFERENCE,
    reference_key: str = REFERENCE_YAML_KEYS[0],
    start: datetime = CONDITIONS_START,
    window: timedelta = timedelta(days=1),
    per_answer_cost: Optional[float] = 0.0,
    with_factors: bool = True,
    comment_col: int = 34,
    ts_format: str = "%Y-%m-%dT%H:%M:%SZ",
    header: str = CONDITIONS_HEADER,
    footer: str = "",
) -> str:
    """
    The starter conditions.yaml, written from the cell table so the names here
    can never drift from `cells.py`.

    Args:
        names: cells to emit, in order; the first window starts at ``start``.
        reference: cell every delta is measured against.
        reference_key: yaml key that carries it (``reference``, or ``baseline``
            for a file an older eval still has to read).
        window: length of each cell's time window; windows run back to back.
        per_answer_cost: emitted per cell; None omits the line entirely.
        with_factors: append the two factor levels to each cell's comment.
        comment_col: column the `#` comments are padded to.
        ts_format: strftime pattern for the window bounds.
        header: comment block above the keys; footer: text appended verbatim.
    """
    lines = [header.rstrip("\n"), "", f"{reference_key}: {reference}", "", "conditions:"]
    for i, name in enumerate(names):
        opened = start + window * i
        lines += _condition_lines(
            name, opened, opened + window,
            per_answer_cost=per_answer_cost, with_factors=with_factors,
            comment_col=comment_col, ts_format=ts_format,
        )
    text = "\n".join(lines) + "\n"
    return text + footer if footer else text


def render_conditions_file(
    *,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    reference: str = DEFAULT_REFERENCE,
    reference_key: str = REFERENCE_YAML_KEYS[0],
    start: datetime = CONDITIONS_START,
    window: timedelta = timedelta(days=1),
    offgrid_start: Optional[datetime] = None,
    offgrid_window: Optional[timedelta] = None,
    offgrid_first: bool = True,
    offgrid_header: str = OFFGRID_HEADER,
    offgrid_commented: bool = False,
    offgrid_comment_prefix: str = "# ",
    per_answer_cost: Optional[float] = 0.0,
    offgrid_per_answer_cost: Optional[float] = None,
    with_factors: bool = True,
    comment_col: int = 34,
    ts_format: str = "%Y-%m-%dT%H:%M:%SZ",
    header: str = CONDITIONS_HEADER_WITH_OFFGRID,
    footer: str = "",
    blank_between: bool = True,
) -> str:
    """
    The starter conditions.yaml with both halves of the design: the scored 2x2
    grid and the off-grid ``prebaseline`` window that the before/after check
    needs. `render_conditions_yaml` emits the grid alone.

    Args:
        grid_names: scored cells, in order; the first window starts at ``start``.
        offgrid_names: cells measured but never scored in the 2x2.
        reference: cell every delta is measured against.
        reference_key: yaml key that carries it (``reference``, or ``baseline``
            for a file an older eval still has to read).
        window: length of each grid cell's window; grid windows run back to back.
        offgrid_start: first off-grid window; None puts it one ``offgrid_window``
            before ``start``, so the file reads in chronological order and the
            before-state cannot overlap a grid cell.
        offgrid_window: length of each off-grid window; None reuses ``window``.
        offgrid_first: emit the off-grid block above the grid.
        offgrid_header: comment block above the off-grid cells; "" omits it.
        offgrid_commented: emit the off-grid block commented out, for a run that
            genuinely has no before-state to measure.
        offgrid_comment_prefix: what commenting it out prepends.
        per_answer_cost: emitted per grid cell; None omits the line entirely.
        offgrid_per_answer_cost: same for off-grid cells; None reuses
            ``per_answer_cost``.
        with_factors: append the two factor levels to each cell's comment.
        comment_col: column the `#` comments are padded to.
        ts_format: strftime pattern for the window bounds.
        header: comment block above the keys; footer: text appended verbatim.
        blank_between: blank line between the off-grid and grid blocks.
    """
    offgrid_window = window if offgrid_window is None else offgrid_window
    if offgrid_start is None:
        offgrid_start = start - offgrid_window * max(1, len(offgrid_names))
    if offgrid_per_answer_cost is None:
        offgrid_per_answer_cost = per_answer_cost
    prefix = offgrid_comment_prefix if offgrid_commented else ""

    off_block: list = []
    if offgrid_names and offgrid_header:
        off_block.append(offgrid_header.rstrip("\n"))
    for i, name in enumerate(offgrid_names):
        opened = offgrid_start + offgrid_window * i
        off_block += _condition_lines(
            name, opened, opened + offgrid_window,
            per_answer_cost=offgrid_per_answer_cost, with_factors=with_factors,
            comment_col=comment_col, ts_format=ts_format, prefix=prefix,
        )

    grid_block: list = []
    for i, name in enumerate(grid_names):
        opened = start + window * i
        grid_block += _condition_lines(
            name, opened, opened + window,
            per_answer_cost=per_answer_cost, with_factors=with_factors,
            comment_col=comment_col, ts_format=ts_format,
        )

    blocks = [off_block, grid_block] if offgrid_first else [grid_block, off_block]
    blocks = [b for b in blocks if b]
    body: list = []
    for i, block in enumerate(blocks):
        if i and blank_between:
            body.append("")
        body += block

    lines = [header.rstrip("\n"), "", f"{reference_key}: {reference}", "", "conditions:"] + body
    text = "\n".join(lines) + "\n"
    return text + footer if footer else text


#: The machine every cell is priced on. `omen-system` covers the whole box, so
#: an embedder that runs on the CPU is billed; `omen` reads the GPU board alone
#: and would bill that embedder nothing. `omen-battery` measures the same box off
#: the battery's discharge rate, for a run made with the charger out.
EXAMPLE_DEVICE: str = "omen-system"

#: How the meter log is sliced. One serving window can be shorter than the 5 s
#: sampling cadence, so a busy span often holds a single row; `min_samples: 1`
#: prices that row instead of falling back to the nameplate.
EXAMPLE_POWER: Dict[str, Any] = {"min_samples": 1}

#: How the one-time chunking is priced, read off the database at dump time.
#: ``chunks`` spans the chunk writes, because a re-ingest rewrites chunk rows and
#: a build log row and never touches ``documents``, which leaves the ``lifecycle``
#: basis reading the upload for every cell. ``index_build_logs`` then carries the
#: real token count, the truncation count and the chunking policy label, and
#: ``prefer_logged`` bills the rag-service's own timed build.
EXAMPLE_INGEST: Dict[str, Any] = {
    "scope": "corpus",
    "basis": "chunks",
    "tokens_total": "index_build_logs",
    "hours_source": "prefer_logged",
}

EXAMPLE_COST_DEFAULTS: Dict[str, float] = {
    "watts": 43.7,
    "kwh_gbp": 0.26,
    "A": 1000.0,
    "Q": 1000.0,
}

EXAMPLE_CELL_COST: Dict[str, float] = {
    "onetime_gpu_hours": 0.0,
    "query_gpu_seconds": 2.1,
}

DECLARATION_COMMENTS: Dict[str, str] = {
    GRID_KEY: "the scored 2x2, in report order",
    OFFGRID_KEY: "measured, never scored inside the 2x2",
    SNAPSHOT_KEY: "dump each cell to json before the next ingest",
    DEVICE_KEY: "whole machine; `omen` is GPU-only and bills a CPU embedder nothing",
    POWER_KEY: "a busy window holding one 5 s sample is priced, not dropped to the nameplate",
    INGEST_KEY: "chunking hours, tokens and policy label, read while the cell is live",
    "watts": "fallback only: 45 W CPU base power + 11.3 W measured GPU idle",
    "kwh_gbp": "Ofgem unit rate; cite the retrieval date",
    "A": "answers the one-time cost amortises over",
    "Q": "queries in the period",
}

COST_FOOTER = """
# `cost:` keys are the arguments of veridic_eval.cfca_cost.cost_per_answer, one
# per quantity, plus `watts` / `kwh_gbp`, which convert to `p_gpu_hour` when no
# price is given directly. `cost_defaults:` is merged under every cell's own
# `cost:` block, so a cell only names what differs: the QDoRA fine-tune hours go
# in `onetime_gpu_hours` on the cells that were fine-tuned. A cell with no
# `cost:` block gets no CFCA line at all rather than a fabricated 0, and an
# unknown key raises instead of being silently dropped.
#
# Two of those keys are measured, not typed. `device:` names the machine once,
# `power:` slices its watt log and `ingest:` reads the ingest stamps off the
# database while the cell is live. `query_gpu_seconds` then carries the summed
# retrieval + rerank + generation latency, `onetime_gpu_hours` carries the
# chunking wall clock, and the number written here survives only for a cell that
# no measurement reached. `run` prints both provenance tables."""


def _flow_list(names: Sequence[str]) -> str:
    """``[a, b, c]``, the shape the grid / off-grid keys read best in."""
    return "[" + ", ".join(str(n) for n in names) + "]"


def _flow_map(values: Dict[str, Any]) -> str:
    """``{a: 1.0, b: 2.0}``, one line for a whole cost block."""
    return "{" + ", ".join(f"{k}: {v}" for k, v in values.items()) + "}"


def _yaml_bool(value: bool) -> str:
    return "true" if value else "false"


def _commented(text: str, note: Optional[str], comment_col: int) -> str:
    """``text`` with ``# note`` at ``comment_col``, or one space past a long line."""
    if not note:
        return text
    return text + " " * max(comment_col - len(text), 1) + f"# {note}"


def render_conditions_declaration(
    *,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    reference: str = DEFAULT_REFERENCE,
    reference_key: str = REFERENCE_YAML_KEYS[0],
    start: datetime = CONDITIONS_START,
    window: timedelta = timedelta(days=1),
    offgrid_start: Optional[datetime] = None,
    offgrid_window: Optional[timedelta] = None,
    offgrid_first: bool = True,
    per_answer_cost: Optional[float] = 0.0,
    offgrid_per_answer_cost: Optional[float] = None,
    declare_grid: bool = True,
    declare_offgrid: bool = True,
    emit_window: bool = True,
    window_commented: bool = False,
    runs_per_cell: Optional[int] = 2,
    runs_live: int = 1,
    runs_stride: Optional[timedelta] = None,
    run_id_template: str = "r{n}",
    emit_run_id: bool = True,
    run_comment: Optional[str] = RUN_COMMENT,
    repeat_comment: Optional[str] = REPEAT_COMMENT,
    snapshot: Optional[bool] = True,
    cell_snapshot: Optional[Dict[str, bool]] = None,
    device: Optional[str] = EXAMPLE_DEVICE,
    power: Optional[Dict[str, Any]] = EXAMPLE_POWER,
    ingest: Optional[Dict[str, Any]] = EXAMPLE_INGEST,
    cost_defaults: Optional[Dict[str, float]] = None,
    cell_cost: Optional[Dict[str, float]] = None,
    cost_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    cost_inline: bool = True,
    comments: Optional[Dict[str, str]] = None,
    header: str = DECLARATION_HEADER,
    offgrid_header: str = OFFGRID_HEADER,
    footer: str = "",
    with_factors: bool = True,
    comment_col: int = 34,
    ts_format: str = "%Y-%m-%dT%H:%M:%SZ",
    blank_between: bool = True,
    offgrid_commented: bool = False,
    offgrid_comment_prefix: str = "# ",
    conditions_key: str = CONDITIONS_KEY,
    grid_key: str = GRID_KEY,
    offgrid_key: str = OFFGRID_KEY,
    snapshot_key: str = SNAPSHOT_KEY,
    device_key: str = DEVICE_KEY,
    power_key: str = POWER_KEY,
    ingest_key: str = INGEST_KEY,
    cost_defaults_key: str = COST_DEFAULTS_KEY,
    cost_key: str = COST_KEY,
) -> str:
    """The starter conditions.yaml as the whole declaration a run reads.

    Everything `veridic-eval run` needs sits in this one file: which cells are
    scored in the 2x2, which are measured outside it, the log window per cell,
    whether each cell is snapshotted to json before the next ingest, and the GBP
    cost quantities behind the CFCA. `render_conditions_file` emits the older
    two-key shape (reference + conditions) for callers that still want it.

    Args:
        grid_names: cells scored in the 2x2, in report order; the first window
            starts at ``start``.
        offgrid_names: cells measured but never scored there (``prebaseline``).
        reference / reference_key: the cell every delta is measured against, and
            the yaml key carrying it (``reference`` or ``baseline``).
        start / window: first grid window and the length of each; grid windows
            run back to back.
        offgrid_start / offgrid_window: None puts the off-grid windows directly
            before ``start`` so the file reads in chronological order.
        offgrid_first: emit the off-grid block above the grid.
        per_answer_cost / offgrid_per_answer_cost: USD API bill per answer; None
            omits the line, and off-grid None reuses the grid value.
        declare_grid / declare_offgrid: write the ``grid:`` / ``offgrid:`` lists.
            Off leaves the split to the vocabulary in `cells.py`.
        runs_per_cell / runs_live: entries written under each cell's ``runs:``
            and how many of them are live, so an unedited file declares one
            sitting per cell. None or 0 writes the flat window instead.
        runs_stride / run_id_template / emit_run_id / run_comment /
            repeat_comment: placement, ids, and trailing comments of those
            entries, all owned by `render_run_entries`.
        emit_window / window_commented: the flat cell-level ``start``/``end``
            pair, read only while ``runs_per_cell`` is off because the parser
            rejects a cell-level window beside a ``runs:`` list. Commenting the
            window out needs another live marker in its place, and two live
            markers must BOTH match.
        snapshot: file-level snapshot default; None omits the key.
        cell_snapshot: per-cell overrides of that default.
        device: the machine every cell is priced on, a `devices.DEVICES` name or
            None to leave the file unmetered and price the typed numbers.
        power: ``power:`` block, the `measure_cell_serving_power` settings; None
            omits the key and takes that function's own defaults, which price
            retrieval + rerank + generation but demand two samples per window.
        ingest: ``ingest:`` block, the `measure_cell_ingest` settings that turn
            the ingest stamps into ``onetime_gpu_hours``; None omits the key.
        cost_defaults: file-level ``cost_defaults:`` block; None omits it.
        cell_cost: ``cost:`` block written under every cell; None omits it, which
            is the "no CFCA yet" file.
        cost_overrides: per-cell ``cost:`` block, replacing ``cell_cost``.
        cost_inline: one flow line per cost block instead of nested keys.
        comments: yaml key -> trailing comment, for the declaration lines and the
            ``cost_defaults`` keys.
        header / offgrid_header / footer: comment block above the keys, above the
            off-grid cells, and text appended verbatim.
        with_factors: append the two factor levels to each cell's comment.
        comment_col: column the ``#`` comments are padded to.
        ts_format: strftime pattern for the window bounds.
        blank_between: blank line between the off-grid and grid blocks.
        offgrid_commented / offgrid_comment_prefix: comment the off-grid cells
            out, for a run that genuinely has no before-state to measure.
        conditions_key / grid_key / offgrid_key / snapshot_key /
            cost_defaults_key / cost_key: the yaml key names, defaulted from
            `conditions.py` so the template cannot drift from the parser.
    """
    grid = [str(n) for n in grid_names]
    off = [str(n) for n in offgrid_names]
    notes = dict(comments or {})
    snaps = dict(cell_snapshot or {})
    costs = dict(cost_overrides or {})
    offgrid_window = window if offgrid_window is None else offgrid_window
    if offgrid_start is None:
        offgrid_start = start - offgrid_window * max(1, len(off))
    if offgrid_per_answer_cost is None:
        offgrid_per_answer_cost = per_answer_cost

    def cell_block(name: str, opened: datetime, closing: datetime, *,
                   pac: Optional[float], prefix: str) -> list:
        block = _condition_lines(
            name, opened, closing,
            per_answer_cost=pac, with_factors=with_factors,
            comment_col=comment_col, ts_format=ts_format, prefix=prefix,
            emit_window=emit_window,
            window_commented=window_commented,
            runs_block=None if not runs_per_cell else render_run_entries(
                opened, closing,
                count=runs_per_cell, live=runs_live, stride=runs_stride,
                run_id_template=run_id_template, emit_run_id=emit_run_id,
                ts_format=ts_format, comment_col=comment_col, prefix=prefix,
                live_comment=run_comment, repeat_comment=repeat_comment,
            ),
        )
        if name in snaps:
            block.append(f"{prefix}    {snapshot_key}: {_yaml_bool(snaps[name])}")
        cost = costs.get(name, cell_cost)
        if cost and cost_inline:
            block.append(_commented(
                f"{prefix}    {cost_key}: {_flow_map(cost)}",
                notes.get(cost_key), comment_col,
            ))
        elif cost:
            block.append(f"{prefix}    {cost_key}:")
            block += [_commented(f"{prefix}      {k}: {v}", notes.get(k), comment_col)
                      for k, v in cost.items()]
        return block

    declaration = [_commented(f"{reference_key}: {reference}", notes.get(reference_key), comment_col)]
    if declare_grid:
        declaration.append(_commented(f"{grid_key}: {_flow_list(grid)}",
                                      notes.get(grid_key), comment_col))
    if declare_offgrid and off:
        declaration.append(_commented(f"{offgrid_key}: {_flow_list(off)}",
                                      notes.get(offgrid_key), comment_col))
    if snapshot is not None:
        declaration.append(_commented(f"{snapshot_key}: {_yaml_bool(snapshot)}",
                                      notes.get(snapshot_key), comment_col))
    if device:
        declaration.append(_commented(f"{device_key}: {device}",
                                      notes.get(device_key), comment_col))
    for key, block in ((power_key, power), (ingest_key, ingest)):
        if block:
            declaration.append(_commented(f"{key}: {_flow_map(block)}",
                                          notes.get(key), comment_col))
    if cost_defaults:
        declaration.append(f"{cost_defaults_key}:")
        declaration += [_commented(f"  {k}: {v}", notes.get(k), comment_col)
                        for k, v in cost_defaults.items()]

    prefix = offgrid_comment_prefix if offgrid_commented else ""
    off_block: list = []
    if off and offgrid_header:
        off_block.append(offgrid_header.rstrip("\n"))
    for i, name in enumerate(off):
        off_block += cell_block(
            name,
            offgrid_start + offgrid_window * i,
            offgrid_start + offgrid_window * (i + 1),
            pac=offgrid_per_answer_cost,
            prefix=prefix,
        )
    grid_block: list = []
    for i, name in enumerate(grid):
        grid_block += cell_block(
            name, start + window * i, start + window * (i + 1),
            pac=per_answer_cost, prefix="",
        )
    blocks = [off_block, grid_block] if offgrid_first else [grid_block, off_block]
    body: list = []
    for i, block in enumerate([b for b in blocks if b]):
        if i and blank_between:
            body.append("")
        body += block

    lines = [header.rstrip("\n"), ""] + declaration + ["", f"{conditions_key}:"] + body
    text = "\n".join(lines) + "\n"
    return text + footer if footer else text


CONDITIONS_TEMPLATE = render_conditions_declaration(
    cost_defaults=EXAMPLE_COST_DEFAULTS,
    cell_cost=EXAMPLE_CELL_COST,
    comments=DECLARATION_COMMENTS,
    footer=COST_FOOTER + "\n",
)

EMBEDDED_TEMPLATES: Dict[str, str] = {
    "benchmark.example.yaml": BENCHMARK_TEMPLATE,
    "conditions.example.yaml": CONDITIONS_TEMPLATE,
}

DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("benchmark.example.yaml", "benchmark.yaml"),
    ("conditions.example.yaml", "conditions.yaml"),
)


def default_search_dirs() -> Tuple[Path, ...]:
    """Working directory first, then the repo root beside the package, then the package."""
    pkg = Path(__file__).resolve().parent
    return (Path.cwd(), pkg.parent, pkg)


def find_example(name: str, search_dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """First existing `name` across `search_dirs`, or None."""
    for directory in (default_search_dirs() if search_dirs is None else search_dirs):
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return None


def write_yaml_templates(
    pairs: Sequence[Tuple[str, str]] = DEFAULT_PAIRS,
    dest_dir: str = ".",
    search_dirs: Optional[Sequence[Path]] = None,
    embedded: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
    allow_embedded: bool = True,
    encoding: str = "utf-8",
    printer: Optional[Callable[[str], None]] = print,
) -> Dict[str, str]:
    """
    Write each (example_name, target_name) pair into `dest_dir`.

    pairs           : (source example filename, file to create) tuples.
    dest_dir        : where the targets are written (created if absent).
    search_dirs     : where example files are looked for; None = default_search_dirs().
    embedded        : name -> text fallback map; None = EMBEDDED_TEMPLATES.
    overwrite       : True rewrites an existing target instead of leaving it.
    allow_embedded  : False makes a missing example file an error line, not a fallback.
    encoding        : read/write encoding for both example and target.
    printer         : per-target status sink; None silences output.

    Returns target -> outcome, one of "exists", "written:<origin>", "missing".
    """
    table = EMBEDDED_TEMPLATES if embedded is None else embedded
    dest = Path(dest_dir)
    results: Dict[str, str] = {}

    for example, target in pairs:
        out_path = dest / target
        if out_path.exists() and not overwrite:
            results[target] = "exists"
            if printer:
                printer(f"{target}: exists, left untouched")
            continue

        source = find_example(example, search_dirs)
        if source is not None:
            text, origin = source.read_text(encoding=encoding), str(source)
        elif allow_embedded and example in table:
            text, origin = table[example], "built-in template"
        else:
            results[target] = "missing"
            if printer:
                printer(f"{target}: not created ({example} not found, no built-in template)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding=encoding)
        results[target] = f"written:{origin}"
        if printer:
            printer(f"{target}: created from {origin} - fill it in")

    return results
