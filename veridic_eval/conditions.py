"""
conditions.yaml -> every input a run needs, parsed once, in one place.

`veridic-eval init` writes the file and `veridic-eval run` reads it, so the two
commands cover every metric. The file carries, per cell: the log window (or the
conversation ids), any USD API bill per answer, the GBP quantities for the CFCA,
and whether the cell is snapshotted to JSON
before the next ingest deletes its served-evidence rows. It also carries which
cells sit outside the scored 2x2, so a `prebaseline` before-window is declared
and measured here instead of through a separate tool.

File shape (YAML or JSON; only ``conditions`` is required)::

    reference: control            # or `baseline:`, for a file an older eval reads
    grid: [control, chunk_opt, qdora, combined]
    offgrid: [prebaseline]        # measured, never scored inside the 2x2
    snapshot: true                # default for every cell below
    cost_defaults:                # merged under each cell's `cost:` block
      watts: 43.7                 # fallback only: the meter overwrites it per cell
      kwh_gbp: 0.26               # Ofgem unit rate, cite the retrieval date
      A: 1000.0
      Q: 1000.0
    conditions:
      prebaseline:
        conversation_ids: ["<uuid>"]
        per_answer_cost: 0.0      # USD API bill per answer, converted to GBP
        snapshot: true
        cost: {onetime_gpu_hours: 0.0, query_gpu_seconds: 2.1}
      control:
        start: 2026-07-01T00:00:00Z
        end:   2026-07-02T00:00:00Z
        per_answer_cost: 0.0

The cost keys are the arguments of ``cfca_cost.cost_per_answer`` one for one,
plus ``watts`` / ``kwh_gbp``, which ``cfca_cost.electricity_gpu_hour`` turns
into ``p_gpu_hour``. An unknown key raises instead of being dropped, so a typo
cannot quietly zero a cost term. A cell with no ``cost:`` block and no
``cost_defaults`` gets no CFCA line at all rather than a fabricated 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .cells import (
    DEFAULT_REFERENCE,
    GRID_ORDER,
    LEGACY_ALIASES,
    OFFGRID_ORDER,
    REFERENCE_YAML_KEYS,
    legacy_notes,
    reference_from_mapping,
    resolve_reference,
    split_grid,
)
from .cfca_cost import cfca, cost_per_answer, electricity_gpu_hour
from .config import DEFAULT_RUN_ID, Condition

# ---------------------------------------------------------------- file keys
CONDITIONS_KEY: str = "conditions"
COST_KEY: str = "cost"
COST_DEFAULTS_KEY: str = "cost_defaults"
GRID_KEY: str = "grid"
OFFGRID_KEY: str = "offgrid"
SNAPSHOT_KEY: str = "snapshot"
PER_ANSWER_COST_KEY: str = "per_answer_cost"
RUNS_KEY: str = "runs"
RUN_ID_KEY: str = "run_id"
#: The machine that served the run, named once for the whole file. Its profile
#: supplies the sampler, so `watts` is measured off the power log per cell
#: instead of typed under every ``cost:`` block.
DEVICE_KEY: str = "device"
#: Where that log lives, and how a cell's slice of it is taken.
POWER_KEY: str = "power"
#: How the one-time ingest is read off the database and priced. The chunking
#: correction lands in ``onetime_gpu_hours``, so this block is what makes a
#: chunking cell cost more than the baseline it is compared against.
INGEST_KEY: str = "ingest"
#: The chunking policy label a cell says its ingest ran under, e.g.
#: ``context_rag_t256_o64`` for the token builder. Nothing here changes a
#: number: the label is the bucket key every cost query groups by, so a cell
#: that ingested without its ``CHUNKING_MODE`` flip lands in the other cell's
#: bucket, and this is the declaration that lets the run notice.
EXPECT_POLICY_KEY: str = "expect_policy"

#: ``cost_per_answer`` arguments, the only names a ``cost:`` block may use
#: besides the two derived ones below.
COST_ARG_NAMES: Tuple[str, ...] = (
    "p_gpu_hour", "p_embed_mtok", "p_in_mtok", "p_out_mtok", "p_store_gb_month",
    "onetime_gpu_hours", "onetime_embed_mtok", "A",
    "query_gpu_seconds", "query_embed_mtok", "query_in_mtok", "query_out_mtok",
    "U", "update_gpu_hours", "update_embed_mtok", "M", "index_gb", "Q",
)
#: Measured inputs that become ``p_gpu_hour``.
COST_DERIVED_NAMES: Tuple[str, ...] = ("watts", "kwh_gbp")
#: Denominators ``cost_per_answer`` divides by.
COST_POSITIVE_NAMES: Tuple[str, ...] = ("A", "Q")

#: Every key a cell may carry. Each one is read: an accepted key that changed
#: nothing would be the silent no-op this parser exists to prevent.
CELL_KEYS: Tuple[str, ...] = (
    "start", "end", "conversation_ids", "query_ids", "note",
    PER_ANSWER_COST_KEY, COST_KEY, SNAPSHOT_KEY, RUNS_KEY, EXPECT_POLICY_KEY,
)
#: Every key one entry under ``runs:`` may carry. A repeat is its own sitting,
#: so it names its own window and, if the chats were pinned, its own ids.
RUN_KEYS: Tuple[str, ...] = (
    RUN_ID_KEY, "start", "end", "conversation_ids", "note",
)
#: Cell-level keys that name one sitting. Under ``runs:`` each entry names its
#: own, so a cell carrying both would leave the repeats overlapping in silence.
WINDOW_KEYS: Tuple[str, ...] = ("start", "end", "conversation_ids")
TOP_LEVEL_KEYS: Tuple[str, ...] = (
    CONDITIONS_KEY, COST_DEFAULTS_KEY, GRID_KEY, OFFGRID_KEY, SNAPSHOT_KEY,
    DEVICE_KEY, POWER_KEY, INGEST_KEY,
) + REFERENCE_YAML_KEYS

#: Every key the ``power:`` block may carry, each one an argument of
#: `power_meter.measure_cell_serving_power`, so the yaml cannot ask for a knob
#: the measurement does not have. ``default_latency_ms`` is absent on purpose:
#: the serving builder takes it inside ``intervals_kwargs``, and accepting it
#: here would read as a setting that never reached the intervals. ``integrator``
#: is absent for the same reason a callable cannot come from yaml: the six
#: bracket knobs below steer the default one instead.
POWER_KEYS: Tuple[str, ...] = (
    "log_path", "missing_log", "tag", "priced", "field_sets", "latency_fields",
    "intervals_kwargs",
    "constant_mode", "constant_w", "constant_interval_s", "constant_upper_bound",
    "keep_alternates", "round_seconds",
    "basis", "subtract_idle", "idle_w", "max_gap_s", "min_samples", "min_coverage",
    "edge", "min_inside", "bracket_before", "bracket_after", "max_bracket_gap_s",
    "require_both_edges", "stats_from",
    "pad_start_s", "pad_end_s", "merge_overlaps", "fallback_tdp",
    "on_uncovered",
)

#: Every key the ``ingest:`` block may carry, each one an argument of
#: `ingest_cost.measure_cell_ingest`. ``read_kwargs`` / ``span_kwargs`` /
#: ``token_kwargs`` are the three nested mappings, for the database read, the
#: span arithmetic, and the logged token count `tokens_total` names.
INGEST_KEYS: Tuple[str, ...] = (
    "scope", "filenames", "basis", "merge_overlaps", "include_embed_mtok",
    "chars_per_token", "tokens_total", "embed_passes", "hours_upper_bound",
    "hours_source", "hours_kwargs",
    "keep_per_document", "as_of_from_records",
    "read_kwargs", "span_kwargs", "token_kwargs",
)

NONE_WORDS: Tuple[str, ...] = ("", "none", "null")


# ------------------------------------------------------------- timestamps
def parse_timestamp(
    value: Any,
    *,
    none_words: Sequence[str] = NONE_WORDS,
    zulu: str = "Z",
    zulu_offset: str = "+00:00",
    assume_tz: Optional[timezone] = None,
) -> Optional[datetime]:
    """One ISO-8601 window bound out of YAML, or None when the file omits it.

    Args:
        none_words: strings read as "no bound", compared casefolded.
        zulu / zulu_offset: the trailing ``Z`` is rewritten before parsing,
            because ``datetime.fromisoformat`` on Python 3.11 accepts the
            offset form only.
        assume_tz: attached to a bound written without an offset; None leaves
            it naive, which is what the DB comparison already assumed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if s.casefold() in tuple(w.casefold() for w in none_words):
            return None
        dt = datetime.fromisoformat(s.replace(zulu, zulu_offset))
    if assume_tz is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_tz)
    return dt


# ------------------------------------------------------------------- runs
def parse_cell_runs(
    spec: Dict[str, Any],
    name: str,
    *,
    runs_key: str = RUNS_KEY,
    run_id_key: str = RUN_ID_KEY,
    run_keys: Sequence[str] = RUN_KEYS,
    window_keys: Sequence[str] = WINDOW_KEYS,
    ids_key: str = "conversation_ids",
    start_key: str = "start",
    end_key: str = "end",
    note_key: str = "note",
    default_run_id: str = DEFAULT_RUN_ID,
    run_id_template: str = "r{n}",
    inherit_ids: bool = False,
    inherit_note: bool = False,
    allow_flat_with_runs: bool = False,
    require_distinct: bool = True,
    require_bounded_runs: bool = True,
    strict_run_keys: bool = True,
    assume_tz: Optional[timezone] = None,
    where: str = "conditions",
) -> Tuple[Condition, ...]:
    """
    One cell's sittings as `Condition` slices, in the order asked.

    No ``runs:`` gives one condition from the flat keys, which is what every
    existing file means. Each run entry carries its own window and ``run_id``.

    Args:
        spec: the cell's mapping, already checked for unknown cell keys.
        runs_key / run_id_key: yaml keys for the list and for an explicit id.
        run_keys / window_keys: keys accepted in a run entry, and the cell-level
            window keys a ``runs:`` list replaces.
        ids_key / start_key / end_key / note_key: window vocabulary.
        run_id_template: id for an entry that names none, ``{n}`` the 1-based
            position. Write ``run_id:`` to pin ids against reordering.
        inherit_ids: repeats reuse the cell's pinned uuids; off by default, or
            every sitting matches the same chats.
        allow_flat_with_runs: keep a cell-level window beside ``runs:``. It is
            then ignored, not merged.
        require_distinct / require_bounded_runs / strict_run_keys: reject two
            entries with one id, an entry with no window, an unknown key.
        assume_tz: timezone for bounds written without an offset.
        where: label used in error messages.
    """
    raw_runs = spec.get(runs_key)

    def _condition(entry: Dict[str, Any], run_id: str) -> Condition:
        return Condition(
            name=name,
            start=parse_timestamp(entry.get(start_key), assume_tz=assume_tz),
            end=parse_timestamp(entry.get(end_key), assume_tz=assume_tz),
            conversation_ids=[str(c) for c in (entry.get(ids_key) or [])],
            run_id=run_id,
            note=None if entry.get(note_key) is None else str(entry[note_key]),
        )

    if raw_runs is None:
        return (_condition(spec, default_run_id),)

    if not isinstance(raw_runs, list):
        raise ValueError(
            f"{where}: cell {name!r}: `{runs_key}:` must be a list of sittings, "
            f"got {type(raw_runs).__name__}"
        )
    if not raw_runs:
        raise ValueError(f"{where}: cell {name!r}: `{runs_key}:` is empty")

    flat = [k for k in window_keys if spec.get(k) not in (None, [], "")]
    if flat and not allow_flat_with_runs:
        raise ValueError(
            f"{where}: cell {name!r} declares both `{runs_key}:` and cell-level "
            f"{sorted(flat)}; move that window into the list as one entry"
        )

    out: List[Condition] = []
    seen: Dict[str, int] = {}
    for i, raw in enumerate(raw_runs):
        entry = dict(raw or {})
        if not isinstance(raw or {}, dict):
            raise ValueError(
                f"{where}: cell {name!r} run {i + 1} must be a mapping, "
                f"got {type(raw).__name__}"
            )
        stray = [k for k in entry if k not in tuple(run_keys)]
        if stray and strict_run_keys:
            raise ValueError(
                f"{where}: cell {name!r} run {i + 1} has unknown key(s) {sorted(stray)}; "
                f"accepted: {sorted(run_keys)}"
            )
        rid = entry.get(run_id_key)
        rid = run_id_template.format(n=i + 1) if rid is None else str(rid)
        if require_distinct and rid in seen:
            raise ValueError(
                f"{where}: cell {name!r}: run id {rid!r} is claimed by entries "
                f"{seen[rid]} and {i + 1}"
            )
        seen[rid] = i + 1
        if inherit_ids and not entry.get(ids_key):
            entry[ids_key] = spec.get(ids_key)
        if inherit_note and entry.get(note_key) is None:
            entry[note_key] = spec.get(note_key)
        cond = _condition(entry, rid)
        if require_bounded_runs and not (cond.start or cond.end or cond.conversation_ids):
            raise ValueError(
                f"{where}: cell {name!r} run {rid!r} names no start/end and no "
                f"{ids_key}, so it cannot be told from the cell's other sittings"
            )
        out.append(cond)
    return tuple(out)


# ------------------------------------------------------------- cost blocks
def parse_cost_block(
    raw: Optional[Dict[str, Any]],
    *,
    defaults: Optional[Dict[str, Any]] = None,
    where: str = COST_KEY,
    arg_names: Sequence[str] = COST_ARG_NAMES,
    derived_names: Sequence[str] = COST_DERIVED_NAMES,
    positive_names: Sequence[str] = COST_POSITIVE_NAMES,
    watts_key: str = "watts",
    kwh_key: str = "kwh_gbp",
    gpu_hour_key: str = "p_gpu_hour",
    derive_gpu_hour: bool = True,
    declared_gpu_hour_wins: bool = True,
    strict: bool = True,
    require: Sequence[str] = (),
    empty_is_none: bool = True,
) -> Optional[Dict[str, float]]:
    """Merge ``defaults`` with one cell's ``cost:`` block into kwargs for
    ``cfca_cost.cost_per_answer``.

    Returns None when nothing is declared, so a cell without cost inputs gets
    no CFCA number rather than a fabricated zero.

    Args:
        defaults: file-level ``cost_defaults``; the cell's own keys win.
        where: name used in error messages (e.g. ``cost[control]``).
        arg_names: accepted keys that pass straight through.
        derived_names: accepted keys that are converted, not passed through.
        positive_names: keys that must be > 0 because they are denominators.
        watts_key / kwh_key / gpu_hour_key: the conversion triple.
        derive_gpu_hour: turn watts + kwh_gbp into ``p_gpu_hour``. Off keeps
            both measured numbers out of the arithmetic entirely.
        declared_gpu_hour_wins: an explicit ``p_gpu_hour`` beats the derived
            one (matching `veridic-eval cost --p-gpu-hour`); False lets the
            measured pair override it.
        strict: an unrecognised key raises. False drops it silently, which is
            only for reading a file written by a newer version.
        require: keys that must be present after the merge.
        empty_is_none: an empty merged block returns None instead of {}.
    """
    merged: Dict[str, Any] = {}
    for source in (defaults or {}, raw or {}):
        if source is None:
            continue
        if not isinstance(source, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(source).__name__}")
        merged.update(source)
    if empty_is_none and not merged:
        return None

    known = set(arg_names) | set(derived_names)
    unknown = [k for k in merged if k not in known]
    if unknown and strict:
        raise ValueError(
            f"{where}: unknown cost key(s) {sorted(unknown)}; "
            f"accepted: {sorted(known)}"
        )

    numbers: Dict[str, float] = {}
    for key, value in merged.items():
        if key not in known:
            continue
        try:
            numbers[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {key} is not a number: {value!r}") from exc

    watts = numbers.pop(watts_key, None)
    kwh = numbers.pop(kwh_key, None)
    if derive_gpu_hour and watts is not None and kwh is not None:
        derived = electricity_gpu_hour(watts, kwh)
        if gpu_hour_key not in numbers or not declared_gpu_hour_wins:
            numbers[gpu_hour_key] = derived
    elif derive_gpu_hour and (watts is None) != (kwh is None) and gpu_hour_key not in numbers:
        raise ValueError(
            f"{where}: {watts_key} and {kwh_key} come as a pair, or give "
            f"{gpu_hour_key} directly"
        )

    for key in positive_names:
        if key in numbers and numbers[key] <= 0.0:
            raise ValueError(f"{where}: {key} is a denominator and must be > 0, got {numbers[key]}")
    missing = [k for k in require if k not in numbers]
    if missing:
        raise ValueError(f"{where}: missing required cost key(s) {sorted(missing)}")
    negative = sorted(k for k, v in numbers.items() if v < 0.0)
    if negative:
        raise ValueError(f"{where}: negative values not allowed: {negative}")
    return numbers or (None if empty_is_none else {})


# ------------------------------------------------------------ parsed file
@dataclass
class DeclaredCell:
    """One entry under ``conditions:``, with everything that entry declares."""

    name: str
    condition: Condition
    per_answer_cost: Optional[float] = None
    query_ids: Optional[List[str]] = None
    cost: Optional[Dict[str, float]] = None
    snapshot: bool = True
    note: Optional[str] = None
    #: The chunking policy label this cell's ingest is supposed to have run
    #: under; None asks for no check.
    expect_policy: Optional[str] = None
    #: Every sitting of this cell, in the order asked. A cell asked once holds
    #: one entry, and ``condition`` is always that first sitting.
    runs: Tuple[Condition, ...] = ()

    def __post_init__(self) -> None:
        if not self.runs:
            self.runs = (self.condition,)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def run_ids(self) -> List[str]:
        return [c.run_id for c in self.runs]

    @property
    def bounded(self) -> bool:
        """True when the cell names explicit conversations, or a window."""
        c = self.condition
        return bool(c.conversation_ids or c.start or c.end)


@dataclass
class ConditionsFile:
    """A parsed conditions file: cells in file order, plus the resolved split."""

    cells: List[DeclaredCell]
    reference: str
    grid: Tuple[str, ...]
    offgrid: Tuple[str, ...]
    path: Optional[str] = None
    reference_requested: Optional[str] = None
    cost_defaults: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: ``device:`` verbatim, a registry name or an inline profile mapping.
    #: `devices.resolve_device` turns it into a profile at measurement time, so
    #: a file can be parsed on a machine that is not the one that served it.
    device: Optional[Any] = None
    #: ``power:`` verbatim, the `measure_cell_serving_power` overrides for this
    #: file.
    power: Dict[str, Any] = field(default_factory=dict)
    #: ``ingest:`` verbatim, the `measure_cell_ingest` overrides for this file.
    ingest: Dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cells]

    @property
    def conditions(self) -> List[Condition]:
        """One condition per cell: the first sitting, which is the whole of a
        cell asked once. Callers that score repeats want `cell_runs`."""
        return [c.condition for c in self.cells]

    @property
    def cell_runs(self) -> Dict[str, Tuple[Condition, ...]]:
        """Every sitting of every cell, keyed by cell name, in file order."""
        return {c.name: c.runs for c in self.cells}

    @property
    def run_counts(self) -> Dict[str, int]:
        return {c.name: c.n_runs for c in self.cells}

    @property
    def per_answer_cost(self) -> Dict[str, float]:
        return {c.name: c.per_answer_cost for c in self.cells if c.per_answer_cost is not None}

    @property
    def cost_inputs(self) -> Dict[str, Dict[str, float]]:
        return {c.name: c.cost for c in self.cells if c.cost}

    @property
    def query_ids(self) -> Dict[str, List[str]]:
        return {c.name: c.query_ids for c in self.cells if c.query_ids}

    @property
    def expect_policy(self) -> Dict[str, str]:
        """``{cell: label}`` for the cells that declared one, for the label check."""
        return {c.name: c.expect_policy for c in self.cells if c.expect_policy}

    def cell(self, name: str) -> DeclaredCell:
        for c in self.cells:
            if c.name == name:
                return c
        raise KeyError(f"no cell {name!r} in {self.path or 'conditions'}")

    def snapshot_names(self, *, only: Optional[Sequence[str]] = None,
                       skip: Sequence[str] = ()) -> List[str]:
        """Cells whose snapshot is switched on, minus ``skip``, kept in file order."""
        wanted = None if only is None else {str(n) for n in only}
        return [
            c.name for c in self.cells
            if c.snapshot and c.name not in set(skip)
            and (wanted is None or c.name in wanted)
        ]


def parse_measurement_block(
    data: Dict[str, Any],
    *,
    key: str,
    accepted: Sequence[str],
    strict: bool = True,
    path: Optional[str] = None,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One file-level measurement block (``power:``, ``ingest:``), key-checked.

    Every accepted name is an argument of the function that runs the
    measurement, so a knob the measurement does not have raises here instead of
    sitting in the yaml looking applied.

    Args:
        key: the yaml key to read; absent or empty gives ``default``.
        accepted: argument names the block may carry.
        strict: False keeps unknown keys, for a caller that forwards them on.
        default: value for an absent block; None means ``{}``.

    Returns:
        A new dict, the block verbatim.
    """
    block = data.get(key)
    if block is None or block == {}:
        return dict(default or {})
    if not isinstance(block, dict):
        raise ValueError(
            f"{path or 'conditions'}: `{key}:` must be a mapping, got {type(block).__name__}"
        )
    stray = [k for k in block if k not in tuple(accepted)]
    if stray and strict:
        raise ValueError(
            f"{path or 'conditions'}: `{key}:` has unknown key(s) {sorted(stray)}; "
            f"accepted: {sorted(accepted)}"
        )
    return dict(block)


def parse_conditions_data(
    data: Dict[str, Any],
    *,
    path: Optional[str] = None,
    conditions_key: str = CONDITIONS_KEY,
    reference_keys: Sequence[str] = REFERENCE_YAML_KEYS,
    reference_default: Optional[str] = DEFAULT_REFERENCE,
    reference_fallback: str = "first",
    grid_key: str = GRID_KEY,
    offgrid_key: str = OFFGRID_KEY,
    grid_vocabulary: Sequence[str] = GRID_ORDER,
    offgrid_vocabulary: Sequence[str] = OFFGRID_ORDER,
    unknown: str = "grid",
    aliases: Dict[str, str] = LEGACY_ALIASES,
    snapshot_key: str = SNAPSHOT_KEY,
    snapshot_default: bool = True,
    cost_key: str = COST_KEY,
    cost_defaults_key: str = COST_DEFAULTS_KEY,
    per_answer_cost_key: str = PER_ANSWER_COST_KEY,
    runs_key: str = RUNS_KEY,
    device_key: str = DEVICE_KEY,
    power_key: str = POWER_KEY,
    power_keys: Sequence[str] = POWER_KEYS,
    strict_power_keys: bool = True,
    ingest_key: str = INGEST_KEY,
    ingest_keys: Sequence[str] = INGEST_KEYS,
    strict_ingest_keys: bool = True,
    expect_policy_key: str = EXPECT_POLICY_KEY,
    cell_keys: Sequence[str] = CELL_KEYS,
    top_level_keys: Sequence[str] = TOP_LEVEL_KEYS,
    strict_cell_keys: bool = True,
    strict_top_level: bool = False,
    strict_cost: bool = True,
    require_cost: Sequence[str] = (),
    on_unbounded: str = "warn",
    assume_tz: Optional[timezone] = None,
    printer: Optional[Callable[[str], None]] = print,
) -> ConditionsFile:
    """Turn the loaded mapping into a `ConditionsFile`. No I/O.

    The reference cell is resolved against the grid alone, so a file that lists
    ``prebaseline`` first and forgets `reference:` cannot make an off-grid
    before-window the thing every delta is measured against.

    Args:
        reference_fallback: `cells.resolve_reference` policy when the file
            names none: ``first`` grid cell, ``default``, ``none`` or ``raise``.
        grid_vocabulary / offgrid_vocabulary: names used when the file declares
            no ``grid:`` / ``offgrid:`` list of its own.
        unknown: where a name in neither list goes ("grid", "offgrid", "drop",
            "raise"), passed to `cells.split_grid`.
        snapshot_default: file-level ``snapshot:`` overrides it, a cell's own
            ``snapshot:`` overrides that.
        runs_key: yaml key holding a cell's repeats, one entry per sitting;
            `parse_cell_runs` owns every knob inside it.
        device_key / power_key: the machine that served the run and the
            overrides for slicing its power log. Kept verbatim so the file
            parses on a machine that has no sampler installed.
        power_keys / strict_power_keys: accepted keys inside ``power:``; a key
            `measure_cell_serving_power` does not take raises rather than being
            read as a measurement setting that never applied.
        ingest_key / ingest_keys / strict_ingest_keys: the same for ``ingest:``,
            whose keys are `ingest_cost.measure_cell_ingest` arguments. Kept
            verbatim so the file parses with no database in reach.
        expect_policy_key: cell key naming the chunking policy label that cell's
            ingest should have run under, read by
            `ingest_cost.check_ingest_policies`. It prices nothing and edits
            nothing; a cell that omits it is not judged.
        strict_cell_keys / strict_top_level: raise on a key we do not read, so
            a misspelled ``conversaton_ids`` fails loudly instead of widening
            the window to the whole log.
        strict_cost / require_cost: passed to `parse_cost_block`.
        on_unbounded: a cell with no window and no conversation ids matches
            every logged answer: ``warn`` prints, ``raise`` refuses, ``ignore``
            says nothing.
        assume_tz: timezone attached to bounds written without an offset.
        printer: where notes go; None silences them.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'conditions'}: top level must be a mapping")
    if on_unbounded not in ("warn", "raise", "ignore"):
        raise ValueError("on_unbounded must be 'warn', 'raise' or 'ignore'")

    unknown_top = [k for k in data if k not in tuple(top_level_keys)]
    if unknown_top and strict_top_level:
        raise ValueError(
            f"{path or 'conditions'}: unknown top-level key(s) {sorted(unknown_top)}; "
            f"accepted: {sorted(top_level_keys)}"
        )

    raw_cells = data.get(conditions_key) or {}
    if not isinstance(raw_cells, dict):
        raise ValueError(f"{path or 'conditions'}: `{conditions_key}:` must be a mapping of cell name -> spec")
    if not raw_cells:
        raise ValueError(f"No conditions found in {path or 'conditions'}")

    file_snapshot = data.get(snapshot_key)
    snapshot_all = snapshot_default if file_snapshot is None else bool(file_snapshot)
    cost_defaults = data.get(cost_defaults_key) or {}

    device = data.get(device_key)
    power = parse_measurement_block(
        data, key=power_key, accepted=power_keys, strict=strict_power_keys, path=path,
    )
    ingest = parse_measurement_block(
        data, key=ingest_key, accepted=ingest_keys, strict=strict_ingest_keys, path=path,
    )

    notes: List[str] = []
    cells: List[DeclaredCell] = []
    for name, spec in raw_cells.items():
        name = str(name)
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"{path or 'conditions'}: cell {name!r} must be a mapping, got {type(spec).__name__}")
        stray = [k for k in spec if k not in tuple(cell_keys)]
        if stray and strict_cell_keys:
            raise ValueError(
                f"{path or 'conditions'}: cell {name!r} has unknown key(s) {sorted(stray)}; "
                f"accepted: {sorted(cell_keys)}"
            )
        runs = parse_cell_runs(
            spec, name,
            runs_key=runs_key,
            assume_tz=assume_tz,
            where=str(path or "conditions"),
        )
        condition = runs[0]
        pac = spec.get(per_answer_cost_key)
        qids = spec.get("query_ids")
        snap = spec.get(snapshot_key)
        cell = DeclaredCell(
            name=name,
            condition=condition,
            per_answer_cost=None if pac is None else float(pac),
            query_ids=[str(q) for q in qids] if qids else None,
            cost=parse_cost_block(
                spec.get(cost_key),
                defaults=cost_defaults,
                where=f"{cost_key}[{name}]",
                strict=strict_cost,
                require=require_cost,
            ),
            snapshot=snapshot_all if snap is None else bool(snap),
            note=None if spec.get("note") is None else str(spec["note"]),
            expect_policy=(
                None if spec.get(expect_policy_key) is None
                else str(spec[expect_policy_key]).strip() or None
            ),
            runs=runs,
        )
        if not cell.bounded:
            msg = (f"cell {name!r} declares no conversation_ids and no "
                   "start/end, so it matches every logged answer")
            if on_unbounded == "raise":
                raise ValueError(f"{path or 'conditions'}: {msg}")
            if on_unbounded == "warn":
                notes.append(msg)
        cells.append(cell)

    names = [c.name for c in cells]
    notes.extend(legacy_notes(names, aliases=aliases))

    declared_grid = data.get(grid_key)
    declared_offgrid = data.get(offgrid_key)
    grid_vocab = tuple(str(n) for n in declared_grid) if declared_grid else tuple(grid_vocabulary)
    offgrid_vocab = tuple(str(n) for n in declared_offgrid) if declared_offgrid else tuple(offgrid_vocabulary)
    overlap = sorted(set(grid_vocab) & set(offgrid_vocab))
    if overlap:
        raise ValueError(f"{path or 'conditions'}: {overlap} listed as both `{grid_key}:` and `{offgrid_key}:`")

    grid, offgrid = split_grid(
        names,
        grid_names=grid_vocab,
        offgrid_names=offgrid_vocab,
        aliases=aliases,
        unknown=unknown,
        order="given",
    )
    requested = reference_from_mapping(data, keys=reference_keys)
    if requested and requested in offgrid:
        raise ValueError(
            f"{path or 'conditions'}: reference {requested!r} is an off-grid cell; the 2x2 "
            f"reference must be one of {grid}. List it under `{grid_key}:` to score it inside "
            "the grid, or point `reference:` at a grid cell."
        )
    if not grid:
        raise ValueError(
            f"{path or 'conditions'}: every declared cell is off-grid {offgrid}; nothing to score"
        )
    reference = resolve_reference(
        grid, requested=requested, default=reference_default,
        fallback=reference_fallback, aliases=aliases, required=True,
    )

    if printer:
        for note in notes:
            printer(f"  note: {note}")
    return ConditionsFile(
        cells=cells,
        reference=str(reference),
        grid=tuple(grid),
        offgrid=tuple(offgrid),
        path=path,
        reference_requested=requested,
        cost_defaults=dict(cost_defaults),
        notes=notes,
        device=device,
        power=dict(power),
        ingest=dict(ingest),
    )


def load_conditions_file(
    path: str,
    *,
    encoding: str = "utf-8",
    yaml_suffixes: Sequence[str] = (".yaml", ".yml"),
    **parse_kwargs: Any,
) -> ConditionsFile:
    """Read a conditions YAML/JSON file and parse it.

    Args:
        yaml_suffixes: extensions read with PyYAML; anything else is JSON.
        parse_kwargs: forwarded verbatim to `parse_conditions_data`.
    """
    with open(path, "r", encoding=encoding) as fh:
        if str(path).endswith(tuple(yaml_suffixes)):
            import yaml

            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    return parse_conditions_data(data, path=path, **parse_kwargs)


# --------------------------------------------------------- measured power
def apply_measured_power(
    cost_inputs: Dict[str, Dict[str, float]],
    measured: Dict[str, Optional[Dict[str, Any]]],
    *,
    kwh_gbp: Optional[float] = None,
    cost_defaults: Optional[Dict[str, Any]] = None,
    kwh_key: str = "kwh_gbp",
    gpu_hour_key: str = "p_gpu_hour",
    seconds_key: str = "query_gpu_seconds",
    watts_key: str = "watts",
    prefer: str = "measured",
    accept_upper_bound: bool = True,
    require_measured: Sequence[str] = (),
    printer: Optional[Callable[[str], None]] = print,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Put measured watts and serving seconds into each cell's cost block.

    ``p_gpu_hour`` is recomputed from the measured watts and the same
    electricity rate; ``query_gpu_seconds`` becomes the logged serving time. An
    unmeasured cell keeps what the file declared.

    Args:
        cost_inputs: parsed blocks, ``{cell: cost_per_answer kwargs}``.
        measured: ``{cell: measure_cell_power(...)["cost_inputs"]}``.
        kwh_gbp: electricity rate; None reads it from ``cost_defaults``.
        prefer: ``measured`` overwrites, ``declared`` measures for the record
            only, ``missing`` fills a cell that declared none.
        accept_upper_bound: False keeps the declared number when the meter fell
            back to a nameplate.
        require_measured: cells that must carry a real measurement, else raise.
        printer: one line per changed cell; None stays silent.

    Returns:
        ``(cost_inputs, provenance)``, the second naming each cell's source.
    """
    if prefer not in ("measured", "declared", "missing"):
        raise ValueError("prefer must be 'measured', 'declared' or 'missing'")
    rate = kwh_gbp
    if rate is None:
        rate = (cost_defaults or {}).get(kwh_key)
    out: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in cost_inputs.items()}
    provenance: Dict[str, str] = {}

    for name, block in out.items():
        got = (measured or {}).get(name)
        if not got or got.get(watts_key) is None:
            provenance[name] = "declared (no measurement)"
            continue
        if got.get("upper_bound") and not accept_upper_bound:
            provenance[name] = f"declared (measurement was an upper bound: {got.get('source')})"
            continue
        has_declared = gpu_hour_key in block
        if prefer == "declared" or (prefer == "missing" and has_declared):
            provenance[name] = f"declared; measured {got[watts_key]} W not applied"
            continue
        if rate is None:
            provenance[name] = (
                f"declared; measured {got[watts_key]} W unusable without {kwh_key}"
            )
            continue
        block[gpu_hour_key] = electricity_gpu_hour(float(got[watts_key]), float(rate))
        if got.get(seconds_key) is not None:
            block[seconds_key] = float(got[seconds_key])
        provenance[name] = str(got.get("source") or "measured")
        if printer:
            printer(f"  {name}: {watts_key}={got[watts_key]} measured, "
                    f"{gpu_hour_key}={block[gpu_hour_key]:.6f} GBP/h, "
                    f"{seconds_key}={block.get(seconds_key)}")

    missing = [n for n in require_measured
               if not str(provenance.get(n, "")).startswith("measured")]
    if missing:
        raise ValueError(
            f"cells {sorted(missing)} have no measured power: "
            + "; ".join(f"{n}: {provenance.get(n, 'absent')}" for n in sorted(missing))
        )
    return out, provenance


# ------------------------------------------------------------------- CFCA
def p_hat_from_report(
    report: Dict[str, Any],
    *,
    cells_key: str = "cells",
    cfca_key: str = "cfca",
    p_key: str = "p_faithful_cited_version",
) -> Dict[str, Optional[float]]:
    """``{cell: P(faithful.cited.right-version)}`` out of a written report.

    That all-pass rate is the CFCA denominator, the
    same number CFCA divides its GBP numerator by.
    """
    out: Dict[str, Optional[float]] = {}
    for name, cell in (report.get(cells_key) or {}).items():
        block = (cell or {}).get(cfca_key) or {}
        value = block.get(p_key)
        out[name] = None if value is None else float(value)
    return out


def cfca_for_cells(
    cost_inputs: Dict[str, Dict[str, float]],
    p_hat: Dict[str, Optional[float]],
    *,
    reference: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    cost_fn: Callable[..., Dict[str, float]] = cost_per_answer,
    cfca_fn: Callable[[float, float], Optional[float]] = cfca,
    p_hat_override: Optional[Dict[str, float]] = None,
    round_to: Optional[int] = 8,
    include_inputs: bool = True,
    include_breakdown: bool = True,
    include_deltas: bool = True,
    currency: str = "GBP",
    on_missing_p_hat: str = "keep",
) -> Dict[str, Any]:
    """GBP cost per answer and CFCA per cell, from declared quantities only.

    Pure arithmetic over ``cfca_cost``: no DB, no model, no network. The result
    is the block `run` writes under ``cfca_gbp`` and renders after the report
    table.

    Args:
        cost_inputs: ``{cell: cost_per_answer kwargs}`` from `ConditionsFile`.
        p_hat: ``{cell: all-pass rate}``, usually `p_hat_from_report`.
        reference: cell the deltas are measured against; None skips deltas.
        order: cell order in the result; None keeps ``cost_inputs`` order.
        cost_fn / cfca_fn: the two formulas, injectable for tests.
        p_hat_override: measured or hand-set rates that win over ``p_hat``.
        round_to: decimals for every emitted number; None keeps full precision.
        include_inputs: echo the quantities used, so the report is auditable
            without the yaml beside it.
        include_breakdown: keep the C_onetime / C_query / recurring split.
        include_deltas: per-cell difference against ``reference``.
        currency: label only; the arithmetic is whatever the prices are in.
        on_missing_p_hat: ``keep`` emits the cost with ``cfca_gbp: None``,
            ``skip`` drops the cell, ``raise`` refuses.
    """
    if on_missing_p_hat not in ("keep", "skip", "raise"):
        raise ValueError("on_missing_p_hat must be 'keep', 'skip' or 'raise'")

    def _round(value: Optional[float]) -> Optional[float]:
        if value is None or round_to is None:
            return value
        return round(float(value), round_to)

    names = list(order) if order is not None else list(cost_inputs)
    rates = dict(p_hat)
    rates.update(p_hat_override or {})

    cells: Dict[str, Any] = {}
    for name in names:
        inputs = cost_inputs.get(name)
        if not inputs:
            continue
        breakdown = cost_fn(**inputs)
        total = float(breakdown["Cost_per_answer"])
        rate = rates.get(name)
        if rate is None:
            if on_missing_p_hat == "raise":
                raise ValueError(f"cfca[{name}]: no P_hat; the cell has no scored answers")
            if on_missing_p_hat == "skip":
                continue
        value = None if not rate else cfca_fn(total, float(rate))
        entry: Dict[str, Any] = {
            "cost_per_answer": _round(total),
            "currency": currency,
            "p_hat": _round(rate),
            "cfca": _round(value),
        }
        if value is None:
            entry["note"] = (
                "CFCA = Cost / P_hat is undefined while P_hat is 0 or unmeasured"
            )
        if include_breakdown:
            entry["breakdown"] = {k: _round(v) for k, v in breakdown.items()}
        if include_inputs:
            entry["inputs"] = {k: _round(v) for k, v in inputs.items()}
        cells[name] = entry

    block: Dict[str, Any] = {"currency": currency, "cells": cells}
    if reference and include_deltas and reference in cells:
        ref = cells[reference]
        block["reference"] = reference
        block["deltas"] = {
            name: {
                "cost_per_answer": _round(_sub(cells[name]["cost_per_answer"], ref["cost_per_answer"])),
                "cfca": _round(_sub(cells[name]["cfca"], ref["cfca"])),
            }
            for name in cells if name != reference
        }
    elif reference:
        block["reference"] = reference
    return block


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def render_cfca_markdown(
    block: Dict[str, Any],
    *,
    title: str = "## CFCA (GBP, declared cost inputs)",
    order: Optional[Sequence[str]] = None,
    empty_text: str = "_No `cost:` block in conditions.yaml, so no CFCA was computed._",
    decimals: int = 6,
    show_breakdown: bool = True,
    show_deltas: bool = True,
    none_text: str = "n/a",
    warm: Optional[Dict[str, Any]] = None,
    warm_label: str = "(wo warmup)",
    warm_rows: Optional[Sequence[str]] = None,
    warm_note: bool = True,
) -> str:
    """The CFCA table appended to report.md. Reads the `cfca_for_cells` block.

    Args:
        title: markdown heading; "" omits it.
        order: row order; None keeps the block's own order.
        empty_text: printed when no cell declared cost inputs.
        decimals: rounding in the table only.
        show_breakdown: add the C_onetime/A, C_query and recurring/Q columns.
        show_deltas: add the difference against the reference cell.
        none_text: cell text for an undefined number.
        warm: the `warmup.warm_only_block` sensitivity. Each cell it covers
            gains one more row directly under that cell's own, so the pair
            reads across: the same cell with the sitting's warm-up and without
            it, every figure on the same axis. Reported rows keep their numbers
            and their position, because the sensitivity is read against the
            run's own row and never over it. None omits the rows and the note.
        warm_label: suffix on the added row's cell name.
        warm_rows: which cells get the extra row; None is every cell the
            sensitivity covers.
        warm_note: print what each cell dropped, and the rule, under the table.
    """
    cells = (block or {}).get("cells") or {}
    lines: List[str] = []
    if title:
        lines += [title, ""]
    if not cells:
        lines.append(empty_text)
        return "\n".join(lines) + "\n"

    def fmt(value: Optional[float]) -> str:
        return none_text if value is None else f"{float(value):.{decimals}f}"

    reference = block.get("reference")
    deltas = block.get("deltas") or {}
    warm_cells = (warm or {}).get("cells") or {}
    warm_deltas = (warm or {}).get("deltas") or {}
    wanted = None if warm_rows is None else {str(c) for c in warm_rows}
    header = ["cell", "cost/answer", "P_hat", "CFCA"]
    if show_breakdown:
        header += ["C_onetime/A", "C_query", "recurring/Q"]
    if show_deltas and reference:
        header += [f"dCFCA vs {reference}"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def one_row(label: str, name: str, entry: Dict[str, Any], against: Dict[str, Any]) -> str:
        row = [label, fmt(entry.get("cost_per_answer")), fmt(entry.get("p_hat")), fmt(entry.get("cfca"))]
        if show_breakdown:
            bd = entry.get("breakdown") or {}
            row += [fmt(bd.get("C_onetime/A")), fmt(bd.get("C_query")), fmt(bd.get("recurring/Q"))]
        if show_deltas and reference:
            row += ["-" if name == reference else fmt((against.get(name) or {}).get("cfca"))]
        return "| " + " | ".join(row) + " |"

    used: List[str] = []
    for name in (order if order is not None else cells):
        entry = cells.get(name)
        if not entry:
            continue
        lines.append(one_row(f"`{name}`", name, entry, deltas))
        warm_entry = warm_cells.get(name) if wanted is None or name in wanted else None
        if warm_entry:
            # Under its own row, on the same axis: the delta is warm against the
            # warm reference, so a reading never crosses the two measurements.
            lines.append(one_row(f"`{name}` {warm_label}", name, warm_entry, warm_deltas))
            used.append(name)
    lines.append("")
    lines.append(
        f"CFCA = cost per answer / P(faithful.cited.right-version), {block.get('currency', 'GBP')}, "
        "from the `cost:` quantities in conditions.yaml."
    )
    if used and warm_note:
        lines.append("")
        lines.append(_warm_note_text(warm, label=warm_label, shown=used))
    return "\n".join(lines) + "\n"


def _warm_note_text(warm: Dict[str, Any], *, label: str, shown: Optional[Sequence[str]] = None) -> str:
    """The line under the table: what each cell dropped, and what moved with it."""
    audit = (warm or {}).get("warmup") or {}
    drop = int((warm or {}).get("drop") or 1)
    dropped = ", ".join(
        "{cell} {ids} {secs:.2f} s ({share:.0f}%)".format(
            cell=f"`{name}`",
            ids="+".join(str(d.get("query_id") or "?") for d in (entry.get("dropped") or [])),
            secs=float(entry.get("dropped_s") or 0.0),
            share=100 * float(entry.get("dropped_share") or 0.0),
        )
        for name, entry in audit.items()
    )
    missing = [name for name in audit if shown is not None and name not in set(shown)]
    rest = (
        f" The same drop was applied to every cell; {', '.join(f'`{n}`' for n in missing)} "
        f"carry it in `cfca_warm_only` in report.json rather than in this table."
        if missing else ""
    )
    return (
        f"A `{label}` row re-prices one input and nothing else: the first {drop} priced answer of "
        f"each sitting is dropped, so `query_gpu_seconds` = (total_s - dropped_s) / (n_answers - "
        f"{drop}), and the same P_hat divides it. A cold model load is billed to whichever answer "
        f"waits for it, which is a property of the sitting and not of the condition. Dropped: "
        f"{dropped}.{rest} Watts, tokens, storage, the one-time build and every score are the run's "
        f"own; nothing here is re-scored, and the reported rows stand."
    )
