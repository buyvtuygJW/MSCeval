"""
Cell names for the 2x2, defined once here and imported everywhere else.

Three words, one referent each, matching 06-fig-experiment-design.svg:

``baseline``
    The one-time construction that fixes the pipeline: 256-tok truncation
    removed, embedder window >=512 tok, chunk size ~400 tok, hybrid BM25 and
    dense fused by RRF (k~60). A build step and a validity prerequisite, so it
    never names a cell and carries no delta of its own.
``control``
    The 2x2 reference cell. It runs that fixed pipeline with RCTS ~400 tok
    chunking and the unadapted base model, and every other cell is reported as
    a CFCA delta against it ("the control cell" in the figure).
``prebaseline``
    The app as it stands before the construction lands: RCTS chunking still
    uncorrected, base model unadapted. It sits below the grid rather than in
    it, and it is dumped so its numbers survive the re-ingest that the chunking
    correction forces. Its delta against ``control`` measures what that
    construction changed, reported uncorrected and outside the scored grid, so
    ``split_grid`` keeps it out of the 2x2 table and its Bonferroni family.

Any other vocabulary still works. Every function takes the candidate names and
the fallbacks as explicit arguments, so a caller can drive the whole module
from a conditions file that shares none of these keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- the table
CONTROL = "control"
CHUNK_OPT = "chunk_opt"
QDORA = "qdora"
COMBINED = "combined"
PREBASELINE = "prebaseline"
PREBASELINE2 = "prebaseline2"

# Factor levels, worded as in the figure's row and column headers.
RCTS = "RCTS ~400 tok"
OPTIMISED = "optimised chunking"
BASE_MODEL = "base model"
QDORA_MODEL = "QDoRA-merged"
RCTS_UNCORRECTED = "RCTS, uncorrected"
TOKEN_BUDGET = "token-budgeted chunking"


@dataclass(frozen=True)
class CellSpec:
    """One cell: its key, the figure's title for it, and its two factor levels."""

    key: str
    title: str
    chunking: str
    model: str
    in_2x2: bool = True
    note: str = ""

    def label(self, *, with_factors: bool = False, with_key: bool = False,
              sep: str = " / ") -> str:
        parts: List[str] = [self.title]
        if with_factors:
            parts += [self.chunking, self.model]
        text = sep.join(parts)
        return f"{self.key} ({text})" if with_key else text


CELLS: Dict[str, CellSpec] = {
    CONTROL: CellSpec(
        key=CONTROL,
        title="Controlled baseline",
        chunking=RCTS,
        model=BASE_MODEL,
        note="the control cell; every other cell is a CFCA delta against it",
    ),
    CHUNK_OPT: CellSpec(
        key=CHUNK_OPT,
        title="Chunking-optimised RAG",
        chunking=OPTIMISED,
        model=BASE_MODEL,
        note="chunking strategy varies, chunk size stays fixed",
    ),
    QDORA: CellSpec(
        key=QDORA,
        title="Fine-tuned RAG",
        chunking=RCTS,
        model=QDORA_MODEL,
        note="QDoRA adapter merged, retrieval stays on",
    ),
    COMBINED: CellSpec(
        key=COMBINED,
        title="Combined",
        chunking=OPTIMISED,
        model=QDORA_MODEL,
        note="optimised chunking plus QDoRA",
    ),
    PREBASELINE: CellSpec(
        key=PREBASELINE,
        title="Pre-baseline app",
        chunking=RCTS_UNCORRECTED,
        model=BASE_MODEL,
        in_2x2=False,
        note="the app before baseline construction; below the grid, dumped "
             "before the re-ingest so its numbers survive",
    ),
    PREBASELINE2: CellSpec(
        key=PREBASELINE2,
        title="Token-budgeted chunking",
        chunking=TOKEN_BUDGET,
        model=BASE_MODEL,
        in_2x2=False,
        note="the pre-baseline app with chunks sized by the embedder's own "
             "tokenizer instead of a word count; the chunker alone below the "
             "grid, dumped before the construction's re-ingest",
    ),
}

GRID_ORDER: Tuple[str, ...] = (CONTROL, CHUNK_OPT, QDORA, COMBINED)
OFFGRID_ORDER: Tuple[str, ...] = (PREBASELINE, PREBASELINE2)
CELL_ORDER: Tuple[str, ...] = GRID_ORDER + OFFGRID_ORDER

# Deltas are measured against the control cell, and a dump with no cell name
# given is the current app rather than anything in the grid.
DEFAULT_REFERENCE: str = CONTROL
DEFAULT_DUMP_CELL: str = PREBASELINE

# Report keys. DELTAS_KEY is what report.py writes; the legacy names are read
# so an older report.json still renders. OFFGRID_DELTAS_KEY is a separate block
# so an off-grid cell never lands in the scored table or the Bonferroni family.
DELTAS_KEY: str = "deltas_vs_control"
LEGACY_DELTAS_KEYS: Tuple[str, ...] = ("deltas_vs_reference", "deltas_vs_baseline")
OFFGRID_DELTAS_KEY: str = "offgrid_deltas"

# Keys that earlier conditions files used. Recognised for messages and for
# opt-in rewriting; nothing rewrites a user's name unless it asks to.
LEGACY_ALIASES: Dict[str, str] = {"baseline": CONTROL, "qlora": QDORA}

# Top-level keys a conditions file may use to name its reference cell.
REFERENCE_YAML_KEYS: Tuple[str, ...] = ("reference", "baseline")


# ---------------------------------------------------------------- lookups
def spec(name: str, *, aliases: Optional[Dict[str, str]] = None,
         default: Optional[CellSpec] = None) -> Optional[CellSpec]:
    """The CellSpec for ``name``, or ``default`` when the name is not ours."""
    if name in CELLS:
        return CELLS[name]
    if aliases:
        mapped = aliases.get(name)
        if mapped in CELLS:
            return CELLS[mapped]
    return default


def is_known(name: str, *, aliases: Optional[Dict[str, str]] = None) -> bool:
    return spec(name, aliases=aliases) is not None


def canonical(name: str, *, aliases: Dict[str, str] = LEGACY_ALIASES,
              unknown: str = "keep") -> str:
    """
    Map a legacy key onto its current one.

    Args:
        aliases: legacy -> current map; pass ``{}`` to disable rewriting.
        unknown: ``keep`` returns an unrecognised name unchanged, ``raise``
            rejects it.
    """
    if name in CELLS:
        return name
    mapped = (aliases or {}).get(name)
    if mapped:
        return mapped
    if unknown == "raise":
        raise ValueError(f"unknown cell name {name!r}; known: {list(CELLS)}")
    return name


def label(name: str, *, with_factors: bool = False, with_key: bool = False,
          aliases: Optional[Dict[str, str]] = None, sep: str = " / ",
          unknown: Optional[str] = None) -> str:
    """
    Display label for ``name``. Unknown names fall back to ``unknown``, or to
    the name itself when ``unknown`` is None.
    """
    found = spec(name, aliases=aliases)
    if found is None:
        return name if unknown is None else unknown
    return found.label(with_factors=with_factors, with_key=with_key, sep=sep)


def cell_names(*, include_2x2: bool = True, include_offgrid: bool = True,
               extra: Sequence[str] = (), extra_first: bool = False,
               order: Sequence[str] = CELL_ORDER) -> List[str]:
    """
    The names to offer in a picker: our own table plus whatever ``extra``
    carries (a parsed conditions file, say), in ``order``, without duplicates.

    Args:
        include_2x2: keep the four grid cells.
        include_offgrid: keep the cells outside the grid (prebaseline).
        extra: caller-supplied names, appended in their own order.
        extra_first: put ``extra`` ahead of the table instead of after it.
    """
    mine = [
        n for n in order
        if n in CELLS and (include_2x2 if CELLS[n].in_2x2 else include_offgrid)
    ]
    groups = ([list(extra), mine] if extra_first else [mine, list(extra)])
    out: List[str] = []
    for group in groups:
        for name in group:
            if name and name not in out:
                out.append(name)
    return out


def split_grid(
    names: Iterable[str],
    *,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    aliases: Dict[str, str] = LEGACY_ALIASES,
    unknown: str = "grid",
    order: str = "given",
    reference: Optional[str] = None,
    reference_in_grid: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Split condition names into the cells scored inside the 2x2 and the cells
    measured but never scored there (``prebaseline``).

    This split is what lets a before/after window exist without touching the
    main result: callers build the paired-delta table and the Bonferroni family
    from the first list only, so declaring an off-grid cell cannot move a single
    grid number.

    Args:
        grid_names: names that belong to the scored grid.
        offgrid_names: names measured outside the grid.
        aliases: legacy name -> current name, so an older conditions file's keys
            land in the right list without being rewritten.
        unknown: where a name in neither list goes. ``grid`` (default) scores any
            other vocabulary exactly as before; ``offgrid`` measures it without
            scoring it; ``drop`` discards it; ``raise`` rejects it.
        order: ``given`` keeps the caller's order inside each list; ``table``
            reorders each list by ``grid_names`` / ``offgrid_names``.
        reference: the cell deltas are measured against, if the caller has one.
        reference_in_grid: an off-grid ``reference`` moves to the front of the
            grid list, because the cell everything is measured against cannot
            sit outside the comparison. False leaves it off-grid, which makes
            the grid list referenceless and is only useful for inspection.
    """
    grid_set, off_set = set(grid_names), set(offgrid_names)
    on: List[str] = []
    off: List[str] = []
    for name in names:
        current = name if name in grid_set or name in off_set else canonical(name, aliases=aliases)
        if current in off_set:
            off.append(name)
        elif current in grid_set or unknown == "grid":
            on.append(name)
        elif unknown == "offgrid":
            off.append(name)
        elif unknown == "drop":
            continue
        else:
            raise ValueError(
                f"cell {name!r} is neither grid {list(grid_names)} "
                f"nor off-grid {list(offgrid_names)}"
            )
    if order == "table":
        rank = {n: i for i, n in enumerate(grid_names)}
        on.sort(key=lambda n: rank.get(n, len(rank)))
        rank = {n: i for i, n in enumerate(offgrid_names)}
        off.sort(key=lambda n: rank.get(n, len(rank)))
    if reference and reference_in_grid and reference in off:
        off.remove(reference)
        on.insert(0, reference)
    return on, off


def grid(*, order: Sequence[str] = GRID_ORDER) -> Dict[str, Dict[str, str]]:
    """``{chunking level: {model level: cell key}}`` for rendering the 2x2."""
    rows: Dict[str, Dict[str, str]] = {}
    for name in order:
        cell = CELLS[name]
        rows.setdefault(cell.chunking, {})[cell.model] = cell.key
    return rows


# ---------------------------------------------------------------- reference
def resolve_reference(
    names: Iterable[str],
    *,
    requested: Optional[str] = None,
    default: Optional[str] = DEFAULT_REFERENCE,
    fallback: str = "first",
    aliases: Dict[str, str] = LEGACY_ALIASES,
    required: bool = False,
) -> Optional[str]:
    """
    Pick the cell that deltas are measured against.

    Precedence: ``requested`` as written, then ``requested`` through
    ``aliases``, then ``default``, then ``fallback``. Each step only wins if
    the name is actually among ``names``.

    Args:
        fallback: ``first`` takes the first of ``names``, ``default`` returns
            ``default`` even when absent, ``none`` gives None, ``raise`` raises.
        required: raise instead of returning None.
    """
    pool = [n for n in names]
    if requested and requested in pool:
        return requested
    if requested:
        mapped = canonical(requested, aliases=aliases)
        if mapped in pool:
            return mapped
    if default and default in pool:
        return default
    if fallback == "first" and pool:
        return pool[0]
    if fallback == "default":
        return default
    if fallback == "raise":
        raise ValueError(f"no reference cell among {pool} (requested {requested!r})")
    if required:
        raise ValueError(f"no reference cell among {pool} (requested {requested!r})")
    return None


def reference_from_mapping(
    data: Dict, *, keys: Sequence[str] = REFERENCE_YAML_KEYS
) -> Optional[str]:
    """The reference cell a conditions file asked for, under any of ``keys``."""
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None


def contrast_key(cell: str, reference: str = DEFAULT_REFERENCE, *,
                 sep: str = "_vs_") -> str:
    return f"{cell}{sep}{reference}"


def deltas_key(reference: str = DEFAULT_REFERENCE, *, follow_reference: bool = False,
               fixed: str = DELTAS_KEY, prefix: str = "deltas_vs_") -> str:
    """
    The report field the delta block is written under. Fixed at ``fixed`` by
    default; ``follow_reference=True`` names it after the reference actually
    used, which matters only if the reference is not the control cell.
    """
    return f"{prefix}{reference}" if follow_reference else fixed


def deltas_of(report: Dict, *, keys: Sequence[str] = (),
              default: Optional[Dict] = None) -> Dict:
    """
    The delta block of a report, whichever key it was written under. Tries
    ``keys`` first, then the current key, then the legacy ones.
    """
    for key in list(keys) + [DELTAS_KEY] + list(LEGACY_DELTAS_KEYS):
        if key in report:
            return report[key]
    return {} if default is None else default


def offgrid_deltas_key(reference: str = DEFAULT_REFERENCE, *, follow_reference: bool = False,
                       fixed: str = OFFGRID_DELTAS_KEY,
                       prefix: str = "offgrid_deltas_vs_") -> str:
    """
    The report field the off-grid delta block is written under. Kept separate
    from ``deltas_key`` so a reader can never mistake a validity check for one
    of the scored contrasts.
    """
    return f"{prefix}{reference}" if follow_reference else fixed


def offgrid_deltas_of(report: Dict, *, keys: Sequence[str] = (),
                      default: Optional[Dict] = None) -> Dict:
    """The off-grid delta block of a report, whichever key it was written under."""
    for key in list(keys) + [OFFGRID_DELTAS_KEY]:
        if key in report:
            return report[key]
    prefix = "offgrid_deltas_vs_"
    for key in report:
        if isinstance(key, str) and key.startswith(prefix):
            return report[key]
    return {} if default is None else default


# ---------------------------------------------------------------- messages
def legacy_notes(names: Iterable[str], *, aliases: Dict[str, str] = LEGACY_ALIASES,
                 template: str = "cell {old!r} is the old name for {new!r}") -> List[str]:
    """One note per legacy name in ``names``, for a CLI print or a UI warning."""
    return [
        template.format(old=name, new=aliases[name])
        for name in names
        if name in aliases and name not in CELLS
    ]


# ------------------------------------------------------------- file layout
#: One cell is one JSON file, and this module owns where that file sits.
#: `pipeline`, `cells_app` and the viewer all ask here, so the four copies of
#: ``<out_dir>/cells/<name>.json`` that used to drift cannot drift again.
CELL_SUBDIR: str = "cells"
CELL_SUFFIX: str = ".json"
CELL_NAME_FMT: str = "{name}{suffix}"


def default_out_dir() -> str:
    """``settings.output_dir``, read at call time so ``--out`` still wins.

    Imported inside the call rather than at module import, so a caller that
    only wants the cell names never pulls the config in.
    """
    from .config import settings

    return settings.output_dir


def cell_dir_for(out_dir: Optional[str] = None, *, subdir: str = CELL_SUBDIR,
                 out_dir_default: Optional[str] = None) -> str:
    """``<out_dir>/cells``, so ``--out`` moves the snapshots with the report.

    Args:
        out_dir: the report directory; None falls back to ``out_dir_default``.
        subdir: the leaf; "" keeps the files in ``out_dir`` itself.
        out_dir_default: that fallback; None reads ``settings.output_dir`` when
            the call happens, which lets a test pin a directory without
            touching the config.
    """
    base = out_dir or out_dir_default or default_out_dir()
    return os.path.join(base, subdir) if subdir else base


def cell_file_path(name: str, *, path: Optional[str] = None,
                   cell_dir: Optional[str] = None,
                   out_dir: Optional[str] = None,
                   subdir: str = CELL_SUBDIR,
                   suffix: str = CELL_SUFFIX,
                   name_fmt: str = CELL_NAME_FMT,
                   out_dir_default: Optional[str] = None,
                   make_dirs: bool = False) -> str:
    """The JSON file holding one cell: ``<cell_dir>/<name>.json``.

    Args:
        name: cell name, e.g. ``control``.
        path: an explicit file, returned as given. A UI text box, an archive
            target or a ``.json.new`` scratch file overrides the derivation.
        cell_dir: the directory the files live in; None derives it from
            ``out_dir`` / ``subdir`` through `cell_dir_for`.
        out_dir / subdir / out_dir_default: handed to `cell_dir_for`.
        suffix / name_fmt: the file naming rule, for a dump that wants
            ``cell-control.json`` or a second suffix.
        make_dirs: create the returned path's directory when it is missing.
    """
    if cell_dir is None:
        cell_dir = cell_dir_for(out_dir, subdir=subdir,
                                out_dir_default=out_dir_default)
    out = path or os.path.join(cell_dir, name_fmt.format(name=name, suffix=suffix))
    if make_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    return out
