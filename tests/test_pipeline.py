"""
The snapshot policy is the only thing standing between a re-ingest and a lost
before/after measurement, so it is tested without a database: an existing file is
never re-dumped, a refresh that comes back empty must not destroy the good file,
an empty first dump must not be left behind for the next run to reuse, and a dump
whose benchmark questions were never asked in the app is rejected on its linkage
count rather than written as a 0-linked file that later runs reuse.
"""
import json
import os
from dataclasses import dataclass

import pytest

from veridic_eval.cells import CONTROL, PREBASELINE, QDORA
from veridic_eval.conditions import ConditionsFile, DeclaredCell
from veridic_eval.config import Condition
from veridic_eval.pipeline import (
    EMPTY,
    FAILED,
    NO_LINKED,
    NO_RECORDS,
    REUSED,
    SKIPPED,
    WRITTEN,
    cell_dir_for,
    load_cells,
    snapshot_cells,
    snapshot_path,
    snapshot_verdict,
)


@dataclass
class Rec:
    """Stands in for a QueryRecord; only `linked` is read by the loader."""

    linked: bool = True


def cell(name, *, snapshot=True, conversation_ids=("c1",), query_ids=None):
    return DeclaredCell(
        name=name,
        condition=Condition(name=name, conversation_ids=list(conversation_ids)),
        snapshot=snapshot,
        query_ids=query_ids,
    )


def dumper(n_records=2, *, n_linked=None, fail=False, per_cell=None):
    """A dump_cell stand-in that writes a marker file and counts its calls.

    `n_linked=None` links every row, the answered case; `n_linked=0` is the cell
    whose questions were never asked, which still writes one record per query.
    """
    calls = []

    def dump(**kwargs):
        calls.append(kwargs)
        if fail:
            raise RuntimeError("db down")
        rows = (per_cell or {}).get(kwargs["cell"], n_records)
        linked = rows if n_linked is None else min(n_linked, rows)
        with open(kwargs["path"], "w", encoding="utf-8") as fh:
            json.dump({"cell": kwargs["cell"], "rows": rows, "linked": linked}, fh)
        return {"path": kwargs["path"], "cell": kwargs["cell"],
                "n_records": rows, "n_linked": linked}

    return dump, calls


def rows_in(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rows"]


# ------------------------------------------------------------------- paths
def test_snapshots_live_beside_the_report_by_default():
    assert cell_dir_for("out") == os.path.join("out", "cells")
    assert cell_dir_for("out", subdir="archive") == os.path.join("out", "archive")
    assert snapshot_path(PREBASELINE, "out/cells") == os.path.join("out/cells", "prebaseline.json")


# ---------------------------------------------------------------- snapshots
def test_first_run_writes_a_file_per_declared_cell(tmp_path):
    dump, calls = dumper(3)
    out = snapshot_cells([cell(PREBASELINE), cell(CONTROL)], cell_dir=str(tmp_path),
                         dump_fn=dump, printer=None)
    assert [c["cell"] for c in calls] == [PREBASELINE, CONTROL]
    assert out[PREBASELINE]["status"] == WRITTEN
    assert out[CONTROL]["n_linked"] == 3
    assert os.path.exists(snapshot_path(CONTROL, str(tmp_path)))


def test_an_existing_snapshot_is_reused_and_never_re_dumped(tmp_path):
    dump, calls = dumper(2)
    snapshot_cells([cell(PREBASELINE)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    again = snapshot_cells([cell(PREBASELINE)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    assert len(calls) == 1
    assert again[PREBASELINE]["status"] == REUSED
    assert rows_in(snapshot_path(PREBASELINE, str(tmp_path))) == 2


def test_refresh_replaces_the_file_and_leaves_no_scratch_copy(tmp_path):
    dump, _ = dumper(2)
    snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    fresh, _ = dumper(7)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), refresh=[CONTROL],
                         dump_fn=fresh, printer=None)
    assert out[CONTROL]["status"] == WRITTEN
    assert rows_in(snapshot_path(CONTROL, str(tmp_path))) == 7
    assert os.listdir(tmp_path) == ["control.json"]


def test_a_refresh_that_comes_back_empty_keeps_the_good_snapshot(tmp_path):
    dump, _ = dumper(4)
    snapshot_cells([cell(PREBASELINE)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    gone, _ = dumper(0)
    out = snapshot_cells([cell(PREBASELINE)], cell_dir=str(tmp_path), refresh=[PREBASELINE],
                         dump_fn=gone, printer=None)
    assert out[PREBASELINE]["status"] == EMPTY
    assert out[PREBASELINE]["kept_previous"] is True
    assert rows_in(snapshot_path(PREBASELINE, str(tmp_path))) == 4
    assert os.listdir(tmp_path) == ["prebaseline.json"]


def test_an_empty_first_dump_is_not_left_for_the_next_run_to_reuse(tmp_path):
    dump, _ = dumper(0)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    assert out[CONTROL]["status"] == EMPTY
    assert not os.path.exists(snapshot_path(CONTROL, str(tmp_path)))
    assert os.listdir(tmp_path) == []


def test_an_empty_dump_can_be_kept_on_purpose(tmp_path):
    dump, _ = dumper(0)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump,
                         keep_empty=True, printer=None)
    assert out[CONTROL]["status"] == WRITTEN
    assert os.path.exists(snapshot_path(CONTROL, str(tmp_path)))


def test_a_dump_nobody_asked_the_questions_of_is_rejected_on_its_linkage(tmp_path):
    dump, _ = dumper(4, n_linked=0)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    assert out[CONTROL]["status"] == EMPTY
    assert out[CONTROL]["reason"] == NO_LINKED
    assert out[CONTROL]["n_records"] == 4 and out[CONTROL]["n_linked"] == 0
    assert os.listdir(tmp_path) == []


def test_the_rejected_dump_is_retried_and_kept_once_the_answers_are_logged(tmp_path):
    unasked, _ = dumper(4, n_linked=0)
    snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=unasked, printer=None)
    answered, _ = dumper(4)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=answered, printer=None)
    assert out[CONTROL]["status"] == WRITTEN
    assert out[CONTROL]["n_linked"] == 4


def test_an_unlinked_dump_is_kept_when_the_caller_lowers_the_rule(tmp_path):
    dump, _ = dumper(4, n_linked=0)
    for kwargs in ({"keep_unlinked": True}, {"min_linked": 0}, {"keep_empty": True}):
        out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump,
                            refresh=[CONTROL], printer=None, **kwargs)
        assert out[CONTROL]["status"] == WRITTEN, kwargs
        assert os.path.exists(snapshot_path(CONTROL, str(tmp_path)))


def test_a_caller_can_replace_the_keep_rule_outright(tmp_path):
    dump, _ = dumper(4, n_linked=0)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump,
                         verdict_fn=lambda *a, **k: (False, "mine"),
                         empty_notes={"mine": "my rule says no"}, printer=None)
    assert out[CONTROL]["status"] == EMPTY
    assert out[CONTROL]["reason"] == "mine"
    assert os.listdir(tmp_path) == []


def test_the_keep_rule_reads_records_first_then_linkage():
    assert snapshot_verdict(2, 2) == (True, "")
    assert snapshot_verdict(0, 0) == (False, NO_RECORDS)
    assert snapshot_verdict(4, 0) == (False, NO_LINKED)
    assert snapshot_verdict(0, 0, keep_empty=True) == (True, "")
    assert snapshot_verdict(4, 0, keep_unlinked=True) == (True, "")
    assert snapshot_verdict(4, 0, min_linked=0) == (True, "")
    assert snapshot_verdict(4, 2, min_linked=3) == (False, NO_LINKED)
    assert snapshot_verdict(2, 2, min_records=3) == (False, NO_RECORDS)


def test_a_cell_with_snapshot_off_is_skipped(tmp_path):
    dump, calls = dumper()
    out = snapshot_cells([cell(CONTROL, snapshot=False)], cell_dir=str(tmp_path),
                         dump_fn=dump, printer=None)
    assert out[CONTROL]["status"] == SKIPPED
    assert calls == []
    out = snapshot_cells([cell(CONTROL, snapshot=False)], cell_dir=str(tmp_path),
                         dump_fn=dump, enabled_only=False, printer=None)
    assert out[CONTROL]["status"] == WRITTEN


def test_only_and_skip_narrow_the_set(tmp_path):
    dump, calls = dumper()
    cells = [cell(PREBASELINE), cell(CONTROL), cell(QDORA)]
    out = snapshot_cells(cells, cell_dir=str(tmp_path), only=[CONTROL], dump_fn=dump, printer=None)
    assert [c["cell"] for c in calls] == [CONTROL]
    assert out[PREBASELINE]["status"] == SKIPPED
    calls.clear()
    snapshot_cells(cells, cell_dir=str(tmp_path), skip=[CONTROL, QDORA],
                   dump_fn=dump, printer=None)
    assert [c["cell"] for c in calls] == [PREBASELINE]


def test_a_failed_dump_is_recorded_or_raised_as_asked(tmp_path):
    dump, _ = dumper(fail=True)
    out = snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    assert out[CONTROL]["status"] == FAILED
    assert "db down" in out[CONTROL]["error"]
    assert not os.path.exists(snapshot_path(CONTROL, str(tmp_path)))
    with pytest.raises(RuntimeError):
        snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump,
                       on_error="raise", printer=None)


def test_the_window_and_query_ids_reach_the_dumper(tmp_path):
    dump, calls = dumper()
    declared = cell(CONTROL, conversation_ids=("c7",), query_ids=["q1"])
    snapshot_cells([declared], cell_dir=str(tmp_path), benchmark_path="b.yaml",
                   dump_fn=dump, printer=None)
    assert calls[0]["conversation_ids"] == ["c7"]
    assert calls[0]["query_ids"] == ["q1"]
    assert calls[0]["benchmark_path"] == "b.yaml"


def test_extra_dump_arguments_are_passed_through(tmp_path):
    dump, calls = dumper()
    snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump,
                   dump_kwargs={"gold_require_all": True}, printer=None)
    assert calls[0]["gold_require_all"] is True


# --------------------------------------------------------------- loading back
def _file(cells, *, grid=(CONTROL,), offgrid=(PREBASELINE,)):
    return ConditionsFile(cells=cells, reference=CONTROL, grid=tuple(grid), offgrid=tuple(offgrid))


def test_a_snapshotted_cell_is_read_from_disk_and_a_missing_one_from_the_logs(tmp_path):
    dump, _ = dumper(2)
    snapshot_cells([cell(PREBASELINE)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    read_calls = []

    def read_fn(path, **kwargs):
        read_calls.append((path, kwargs))
        return [Rec(), Rec()]

    def extract_fn(queries, condition):
        return [Rec()]

    cells, sources = load_cells(
        _file([cell(PREBASELINE), cell(CONTROL)]), cell_dir=str(tmp_path),
        queries=[], read_fn=read_fn, extract_fn=extract_fn, printer=None,
    )
    assert sources == {PREBASELINE: "snapshot", CONTROL: "live"}
    assert len(cells[PREBASELINE]) == 2 and len(cells[CONTROL]) == 1
    assert read_calls[0][1]["expect_cell"] == PREBASELINE


def test_live_reads_can_be_switched_off_entirely(tmp_path):
    cells, sources = load_cells(
        _file([cell(CONTROL)]), cell_dir=str(tmp_path), live_fallback=False,
        queries=[], read_fn=lambda *a, **k: [Rec()], extract_fn=lambda *a: [Rec()],
        printer=None,
    )
    assert cells == {} and sources == {}


def test_live_mode_ignores_the_snapshot_on_disk(tmp_path):
    dump, _ = dumper(2)
    snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), dump_fn=dump, printer=None)
    cells, sources = load_cells(
        _file([cell(CONTROL)]), cell_dir=str(tmp_path), source="live", queries=[],
        read_fn=lambda *a, **k: [Rec(), Rec()], extract_fn=lambda *a: [Rec()],
        printer=None,
    )
    assert sources == {CONTROL: "live"} and len(cells[CONTROL]) == 1


def test_an_empty_cell_is_left_out_of_the_report_unless_asked_for(tmp_path):
    cells, _ = load_cells(
        _file([cell(CONTROL)]), cell_dir=str(tmp_path), queries=[],
        read_fn=lambda *a, **k: [], extract_fn=lambda *a: [], printer=None,
    )
    assert cells == {}
    kept, _ = load_cells(
        _file([cell(CONTROL)]), cell_dir=str(tmp_path), queries=[], drop_empty=False,
        read_fn=lambda *a, **k: [], extract_fn=lambda *a: [], printer=None,
    )
    assert kept == {CONTROL: []}


def test_a_cell_the_file_dropped_from_both_lists_is_not_scored(tmp_path):
    cells, _ = load_cells(
        _file([cell(CONTROL), cell("pilot")]), cell_dir=str(tmp_path), queries=[],
        read_fn=lambda *a, **k: [Rec()], extract_fn=lambda *a: [Rec()], printer=None,
    )
    assert list(cells) == [CONTROL]


def test_bad_source_and_bad_error_policy_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_cells(_file([cell(CONTROL)]), cell_dir=str(tmp_path), source="guess")
    with pytest.raises(ValueError):
        snapshot_cells([cell(CONTROL)], cell_dir=str(tmp_path), on_error="shrug")
