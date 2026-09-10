"""
The off-grid split is what fixes the Bonferroni family at len(grid) - 1, so it
is tested on its own: declaring `prebaseline` must leave the scored grid, the
family size and the delta-block key exactly as they were without it.
"""
import pytest

from veridic_eval.cells import (
    CHUNK_OPT,
    COMBINED,
    CONTROL,
    DELTAS_KEY,
    GRID_ORDER,
    OFFGRID_DELTAS_KEY,
    PREBASELINE,
    QDORA,
    contrast_key,
    deltas_key,
    deltas_of,
    offgrid_deltas_key,
    offgrid_deltas_of,
    split_grid,
)

ALL_FIVE = [PREBASELINE, CONTROL, CHUNK_OPT, QDORA, COMBINED]


def _family(grid):
    """The divisor report.py derives from a grid."""
    return max(1, len(grid) - 1)


# ---------------------------------------------------------------- the split
def test_prebaseline_leaves_the_grid_and_the_family_at_three():
    grid, offgrid = split_grid(ALL_FIVE)
    assert grid == [CONTROL, CHUNK_OPT, QDORA, COMBINED]
    assert offgrid == [PREBASELINE]
    assert _family(grid) == 3


def test_declaring_prebaseline_changes_no_grid_number():
    with_off, offgrid = split_grid(ALL_FIVE)
    without_off, empty = split_grid(list(GRID_ORDER))
    assert with_off == without_off
    assert _family(with_off) == _family(without_off) == 3
    assert offgrid == [PREBASELINE]
    assert empty == []


def test_four_cell_grid_alone_has_no_offgrid_block():
    grid, offgrid = split_grid(list(GRID_ORDER))
    assert grid == list(GRID_ORDER)
    assert offgrid == []


def test_legacy_names_land_in_the_grid_unrewritten():
    grid, offgrid = split_grid(["baseline", "qlora"])
    assert grid == ["baseline", "qlora"]
    assert offgrid == []
    assert _family(grid) == 1


def test_unknown_cell_defaults_into_the_grid_and_enlarges_the_family():
    grid, offgrid = split_grid(list(GRID_ORDER) + ["pilot"])
    assert grid[-1] == "pilot"
    assert offgrid == []
    assert _family(grid) == 4


def test_unknown_cell_can_be_measured_off_grid_instead():
    grid, offgrid = split_grid(list(GRID_ORDER) + ["pilot"], unknown="offgrid")
    assert grid == list(GRID_ORDER)
    assert offgrid == ["pilot"]
    assert _family(grid) == 3


def test_unknown_cell_can_be_dropped_or_rejected():
    grid, offgrid = split_grid(list(GRID_ORDER) + ["pilot"], unknown="drop")
    assert grid == list(GRID_ORDER)
    assert offgrid == []
    with pytest.raises(ValueError):
        split_grid(list(GRID_ORDER) + ["pilot"], unknown="raise")


def test_given_order_is_kept_and_table_order_is_restored():
    scrambled = [COMBINED, QDORA, CHUNK_OPT, CONTROL]
    assert split_grid(scrambled)[0] == scrambled
    assert split_grid(scrambled, order="table")[0] == list(GRID_ORDER)


def test_offgrid_reference_is_pulled_into_the_grid_unless_refused():
    grid, offgrid = split_grid([PREBASELINE, CONTROL, QDORA], reference=PREBASELINE)
    assert grid == [PREBASELINE, CONTROL, QDORA]
    assert offgrid == []
    loose_grid, loose_off = split_grid(
        [PREBASELINE, CONTROL, QDORA], reference=PREBASELINE, reference_in_grid=False
    )
    assert loose_grid == [CONTROL, QDORA]
    assert loose_off == [PREBASELINE]


def test_vocabulary_is_overridable_per_run():
    grid, offgrid = split_grid(
        [CONTROL, QDORA, "pilot"], grid_names=(CONTROL, QDORA), offgrid_names=("pilot",)
    )
    assert grid == [CONTROL, QDORA]
    assert offgrid == ["pilot"]
    assert _family(grid) == 1


# ---------------------------------------------------------------- block keys
def test_grid_and_offgrid_blocks_never_share_a_key():
    assert deltas_key() == DELTAS_KEY
    assert offgrid_deltas_key() == OFFGRID_DELTAS_KEY
    assert deltas_key() != offgrid_deltas_key()
    assert deltas_key(QDORA, follow_reference=True) == "deltas_vs_qdora"
    assert (
        offgrid_deltas_key(QDORA, follow_reference=True) == "offgrid_deltas_vs_qdora"
    )


def test_contrast_key_names_cell_then_reference():
    assert contrast_key(PREBASELINE, CONTROL) == "prebaseline_vs_control"
    assert contrast_key(QDORA) == "qdora_vs_control"


def test_readers_pick_up_their_own_block_only():
    report = {
        DELTAS_KEY: {"qdora_vs_control": {"faithful": {}}},
        OFFGRID_DELTAS_KEY: {"prebaseline_vs_control": {"faithful": {}}},
    }
    assert list(deltas_of(report)) == ["qdora_vs_control"]
    assert list(offgrid_deltas_of(report)) == ["prebaseline_vs_control"]


def test_missing_blocks_read_as_empty():
    assert deltas_of({}) == {}
    assert offgrid_deltas_of({}) == {}
    assert deltas_of({}, default={"x": 1}) == {"x": 1}


def test_legacy_and_reference_named_blocks_are_still_read():
    assert deltas_of({"deltas_vs_baseline": {"a": 1}}) == {"a": 1}
    assert offgrid_deltas_of({"offgrid_deltas_vs_control": {"b": 2}}) == {"b": 2}
