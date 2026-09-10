"""
conditions.yaml is now the single declaration of the experiment, so the parser is
tested on the things that would silently corrupt a result: an off-grid cell used
as the reference, a fallback reference picked from the off-grid window because it
happens to be listed first, a misspelled key that would widen a window to the
whole log, and a cost typo that would zero a term of the CFCA.
"""
from datetime import datetime, timezone

import pytest

from veridic_eval.cells import CHUNK_OPT, COMBINED, CONTROL, PREBASELINE, QDORA
from veridic_eval.conditions import (
    cfca_for_cells,
    load_conditions_file,
    parse_conditions_data,
    parse_cost_block,
    parse_timestamp,
    p_hat_from_report,
    render_cfca_markdown,
)

WINDOW = {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"}


def _file(**over):
    """The example file's shape: prebaseline first, then the four grid cells."""
    data = {
        "reference": CONTROL,
        "conditions": {
            PREBASELINE: {"start": "2026-06-30T00:00:00Z", "end": "2026-07-01T00:00:00Z"},
            CONTROL: dict(WINDOW),
            CHUNK_OPT: {"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
            QDORA: {"start": "2026-07-03T00:00:00Z", "end": "2026-07-04T00:00:00Z"},
            COMBINED: {"start": "2026-07-04T00:00:00Z", "end": "2026-07-05T00:00:00Z"},
        },
    }
    data.update(over)
    return data


def _parse(data, **kwargs):
    kwargs.setdefault("printer", None)
    return parse_conditions_data(data, **kwargs)


# ------------------------------------------------------------------- the split
def test_five_declared_windows_split_into_four_scored_and_one_measured():
    parsed = _parse(_file())
    assert parsed.names == [PREBASELINE, CONTROL, CHUNK_OPT, QDORA, COMBINED]
    assert parsed.grid == (CONTROL, CHUNK_OPT, QDORA, COMBINED)
    assert parsed.offgrid == (PREBASELINE,)
    assert parsed.reference == CONTROL


def test_reference_fallback_skips_the_offgrid_window_listed_first():
    data = _file()
    del data["reference"]
    assert _parse(data).reference == CONTROL


def test_reference_fallback_takes_the_first_grid_cell_when_control_is_absent():
    data = _file()
    del data["reference"]
    del data["conditions"][CONTROL]
    assert _parse(data).reference == CHUNK_OPT


def test_offgrid_reference_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="off-grid"):
        _parse(_file(reference=PREBASELINE))


def test_all_offgrid_file_has_nothing_to_score():
    with pytest.raises(ValueError, match="nothing to score"):
        _parse({"conditions": {PREBASELINE: dict(WINDOW)}, "offgrid": [PREBASELINE]})


def test_declared_grid_and_offgrid_lists_win_over_the_vocabulary():
    parsed = _parse(_file(grid=[CONTROL, "pilot"], offgrid=[PREBASELINE, QDORA],
                          conditions={CONTROL: dict(WINDOW), "pilot": dict(WINDOW),
                                      QDORA: dict(WINDOW), PREBASELINE: dict(WINDOW)}))
    assert parsed.grid == (CONTROL, "pilot")
    assert parsed.offgrid == (QDORA, PREBASELINE)


def test_a_name_in_both_lists_is_a_file_error():
    with pytest.raises(ValueError, match="both"):
        _parse(_file(grid=[CONTROL, PREBASELINE], offgrid=[PREBASELINE]))


def test_unknown_cell_is_scored_in_the_grid_unless_told_otherwise():
    data = _file()
    data["conditions"]["pilot"] = dict(WINDOW)
    assert "pilot" in _parse(data).grid
    assert "pilot" in _parse(data, unknown="offgrid").offgrid


# --------------------------------------------------------------- per-cell keys
def test_windows_conversation_ids_and_query_ids_all_reach_the_condition():
    parsed = _parse(_file(conditions={
        CONTROL: {"conversation_ids": ["abc", 123], "query_ids": ["q1", 2],
                  "per_answer_cost": "0.25"},
    }))
    cell = parsed.cell(CONTROL)
    assert cell.condition.conversation_ids == ["abc", "123"]
    assert cell.query_ids == ["q1", "2"]
    assert cell.per_answer_cost == 0.25
    assert parsed.per_answer_cost == {CONTROL: 0.25}
    assert parsed.query_ids == {CONTROL: ["q1", "2"]}


def test_missing_per_answer_cost_stays_absent_rather_than_becoming_zero():
    parsed = _parse(_file())
    assert parsed.per_answer_cost == {}
    assert parsed.cell(CONTROL).per_answer_cost is None


def test_misspelled_cell_key_raises_instead_of_widening_the_window():
    with pytest.raises(ValueError, match="unknown key"):
        _parse(_file(conditions={CONTROL: {"conversaton_ids": ["x"]}}))


def test_unbounded_cell_is_noted_and_can_be_refused():
    parsed = _parse(_file(conditions={CONTROL: {"per_answer_cost": 0.0}}))
    assert any("every logged answer" in note for note in parsed.notes)
    with pytest.raises(ValueError, match="every logged answer"):
        _parse(_file(conditions={CONTROL: {}}), on_unbounded="raise")


def test_snapshot_defaults_from_the_file_and_is_overridable_per_cell():
    parsed = _parse(_file())
    assert all(c.snapshot for c in parsed.cells)
    parsed = _parse(_file(snapshot=False))
    assert not any(c.snapshot for c in parsed.cells)
    data = _file(snapshot=False)
    data["conditions"][PREBASELINE]["snapshot"] = True
    parsed = _parse(data)
    assert parsed.snapshot_names() == [PREBASELINE]
    assert parsed.cell(CONTROL).snapshot is False


def test_snapshot_names_honours_only_and_skip():
    parsed = _parse(_file())
    assert parsed.snapshot_names(only=[CONTROL, QDORA]) == [CONTROL, QDORA]
    assert PREBASELINE not in parsed.snapshot_names(skip=[PREBASELINE])


# ------------------------------------------------------------------ cost block
def test_watts_and_kwh_become_the_gpu_hour_price():
    block = parse_cost_block({"watts": 140.0, "kwh_gbp": 0.26})
    assert block == {"p_gpu_hour": pytest.approx(0.0364)}


def test_declared_gpu_hour_price_wins_over_the_measured_pair():
    block = parse_cost_block({"watts": 140.0, "kwh_gbp": 0.26, "p_gpu_hour": 1.5})
    assert block["p_gpu_hour"] == 1.5
    loose = parse_cost_block({"watts": 140.0, "kwh_gbp": 0.26, "p_gpu_hour": 1.5},
                             declared_gpu_hour_wins=False)
    assert loose["p_gpu_hour"] == pytest.approx(0.0364)


def test_half_a_price_pair_is_an_error():
    with pytest.raises(ValueError, match="come as a pair"):
        parse_cost_block({"watts": 140.0})


def test_cost_defaults_are_merged_under_the_cell_and_lose_to_it():
    parsed = _parse(_file(cost_defaults={"A": 1000.0, "Q": 1000.0, "query_gpu_seconds": 1.0},
                          conditions={CONTROL: dict(WINDOW, cost={"query_gpu_seconds": 2.1})}))
    assert parsed.cell(CONTROL).cost == {"A": 1000.0, "Q": 1000.0, "query_gpu_seconds": 2.1}


def test_cost_defaults_alone_give_every_cell_a_block():
    parsed = _parse(_file(cost_defaults={"A": 10.0, "Q": 10.0}))
    assert set(parsed.cost_inputs) == set(parsed.names)


def test_no_cost_anywhere_means_no_cfca_rather_than_a_zero():
    parsed = _parse(_file())
    assert parsed.cost_inputs == {}
    assert parsed.cell(CONTROL).cost is None


def test_a_cost_typo_raises_instead_of_zeroing_a_term():
    with pytest.raises(ValueError, match="unknown cost key"):
        _parse(_file(conditions={CONTROL: dict(WINDOW, cost={"query_gpu_second": 2.1})}))


def test_denominators_must_be_positive_and_nothing_may_be_negative():
    with pytest.raises(ValueError, match="must be > 0"):
        parse_cost_block({"A": 0.0})
    with pytest.raises(ValueError, match="must be > 0"):
        parse_cost_block({"Q": -1.0})
    with pytest.raises(ValueError, match="negative"):
        parse_cost_block({"query_gpu_seconds": -2.0})


def test_non_numeric_cost_value_names_the_key():
    with pytest.raises(ValueError, match="index_gb"):
        parse_cost_block({"index_gb": "big"})


def test_required_cost_keys_can_be_demanded():
    with pytest.raises(ValueError, match="missing required"):
        parse_cost_block({"A": 1.0}, require=("Q",))


# ------------------------------------------------------------------ timestamps
def test_zulu_and_blank_bounds_parse_the_way_the_files_write_them():
    assert parse_timestamp("2026-07-01T00:00:00Z") == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("null") is None
    fixed = datetime(2026, 7, 1)
    assert parse_timestamp(fixed) is fixed
    assert parse_timestamp("2026-07-01T00:00:00", assume_tz=timezone.utc).tzinfo is timezone.utc


# ------------------------------------------------------------------- the CFCA
def test_cfca_is_cost_over_the_all_pass_rate_with_deltas_against_the_reference():
    inputs = {
        CONTROL: {"p_gpu_hour": 3.6, "query_gpu_seconds": 1.0, "A": 1.0, "Q": 1.0},
        QDORA: {"p_gpu_hour": 3.6, "query_gpu_seconds": 2.0, "A": 1.0, "Q": 1.0},
    }
    block = cfca_for_cells(inputs, {CONTROL: 0.5, QDORA: 0.5}, reference=CONTROL)
    assert block["cells"][CONTROL]["cost_per_answer"] == pytest.approx(0.001)
    assert block["cells"][CONTROL]["cfca"] == pytest.approx(0.002)
    assert block["cells"][QDORA]["cfca"] == pytest.approx(0.004)
    assert block["deltas"][QDORA]["cfca"] == pytest.approx(0.002)
    assert CONTROL not in block["deltas"]


def test_a_cell_without_a_measured_rate_keeps_its_cost_and_says_cfca_is_undefined():
    inputs = {CONTROL: {"query_gpu_seconds": 1.0}}
    block = cfca_for_cells(inputs, {CONTROL: None})
    assert block["cells"][CONTROL]["cfca"] is None
    assert "undefined" in block["cells"][CONTROL]["note"]
    assert cfca_for_cells(inputs, {CONTROL: None}, on_missing_p_hat="skip")["cells"] == {}
    with pytest.raises(ValueError, match="no P_hat"):
        cfca_for_cells(inputs, {CONTROL: None}, on_missing_p_hat="raise")


def test_zero_rate_is_undefined_too_and_never_divides():
    block = cfca_for_cells({CONTROL: {"query_gpu_seconds": 1.0}}, {CONTROL: 0.0})
    assert block["cells"][CONTROL]["cfca"] is None


def test_cells_without_cost_inputs_are_absent_from_the_block():
    block = cfca_for_cells({CONTROL: {"A": 2.0}}, {CONTROL: 1.0, QDORA: 1.0})
    assert list(block["cells"]) == [CONTROL]


def test_p_hat_comes_from_the_cfca_block_of_a_written_report():
    report = {"cells": {CONTROL: {"cfca": {"p_faithful_cited_version": 0.82}},
                        QDORA: {"cfca": {}}}}
    assert p_hat_from_report(report) == {CONTROL: 0.82, QDORA: None}


def test_markdown_says_so_when_no_cell_declared_cost_inputs():
    text = render_cfca_markdown({"cells": {}})
    assert "no CFCA" in text
    table = render_cfca_markdown(
        cfca_for_cells({CONTROL: {"query_gpu_seconds": 1.0}}, {CONTROL: 0.5}, reference=CONTROL)
    )
    assert f"`{CONTROL}`" in table and "CFCA" in table


# ------------------------------------------------------------------- the file
def test_a_written_yaml_round_trips_through_the_loader(tmp_path):
    path = tmp_path / "conditions.yaml"
    path.write_text(
        "reference: control\n"
        "offgrid: [prebaseline]\n"
        "cost_defaults: {watts: 140.0, kwh_gbp: 0.26, A: 10.0, Q: 10.0}\n"
        "conditions:\n"
        "  prebaseline:\n"
        "    conversation_ids: [\"c0\"]\n"
        "    per_answer_cost: 0.0\n"
        "  control:\n"
        "    conversation_ids: [\"c1\"]\n"
        "    per_answer_cost: 0.0\n"
        "    cost: {onetime_gpu_hours: 2.0}\n",
        encoding="utf-8",
    )
    parsed = load_conditions_file(str(path), printer=None)
    assert parsed.path == str(path)
    assert parsed.grid == (CONTROL,) and parsed.offgrid == (PREBASELINE,)
    assert parsed.cell(CONTROL).cost["onetime_gpu_hours"] == 2.0
    assert parsed.cell(PREBASELINE).cost["p_gpu_hour"] == pytest.approx(0.0364)


def test_an_empty_conditions_block_is_refused(tmp_path):
    path = tmp_path / "conditions.yaml"
    path.write_text("reference: control\nconditions: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No conditions found"):
        load_conditions_file(str(path), printer=None)
