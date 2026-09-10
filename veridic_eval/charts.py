"""
Charts for the results view, built from a written report and nothing else.

`run` scores and writes `out/report.json`; every function here reads the rows
`cells_app` already tabulates from that file, so a chart can never disagree with
the table beside it or invent a metric the report does not carry. Altair ships
with Streamlit, and each builder returns a chart object rather than drawing, so
the same call serves the app, a notebook and a saved `.html`.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "sane_module",
    "numeric_value_keys",
    "levels_long",
    "delta_ci_chart",
    "levels_chart",
    "cost_chart",
]


def sane_module(
    name: str = "pandas",
    *,
    needs: Sequence[str] = ("Timestamp",),
    drop_partial: bool = True,
    reimport: bool = True,
) -> Optional[Any]:
    """The `sys.modules` entry when it is whole, else drop the half-built one.

    Altair reads ``sys.modules.get("pandas")`` and dereferences ``.Timestamp``
    on every ``to_dict``, so a half-built entry raises AttributeError inside a
    chart that never touched pandas. Absent is a state altair handles by
    skipping the branch; half-built is not, so it goes.

    Args:
        needs: attributes that prove the import finished.
        drop_partial: False reports only, leaving `sys.modules` untouched.
        reimport: retry the import once after dropping, to get the real module
            back; False leaves it absent, which the chart builders survive.

    Returns:
        The whole module, or None when it is absent, half-built or unimportable.
    """
    mod = sys.modules.get(name)
    if mod is None or all(hasattr(mod, a) for a in needs):
        return mod
    if not drop_partial:
        return None
    sys.modules.pop(name, None)
    if not reimport:
        return None
    try:
        import importlib

        fresh = importlib.import_module(name)
    except Exception:
        return None
    if all(hasattr(fresh, a) for a in needs):
        return fresh
    sys.modules.pop(name, None)
    return None


def _alt(on_missing: str = "none", *, sane: Sequence[str] = ("pandas",)):
    """Altair, or None when it is not installed.

    Args:
        sane: modules passed through `sane_module` first, because altair reads
            them out of `sys.modules` while it builds a chart.
    """
    try:
        import altair as alt
    except Exception:
        if on_missing == "raise":
            raise
        return None
    for name in sane:
        sane_module(name)
    return alt


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def numeric_value_keys(
    rows: Sequence[Dict[str, Any]],
    *,
    skip: Sequence[str] = ("cell", "linked", "judged", "contrast", "metric"),
    only: Optional[Sequence[str]] = None,
) -> List[str]:
    """Column names worth plotting: numeric in at least one row, in row order.

    Args:
        skip: identity and text columns that carry no magnitude.
        only: keep this exact list instead, in this order, for a fixed figure.
    """
    if only is not None:
        return list(only)
    out: List[str] = []
    for row in rows:
        for k, v in row.items():
            if k in skip or k in out:
                continue
            if _is_number(v):
                out.append(k)
    return out


def typed_fields(
    rows: Sequence[Dict[str, Any]],
    keys: Optional[Sequence[str]] = None,
    *,
    quantitative: str = "Q",
    nominal: str = "N",
    skip: Sequence[str] = (),
) -> List[str]:
    """``name:type`` shorthands, which inline data cannot get any other way.

    A chart fed `alt.Data(values=...)` is not a DataFrame, so altair infers no
    type and a bare field name raises instead of drawing. Numeric in any row
    reads quantitative, everything else nominal.

    Args:
        keys: fields to type, in this order; None takes every key the rows
            carry, first seen first. A name already carrying ``:`` is passed
            through, so a caller can pin one field to ``T`` or ``O``.
        quantitative / nominal: the two type letters, for a caller plotting
            against a scale that wants ``O`` where this picks ``N``.
        skip: fields left out, for a column that must not reach the tooltip.
    """
    names = list(keys) if keys is not None else list(
        dict.fromkeys(k for r in rows for k in r.keys())
    )
    out: List[str] = []
    for name in names:
        if name in skip:
            continue
        if ":" in name:
            out.append(name)
            continue
        numeric = any(_is_number(r.get(name)) for r in rows)
        out.append(f"{name}:{quantitative if numeric else nominal}")
    return out


def levels_long(
    rows: Sequence[Dict[str, Any]],
    *,
    cell_key: str = "cell",
    value_keys: Optional[Sequence[str]] = None,
    skip: Sequence[str] = ("cell", "linked", "judged"),
    metric_name: str = "metric",
    value_name: str = "value",
) -> List[Dict[str, Any]]:
    """Per-cell rows melted to ``{cell, metric, value}``, dropping non-numbers.

    A missing metric is left out rather than plotted as 0, so a cell that never
    reached a number cannot read as a floor.
    """
    keys = numeric_value_keys(rows, skip=skip, only=value_keys)
    long: List[Dict[str, Any]] = []
    for row in rows:
        for k in keys:
            v = row.get(k)
            if _is_number(v):
                long.append({cell_key: row.get(cell_key), metric_name: k, value_name: v})
    return long


def delta_ci_chart(
    rows: Sequence[Dict[str, Any]],
    *,
    value_key: str = "delta",
    group_key: str = "contrast",
    facet_key: str = "metric",
    lo_key: Optional[str] = "ci_low",
    hi_key: Optional[str] = "ci_high",
    flag_key: Optional[str] = "significant",
    flag_title: str = "significant",
    zero_rule: bool = True,
    columns: int = 2,
    width: int = 260,
    height: int = 22,
    title: str = "",
    on_missing: str = "none",
):
    """Delta per contrast, one panel per metric, CI as a rule through the bar.

    Args:
        value_key / group_key / facet_key: the delta, the contrast it belongs to,
            and the metric that gets its own panel.
        lo_key / hi_key: CI bounds; None draws bars with no interval, which is
            what an off-grid CFCA row carries.
        flag_key: colours the bar by significance. None leaves one colour, for a
            block that carries no correction and must not read as tested.
        zero_rule: draw the no-change line, the only reference a delta has.
        columns / width / height: panel grid and bar geometry.
        title: chart title, empty for none.
        on_missing: ``none`` returns None without altair, ``raise`` propagates.

    Returns:
        An altair chart, or None when there is nothing to draw.
    """
    alt = _alt(on_missing)
    if alt is None or not rows:
        return None
    data = [dict(r) for r in rows]
    tooltip_keys = [c for c in data[0].keys()]
    has_ci = bool(lo_key and hi_key) and any(
        _is_number(r.get(lo_key)) and _is_number(r.get(hi_key)) for r in data
    )
    # One top-level dataset for every layer: a facet slices the parent data, and a
    # layer that declares its own data is not sliced, so the zero line rides on a
    # constant field in the same rows instead of a second inline dataset.
    zero_field = "_zero"
    if zero_rule:
        for r in data:
            r[zero_field] = 0
    enc_y = alt.Y(f"{group_key}:N", title=None, sort=None)
    colour = (
        alt.Color(f"{flag_key}:N", title=flag_title)
        if flag_key and any(flag_key in r for r in data)
        else alt.value("#4c78a8")
    )
    bars = alt.Chart().mark_bar().encode(
        x=alt.X(f"{value_key}:Q", title=value_key),
        y=enc_y,
        color=colour,
        tooltip=tooltip_keys,
    )
    layers = [bars]
    if has_ci:
        layers.append(
            alt.Chart().mark_rule(size=2, color="black").encode(
                x=f"{lo_key}:Q", x2=f"{hi_key}:Q", y=enc_y,
            )
        )
    if zero_rule:
        layers.append(
            alt.Chart().mark_rule(
                strokeDash=[4, 4], color="grey"
            ).encode(x=f"{zero_field}:Q")
        )
    chart = alt.layer(*layers, data=alt.Data(values=data)).properties(
        width=width, height={"step": height}
    )
    chart = chart.facet(facet=alt.Facet(f"{facet_key}:N", title=None), columns=columns)
    return chart.properties(title=title) if title else chart


def levels_chart(
    rows: Sequence[Dict[str, Any]],
    *,
    cell_key: str = "cell",
    value_keys: Optional[Sequence[str]] = None,
    skip: Sequence[str] = ("cell", "linked", "judged"),
    columns: int = 3,
    width: int = 200,
    height: int = 22,
    title: str = "",
    on_missing: str = "none",
):
    """The raw per-cell level behind every delta, one panel per metric.

    Args:
        value_keys: fix the metrics and their order; None takes every numeric
            column the rows carry.
        skip / cell_key / columns / width / height / title / on_missing: as in
            `levels_long` and `delta_ci_chart`.
    """
    alt = _alt(on_missing)
    long = levels_long(rows, cell_key=cell_key, value_keys=value_keys, skip=skip)
    if alt is None or not long:
        return None
    chart = alt.Chart(alt.Data(values=long)).mark_bar().encode(
        x=alt.X("value:Q", title=None),
        y=alt.Y(f"{cell_key}:N", title=None, sort=None),
        color=alt.Color(f"{cell_key}:N", legend=None),
        tooltip=[f"{cell_key}:N", "metric:N", "value:Q"],
    ).properties(width=width, height={"step": height})
    chart = chart.facet(
        facet=alt.Facet("metric:N", title=None), columns=columns
    ).resolve_scale(x="independent")
    return chart.properties(title=title) if title else chart


def cost_chart(
    rows: Sequence[Dict[str, Any]],
    *,
    cell_key: str = "cell",
    value_key: str = "cfca",
    extra_keys: Sequence[str] = (),
    series_key: Optional[str] = None,
    labels: Optional[Mapping[str, str]] = None,
    value_title: str = "GBP per faithfully-cited answer",
    one_time_note: str = "one-time index cost is reported apart, never per answer",
    width: int = 520,
    height: int = 24,
    title: str = "",
    on_missing: str = "none",
):
    """CFCA per cell in GBP, recurring only.

    Args:
        value_key: the recurring per-answer number, `cells_table`'s ``cfca``.
        extra_keys: further numeric columns to draw beside it, each its own
            series, for a report that carries more than one cost column. Two
            costs of the same answer are drawn side by side and never stacked:
            their sum is not a quantity, and a stacked pair would print one.
        series_key: column naming each row's series, for a table that carries a
            sensitivity as extra rows instead of extra columns. Each row is then
            its own bar at its own place on the axis and the key only colours
            it, so a pair is read down the axis and can never be summed.
            Ignored when no row carries it, which is a report written before
            the sensitivity existed.
        labels: series name -> legend text, for a column or a series whose name
            is a field name; a name absent from it keeps its own.
        one_time_note: kept in the subtitle, because capex divided over answers
            is the one arithmetic this metric must never show.
    """
    alt = _alt(on_missing)
    if alt is None or not rows:
        return None
    names = dict(labels or {})
    tagged = bool(series_key) and any(r.get(series_key) for r in rows)
    if tagged:
        long = [
            {cell_key: r.get(cell_key),
             "series": names.get(str(r.get(series_key)), str(r.get(series_key))),
             "value": r.get(value_key)}
            for r in rows if _is_number(r.get(value_key))
        ]
        order = list(dict.fromkeys(item["series"] for item in long))
    else:
        keys = [value_key, *extra_keys]
        long = [
            {cell_key: r.get(cell_key), "series": names.get(k, k), "value": r.get(k)}
            for r in rows for k in keys if _is_number(r.get(k))
        ]
        order = [names.get(k, k) for k in keys]
    if not long:
        return None
    grouped = len(set(order)) > 1
    encoding: Dict[str, Any] = {
        "x": alt.X("value:Q", title=value_title, stack=None),
        "y": alt.Y(f"{cell_key}:N", title=None, sort=None),
        "color": (alt.Color("series:N", title=None, sort=order)
                  if grouped else alt.value("#54a24b")),
        "tooltip": [f"{cell_key}:N", "series:N", "value:Q"],
    }
    if grouped and not tagged:
        encoding["yOffset"] = alt.YOffset("series:N", sort=order)
    step = height if tagged else height * max(1, len(set(order)))
    chart = alt.Chart(alt.Data(values=long)).mark_bar().encode(**encoding).properties(
        width=width, height={"step": step}
    )
    return chart.properties(
        title={"text": title or "CFCA cost", "subtitle": one_time_note}
    )
