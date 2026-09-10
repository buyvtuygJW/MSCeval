"""
Power measurement: one log for the whole experiment, sliced per cell after.

Questions are asked by hand in the app, so a cell is measured after the fact.
`measure_cell_power` returns two numbers: `window` covers the sitting start to
end, `busy` integrates only each answer's serving interval. `cost_inputs` uses
busy, because `query_gpu_seconds` in the CFCA is per-answer serving time.

`measure_cell_serving_power` is the front door: it prices the whole serving
stretch (retrieval + rerank + generation) and reports the generation-only number
beside it, and it prices a machine with no power sampler from one declared
constant. The one-time ingest cost is `ingest_cost`, not this module.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .devices import resolve_device, sample_power_w

CSV_HEADER: Tuple[str, ...] = ("iso_ts", "watts", "device", "tag")
BASES: Tuple[str, ...] = ("busy", "window")

#: Latency fields that together make one answer's serving stretch, in the order
#: the pipeline runs them: retrieve, rerank, generate. `extract` logs the first
#: two from ``rag_logs`` and the third from ``completion_logs``, so a cell that
#: prices generation alone bills none of its retrieval or reranking.
SERVING_LATENCY_FIELDS: Tuple[str, ...] = (
    "retrieval_latency_ms", "rerank_latency_ms", "latency_ms",
)
GENERATION_LATENCY_FIELDS: Tuple[str, ...] = ("latency_ms",)
#: Named field sums `measure_cell_serving_power` reports side by side.
SERVING_FIELD_SETS: Dict[str, Tuple[str, ...]] = {
    "serving": SERVING_LATENCY_FIELDS,
    "generation": GENERATION_LATENCY_FIELDS,
}


@dataclass(frozen=True)
class Interval:
    """One stretch of wall clock, half-open ``[start, end)``."""

    start: datetime
    end: datetime
    label: str = ""

    @property
    def seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())


# --------------------------------------------------------------------------
# 1. log
# --------------------------------------------------------------------------

def log_power(
    device: Any,
    *,
    out_path: str = "out/power/power.csv",
    tag: str = "",
    interval_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    append: bool = True,
    flush_every: int = 1,
    require_all_channels: bool = True,
    on_sample: Optional[Callable[[datetime, float], None]] = None,
    stop_when: Optional[Callable[[], bool]] = None,
    runner: Optional[Callable[..., Any]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    make_dirs: bool = True,
    printer: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """
    Poll one device until stopped, appending to the log every cell is cut from.

    Args:
        tag: written on every row, to split one file by machine restart or stage.
        interval_s: cadence; None uses the device's own.
        duration_s: stop after N seconds; None runs until Ctrl+C or `stop_when`.
        append: False truncates, discarding every cell already measured.
        flush_every: rows between flushes; 1 survives a hard kill.
        require_all_channels: a missing channel raises instead of logging a
            part sum.
        on_sample / stop_when / runner / sleeper / clock: injection points.

    Returns:
        ``{path, device, rows, valid_rows, first_ts, last_ts, elapsed_s}``.
    """
    profile = resolve_device(device, where="log_power(device=)")
    cadence = profile.interval_s if interval_s is None else float(interval_s)
    if cadence <= 0:
        raise ValueError(f"interval_s must be > 0, got {cadence}")
    if make_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fresh = (not append) or not os.path.exists(out_path)
    rows = valid = 0
    first_ts = last_ts = None
    t0 = time.monotonic()
    stopping = {"now": False}

    def _stop(_sig, _frm):
        stopping["now"] = True

    try:
        previous = signal.signal(signal.SIGINT, _stop)
    except (ValueError, AttributeError, OSError):
        previous = None
    if printer:
        printer(f"power: {profile.label or profile.name} via "
                f"{' '.join(profile.command()) or profile.sampler} every {cadence}s "
                f"-> {out_path} (Ctrl+C to stop)")
    try:
        with open(out_path, "w" if fresh else "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if fresh:
                writer.writerow(CSV_HEADER)
            while not stopping["now"]:
                watts = sample_power_w(profile, runner=runner,
                                       require_all_channels=require_all_channels)
                ts = clock()
                writer.writerow([ts.isoformat(),
                                 "" if math.isnan(watts) else round(watts, 3),
                                 profile.name, tag])
                rows += 1
                valid += 0 if math.isnan(watts) else 1
                first_ts = first_ts or ts
                last_ts = ts
                if flush_every and rows % flush_every == 0:
                    fh.flush()
                if on_sample:
                    on_sample(ts, watts)
                if stop_when and stop_when():
                    break
                if duration_s is not None and time.monotonic() - t0 >= duration_s:
                    break
                sleeper(cadence)
    except KeyboardInterrupt:
        pass
    finally:
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except Exception:
                pass
    out = {
        "path": os.path.abspath(out_path),
        "device": profile.as_dict(),
        "tag": tag,
        "rows": rows,
        "valid_rows": valid,
        "first_ts": first_ts.isoformat() if first_ts else None,
        "last_ts": last_ts.isoformat() if last_ts else None,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    if printer:
        printer(f"power: {rows} rows ({valid} valid) -> {out['path']}")
    return out


# --------------------------------------------------------------------------
# 2. slice
# --------------------------------------------------------------------------

def read_power_csv(
    path: str,
    *,
    tag: Optional[str] = None,
    device: Optional[str] = None,
    drop_invalid: bool = True,
    assume_tz: Optional[timezone] = timezone.utc,
    sort: bool = True,
) -> List[Tuple[datetime, float]]:
    """
    Load a power log as (timestamp, watts).

    Args:
        tag / device: keep only rows from one tag or machine.
        drop_invalid: drop failed samples. Off keeps them as NaN.
    """
    out: List[Tuple[datetime, float]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if tag is not None and (row.get("tag") or "") != tag:
                continue
            if device is not None and (row.get("device") or "") != device:
                continue
            raw = (row.get("watts") or "").strip()
            try:
                watts = float(raw) if raw else float("nan")
            except ValueError:
                watts = float("nan")
            if drop_invalid and math.isnan(watts):
                continue
            try:
                ts = datetime.fromisoformat((row.get("iso_ts") or "").replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None and assume_tz is not None:
                ts = ts.replace(tzinfo=assume_tz)
            out.append((ts, watts))
    return sorted(out, key=lambda r: r[0]) if sort else out


def require_power_log(
    path: str,
    *,
    device: Any = None,
    tag: Optional[str] = None,
    missing: str = "raise",
    empty_is_missing: bool = False,
    meter_command: Optional[str] = None,
    where: str = "power pricing",
    reader: Optional[Callable[..., Sequence[Tuple[datetime, float]]]] = None,
    **read_kwargs: Any,
) -> List[Tuple[datetime, float]]:
    """
    Load a power log, naming the meter run that should have written it.

    `read_power_csv` opens the path directly, so a cell priced before the meter
    ever ran fails inside a csv reader, with neither the device nor the command
    that fixes it. This owns that message.

    Args:
        device: profile, name or mapping; names the device and its meter command.
        tag: forwarded to the reader, to slice one stage out of a shared log.
        missing: an absent log ``raise``s, or returns no rows (``ignore``), which
            leaves coverage at 0 and prices the cell off the nameplate.
        empty_is_missing: True also raises when the log exists but holds no row
            for this tag and device, so a truncated file is caught here.
        meter_command: exact command quoted in the message; None builds it from
            the device name.
        where: caller named in the message.
        reader: injected loader, for testing without a file.
        **read_kwargs: passed to the reader, e.g. ``device=``, ``drop_invalid=``.

    Returns:
        The rows, or ``[]`` when the log is absent and `missing` is ``ignore``.
    """
    if missing not in ("raise", "ignore"):
        raise ValueError(f"missing must be 'raise' or 'ignore', got {missing!r}")
    profile = resolve_device(device, where=f"{where}(device=)", required=False)
    name = profile.name if profile is not None else str(device or "device")
    full = os.path.abspath(path)
    absent = not os.path.isfile(full)
    load = reader or read_power_csv
    rows = [] if absent else list(load(path, tag=tag, **read_kwargs))
    if absent or (empty_is_missing and not rows):
        if missing == "raise":
            cmd = meter_command or f"veridic-eval meter --device {name}"
            what = ("no power log at" if absent
                    else f"no rows for tag {tag!r} in the power log at")
            raise ValueError(
                f"device {name!r}: {what} {full}; run `{cmd}` first, hand the "
                f"rows in as samples=, or set missing='ignore' to price the "
                f"cell off the {name} nameplate")
        return []
    return rows


def integrate_power(
    samples: Sequence[Tuple[datetime, float]],
    start: datetime,
    end: datetime,
    *,
    max_gap_s: float = 15.0,
    min_samples: int = 2,
    edge: str = "hold",
    idle_w: float = 0.0,
    clamp_negative: bool = True,
) -> Dict[str, Any]:
    """
    Trapezoidal energy over one window, holes counted rather than filled.

    Args:
        max_gap_s: samples further apart than this are a hole. Its seconds leave
            both the energy and the covered time, so a logger that died lowers
            coverage instead of inventing watts.
        min_samples: fewer than this inside the window returns unmeasured.
        edge: ``hold`` extends the neighbouring samples to the window edges;
            ``trim`` measures between the first and last sample inside it.
        idle_w: subtracted from every sample, for serving-only power.
        clamp_negative: a sample under `idle_w` counts as 0.

    Returns:
        ``{measured, seconds, covered_s, coverage, energy_wh, mean_w, peak_w,
        min_w, samples, gaps, gap_s}``. ``mean_w`` divides by covered time.
    """
    if edge not in ("hold", "trim"):
        raise ValueError(f"edge must be 'hold' or 'trim', got {edge!r}")
    span = max(0.0, (end - start).total_seconds())
    inside = [(t, w) for t, w in samples if start <= t <= end and not math.isnan(w)]
    empty = {
        "measured": False, "seconds": round(span, 3), "covered_s": 0.0,
        "coverage": 0.0, "energy_wh": None, "mean_w": None, "peak_w": None,
        "min_w": None, "samples": len(inside), "gaps": 0, "gap_s": 0.0,
    }
    if len(inside) < max(1, min_samples) or span <= 0:
        return empty

    def _adj(w: float) -> float:
        v = w - idle_w
        return max(0.0, v) if clamp_negative else v

    points: List[Tuple[datetime, float]] = list(inside)
    if edge == "hold":
        before = [(t, w) for t, w in samples if t < start and not math.isnan(w)]
        after = [(t, w) for t, w in samples if t > end and not math.isnan(w)]
        if before and (start - before[-1][0]).total_seconds() <= max_gap_s:
            points.insert(0, (start, before[-1][1]))
        if after and (after[0][0] - end).total_seconds() <= max_gap_s:
            points.append((end, after[0][1]))

    joules = covered = gap_s = 0.0
    gaps = 0
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        if dt > max_gap_s:
            gaps += 1
            gap_s += dt
            continue
        joules += 0.5 * (_adj(w0) + _adj(w1)) * dt
        covered += dt
    if covered <= 0:
        return empty
    watts = [_adj(w) for _, w in inside]
    return {
        "measured": True,
        "seconds": round(span, 3),
        "covered_s": round(covered, 3),
        "coverage": round(covered / span, 4) if span else 0.0,
        "energy_wh": round(joules / 3600.0, 6),
        "mean_w": round(joules / covered, 3),
        "peak_w": round(max(watts), 3),
        "min_w": round(min(watts), 3),
        "samples": len(inside),
        "gaps": gaps,
        "gap_s": round(gap_s, 3),
    }


def integrate_window(
    samples: Sequence[Tuple[datetime, float]],
    start: datetime,
    end: datetime,
    *,
    max_gap_s: float = 15.0,
    min_samples: int = 1,
    min_inside: int = 0,
    edge: str = "hold",
    bracket_before: bool = True,
    bracket_after: bool = True,
    max_bracket_gap_s: Optional[float] = None,
    require_both_edges: bool = True,
    stats_from: str = "inside",
    idle_w: float = 0.0,
    clamp_negative: bool = True,
) -> Dict[str, Any]:
    """
    Trapezoidal energy over one window, bracketed before it is judged.

    `integrate_power` counts the samples inside the window first and returns
    unmeasured before ``edge='hold'`` reaches the neighbours, so a window
    shorter than the logger's cadence goes unmeasured even with a sample
    seconds either side of it. This one brackets first and gates on the points
    it will integrate, which is what prices a 4 s answer off a 6 s log.

    Args:
        max_gap_s: consecutive points further apart than this are a hole. Its
            seconds leave both the energy and the covered time, so a logger
            that died lowers coverage instead of inventing watts.
        min_samples: fewer points than this, held edges included, is unmeasured.
        min_inside: samples demanded strictly inside the window. 0 lets the held
            edges alone carry a window shorter than the cadence; 1 or more
            demands the meter fired during the window itself.
        edge: ``hold`` extends the neighbouring samples to the window edges,
            ``trim`` measures between the first and last sample inside it,
            ``none`` is ``trim`` under a name that says no edge is invented.
        bracket_before / bracket_after: hold that one side, when edge is ``hold``.
        max_bracket_gap_s: how stale a neighbour may be and still reach an edge.
            None takes `max_gap_s`.
        require_both_edges: with nothing inside, one held edge is a single point
            that spans nothing, so both sides must exist. False keeps the
            one-sided case, which integrates to zero and returns unmeasured.
        stats_from: ``inside`` reports peak_w / min_w over the samples the meter
            really took inside the window, falling back to the points when it
            took none; ``points`` reports over the held edges as well.
        idle_w: subtracted from every sample, for serving-only power.
        clamp_negative: a sample under `idle_w` counts as 0.

    Returns:
        `integrate_power`'s keys, where ``samples`` stays the count inside, plus
        ``points`` (what was integrated), ``edges`` (how many were held) and
        ``bracket_only`` (True when no sample fell inside the window).
    """
    if edge not in ("hold", "trim", "none"):
        raise ValueError(f"edge must be 'hold', 'trim' or 'none', got {edge!r}")
    if stats_from not in ("inside", "points"):
        raise ValueError(f"stats_from must be 'inside' or 'points', got {stats_from!r}")
    bracket_gap = max_gap_s if max_bracket_gap_s is None else float(max_bracket_gap_s)
    span = max(0.0, (end - start).total_seconds())
    clean = sorted(((t, w) for t, w in samples if not math.isnan(w)),
                   key=lambda row: row[0])
    inside = [(t, w) for t, w in clean if start <= t <= end]
    points: List[Tuple[datetime, float]] = list(inside)
    edges = 0
    if edge == "hold":
        before = [(t, w) for t, w in clean if t < start]
        after = [(t, w) for t, w in clean if t > end]
        if (bracket_before and before
                and (start - before[-1][0]).total_seconds() <= bracket_gap):
            points.insert(0, (start, before[-1][1]))
            edges += 1
        if (bracket_after and after
                and (after[0][0] - end).total_seconds() <= bracket_gap):
            points.append((end, after[0][1]))
            edges += 1

    empty = {
        "measured": False, "seconds": round(span, 3), "covered_s": 0.0,
        "coverage": 0.0, "energy_wh": None, "mean_w": None, "peak_w": None,
        "min_w": None, "samples": len(inside), "points": len(points),
        "edges": edges, "bracket_only": False, "gaps": 0, "gap_s": 0.0,
    }
    if span <= 0 or len(inside) < min_inside or len(points) < max(1, min_samples):
        return empty
    if not inside and require_both_edges and edges < 2:
        return empty

    def _adj(w: float) -> float:
        v = w - idle_w
        return max(0.0, v) if clamp_negative else v

    joules = covered = gap_s = 0.0
    gaps = 0
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        if dt > max_gap_s:
            gaps += 1
            gap_s += dt
            continue
        joules += 0.5 * (_adj(w0) + _adj(w1)) * dt
        covered += dt
    if covered <= 0:
        return empty
    stat_points = inside if (stats_from == "inside" and inside) else points
    watts = [_adj(w) for _, w in stat_points]
    return {
        "measured": True,
        "seconds": round(span, 3),
        "covered_s": round(covered, 3),
        "coverage": round(covered / span, 4) if span else 0.0,
        "energy_wh": round(joules / 3600.0, 6),
        "mean_w": round(joules / covered, 3),
        "peak_w": round(max(watts), 3),
        "min_w": round(min(watts), 3),
        "samples": len(inside),
        "points": len(points),
        "edges": edges,
        "bracket_only": not inside,
        "gaps": gaps,
        "gap_s": round(gap_s, 3),
    }


def answer_intervals(
    records: Sequence[Any],
    *,
    answered_at_field: str = "answered_at",
    latency_field: str = "latency_ms",
    min_latency_ms: float = 1.0,
    default_latency_ms: Optional[float] = None,
    label_field: str = "query_id",
    assume_tz: Optional[timezone] = timezone.utc,
) -> List[Interval]:
    """
    Each answer's serving stretch, back from its completion timestamp.

    Args:
        default_latency_ms: duration for a record whose latency was never
            logged. None skips it rather than invent one.
    """
    out: List[Interval] = []
    for rec in records:
        raw_ts = getattr(rec, answered_at_field, None)
        lat = getattr(rec, latency_field, None)
        if lat is None:
            lat = default_latency_ms
        if raw_ts is None or lat is None or float(lat) < min_latency_ms:
            continue
        ts = raw_ts
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        out.append(Interval(ts - timedelta(milliseconds=float(lat)), ts,
                            str(getattr(rec, label_field, "") or "")))
    return sorted(out, key=lambda iv: iv.start)


def serving_intervals(
    records: Sequence[Any],
    *,
    latency_fields: Sequence[str] = SERVING_LATENCY_FIELDS,
    answered_at_field: str = "answered_at",
    require_fields: Sequence[str] = (),
    missing: str = "zero",
    min_latency_ms: float = 1.0,
    max_latency_ms: Optional[float] = None,
    on_overrun: str = "clamp",
    default_latency_ms: Optional[float] = None,
    scale_ms: float = 1.0,
    label_field: str = "query_id",
    assume_tz: Optional[timezone] = timezone.utc,
) -> List[Interval]:
    """
    Each answer's stretch back from its completion stamp, summing named latencies.

    `answer_intervals` prices one field. This one prices a sum, so retrieval and
    reranking are billed alongside generation instead of falling outside the
    interval. The stretch still ends at ``answered_at``: the fields are stages of
    the same answer, so their sum is the wall clock behind that stamp.

    Args:
        latency_fields: fields summed into one duration, in pipeline order.
        require_fields: fields that must be present and non-None on every
            record whatever ``missing`` says; a gap raises.
        missing: an absent or None field contributes ``zero``, drops the record
            (``skip``), or raises.
        min_latency_ms: a sum below this is not a served answer; skipped.
        max_latency_ms: sanity ceiling for a stuck stage; None keeps any sum.
        on_overrun: past the ceiling, ``clamp`` to it, ``skip`` the record,
            ``raise``, or ``keep`` the sum as logged.
        default_latency_ms: duration when no field on the record carries one.
            None skips that record rather than invent a stretch.
        scale_ms: multiplier onto every field, for a log in another unit.
        label_field: interval label, for the per-answer table.
        assume_tz: timezone for a naive stamp; None leaves it naive.

    Returns:
        Intervals sorted by start, one per priced answer.
    """
    if missing not in ("zero", "skip", "raise"):
        raise ValueError("missing must be 'zero', 'skip' or 'raise'")
    if on_overrun not in ("clamp", "skip", "raise", "keep"):
        raise ValueError("on_overrun must be 'clamp', 'skip', 'raise' or 'keep'")
    fields = tuple(latency_fields)
    if not fields:
        raise ValueError("latency_fields must name at least one field")
    unknown = [n for n in require_fields if n not in fields]
    if unknown:
        raise ValueError(f"require_fields not in latency_fields: {sorted(unknown)}")

    out: List[Interval] = []
    for rec in records:
        label = str(getattr(rec, label_field, "") or "")
        total = 0.0
        seen = 0
        drop = False
        for name in fields:
            raw = getattr(rec, name, None)
            if raw is None:
                if name in require_fields:
                    raise ValueError(f"record {label or '?'}: {name} required and absent")
                if missing == "raise":
                    raise ValueError(f"record {label or '?'}: {name} absent")
                if missing == "skip":
                    drop = True
                    break
                continue
            try:
                total += float(raw) * scale_ms
            except (TypeError, ValueError):
                raise ValueError(f"record {label or '?'}: {name}={raw!r} is not a number")
            seen += 1
        if drop:
            continue
        if seen == 0:
            if default_latency_ms is None:
                continue
            total = float(default_latency_ms)
        if total < min_latency_ms:
            continue
        if max_latency_ms is not None and total > float(max_latency_ms):
            if on_overrun == "raise":
                raise ValueError(f"record {label or '?'}: latency sum {total} ms "
                                 f"over max_latency_ms {max_latency_ms}")
            if on_overrun == "skip":
                continue
            if on_overrun == "clamp":
                total = float(max_latency_ms)

        raw_ts = getattr(rec, answered_at_field, None)
        if raw_ts is None:
            continue
        ts = raw_ts
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        out.append(Interval(ts - timedelta(milliseconds=total), ts, label))
    return sorted(out, key=lambda iv: iv.start)


def constant_samples(
    start: datetime,
    end: datetime,
    watts: float,
    *,
    interval_s: float = 1.0,
    pad_s: float = 0.0,
    min_rows: int = 2,
    max_rows: int = 2_000_000,
) -> List[Tuple[datetime, float]]:
    """
    A declared watt figure rendered as log rows, so a window can be integrated.

    For a machine with no per-channel sampler: the number is one cited constant
    (wall meter reading, or CPU package plus GPU board nameplate), and every
    sample carries it. Coverage comes out at 1.0 by construction, which is why
    the caller must label the result as declared rather than sampled.

    Args:
        start / end: window to cover; ``end`` before ``start`` raises.
        watts: the constant; must be > 0.
        interval_s: spacing between rows.
        pad_s: extra stretch on both ends, to cover a padded window.
        min_rows: rows for a window shorter than one interval.
        max_rows: ceiling that turns a silly interval into an error instead of
            an out-of-memory kill.
    """
    if end < start:
        raise ValueError(f"constant_samples: end {end.isoformat()} before start {start.isoformat()}")
    if not watts > 0:
        raise ValueError(f"constant_samples: watts must be > 0, got {watts!r}")
    if interval_s <= 0:
        raise ValueError(f"constant_samples: interval_s must be > 0, got {interval_s!r}")
    if min_rows < 2:
        raise ValueError("constant_samples: min_rows must be >= 2")
    first = start - timedelta(seconds=pad_s)
    last = end + timedelta(seconds=pad_s)
    span = (last - first).total_seconds()
    n = max(min_rows, int(math.floor(span / interval_s)) + 1)
    if n > max_rows:
        raise ValueError(f"constant_samples: {n} rows over max_rows {max_rows}; "
                         f"raise interval_s ({interval_s}s over {round(span, 1)}s)")
    step = span / (n - 1) if span > 0 else 0.0
    w = float(watts)
    return [(first + timedelta(seconds=step * i), w) for i in range(n)]


def integrate_intervals(
    samples: Sequence[Tuple[datetime, float]],
    intervals: Sequence[Interval],
    *,
    merge_overlaps: bool = True,
    per_interval: bool = False,
    integrator: Callable[..., Dict[str, Any]] = integrate_window,
    **integrate_kwargs: Any,
) -> Dict[str, Any]:
    """
    Sum one-window integration over many stretches.

    Args:
        merge_overlaps: concurrent answers share seconds; merging counts them once.
        per_interval: also return the per-answer rows.
        integrator: `integrate_window` by default, so an answer shorter than the
            log's cadence is bracketed instead of dropped. `integrate_power`
            restores the interior-only gate, and then only its own kwargs pass.
    """
    spans = sorted(intervals, key=lambda iv: iv.start)
    if merge_overlaps and spans:
        merged: List[Interval] = [spans[0]]
        for iv in spans[1:]:
            last = merged[-1]
            if iv.start <= last.end:
                merged[-1] = Interval(last.start, max(last.end, iv.end), last.label)
            else:
                merged.append(iv)
        spans = merged
    joules = covered = span_s = gap_s = 0.0
    gaps = n_measured = bracket_only = inside_samples = 0
    peaks: List[float] = []
    rows: List[Dict[str, Any]] = []
    for iv in spans:
        got = integrator(samples, iv.start, iv.end, **integrate_kwargs)
        span_s += iv.seconds
        if per_interval:
            rows.append({"label": iv.label, "start": iv.start.isoformat(),
                         "end": iv.end.isoformat(), **got})
        if not got["measured"]:
            continue
        n_measured += 1
        joules += got["energy_wh"] * 3600.0
        covered += got["covered_s"]
        gap_s += got["gap_s"]
        gaps += got["gaps"]
        peaks.append(got["peak_w"])
        inside_samples += int(got.get("samples") or 0)
        bracket_only += 1 if got.get("bracket_only") else 0
    out = {
        "measured": covered > 0,
        "intervals": len(spans),
        "measured_intervals": n_measured,
        "bracket_only_intervals": bracket_only,
        "inside_samples": inside_samples,
        "seconds": round(span_s, 3),
        "covered_s": round(covered, 3),
        "coverage": round(covered / span_s, 4) if span_s else 0.0,
        "energy_wh": round(joules / 3600.0, 6) if covered else None,
        "mean_w": round(joules / covered, 3) if covered else None,
        "peak_w": round(max(peaks), 3) if peaks else None,
        "gaps": gaps,
        "gap_s": round(gap_s, 3),
    }
    if per_interval:
        out["per_interval"] = rows
    return out


# --------------------------------------------------------------------------
# 3. price
# --------------------------------------------------------------------------

def measure_cell_power(
    records: Sequence[Any],
    *,
    device: Any,
    log_path: str = "out/power/power.csv",
    samples: Optional[Sequence[Tuple[datetime, float]]] = None,
    missing_log: str = "raise",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    tag: Optional[str] = None,
    basis: str = "busy",
    subtract_idle: bool = False,
    idle_w: Optional[float] = None,
    max_gap_s: float = 15.0,
    min_samples: int = 2,
    min_coverage: float = 0.8,
    edge: str = "hold",
    min_inside: int = 0,
    bracket_before: bool = True,
    bracket_after: bool = True,
    max_bracket_gap_s: Optional[float] = None,
    require_both_edges: bool = True,
    stats_from: str = "inside",
    integrator: Callable[..., Dict[str, Any]] = integrate_window,
    pad_start_s: float = 0.0,
    pad_end_s: float = 0.0,
    merge_overlaps: bool = True,
    per_interval: bool = False,
    default_latency_ms: Optional[float] = None,
    intervals_fn: Optional[Callable[..., Sequence[Interval]]] = None,
    intervals_kwargs: Optional[Dict[str, Any]] = None,
    fallback_tdp: bool = True,
    on_uncovered: str = "warn",
    printer: Optional[Callable[[str], None]] = print,
    cell: str = "",
) -> Dict[str, Any]:
    """
    One cell's measured electricity, start to end and serving-only.

    Args:
        records: the cell's `QueryRecord`s, carrying `answered_at`, `latency_ms`.
        samples: pre-loaded rows, to score many cells off one read of the log.
        missing_log: no log on disk ``raise``s with the meter command, or scores
            the cell on no rows (``ignore``), which prices it off the nameplate.
        start / end: None derives the window from the first and last answer.
        basis: which numbers `cost_inputs` carries, ``busy`` or ``window``.
        subtract_idle / idle_w: price above the idle floor; None takes the
            device's own.
        min_coverage: fraction of the window samples must cover to count.
        min_inside / bracket_before / bracket_after / max_bracket_gap_s /
            require_both_edges / stats_from: `integrate_window`'s bracketing, so
            an answer shorter than the log's cadence is priced from the samples
            either side of it instead of counted as a hole.
        integrator: `integrate_power` restores the interior-only gate, which
            drops every answer the meter never fired inside.
        pad_start_s / pad_end_s: widen the window for a logger clock that drifts.
        intervals_fn: builds the priced stretches; None uses `answer_intervals`
            on ``latency_ms`` alone. `serving_intervals` sums retrieval and
            reranking in as well.
        intervals_kwargs: passed to whichever builder runs. ``default_latency_ms``
            is forwarded to the default builder only, so a custom builder takes
            it here instead.
        fallback_tdp: on failed coverage, fall back to the nameplate and label
            the result `upper_bound` rather than return nothing.
        on_uncovered: ``warn`` | ``raise`` | ``ignore``.

    Returns:
        ``{cell, device, basis, window, busy, cost_inputs, quality}``, where
        `cost_inputs` is ``{watts, query_gpu_seconds, source, upper_bound}``.
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {list(BASES)}, got {basis!r}")
    if on_uncovered not in ("warn", "raise", "ignore"):
        raise ValueError("on_uncovered must be 'warn', 'raise' or 'ignore'")
    profile = resolve_device(device, where="measure_cell_power(device=)")
    floor = (profile.idle_w if idle_w is None else float(idle_w)) if subtract_idle else 0.0
    rows = (list(samples) if samples is not None else
            require_power_log(log_path, device=profile, tag=tag,
                              missing=missing_log, where="measure_cell_power"))

    if intervals_fn is None:
        spans = answer_intervals(records, default_latency_ms=default_latency_ms,
                                 **(intervals_kwargs or {}))
    else:
        spans = list(intervals_fn(records, **(intervals_kwargs or {})))
    if start is None:
        start = min((iv.start for iv in spans), default=None)
    if end is None:
        end = max((iv.end for iv in spans), default=None)
    n_answers = len(spans)
    common = dict(max_gap_s=max_gap_s, min_samples=min_samples, edge=edge, idle_w=floor)
    # Only `integrate_window` takes the bracketing knobs, so an `integrator` that
    # predates them (`integrate_power`) keeps working instead of raising TypeError.
    accepted = inspect.signature(integrator).parameters
    common.update({k: v for k, v in (
        ("min_inside", min_inside),
        ("bracket_before", bracket_before),
        ("bracket_after", bracket_after),
        ("max_bracket_gap_s", max_bracket_gap_s),
        ("require_both_edges", require_both_edges),
        ("stats_from", stats_from),
    ) if k in accepted})

    if start is None or end is None:
        window = {"measured": False,
                  "reason": "no answer carries answered_at + a logged latency"}
        busy = dict(window)
    else:
        window = integrator(rows, start - timedelta(seconds=pad_start_s),
                            end + timedelta(seconds=pad_end_s), **common)
        window["start"] = (start - timedelta(seconds=pad_start_s)).isoformat()
        window["end"] = (end + timedelta(seconds=pad_end_s)).isoformat()
        busy = integrate_intervals(rows, spans, merge_overlaps=merge_overlaps,
                                   per_interval=per_interval,
                                   integrator=integrator, **common)

    chosen = busy if basis == "busy" else window
    ok = bool(chosen.get("measured")) and chosen.get("coverage", 0.0) >= min_coverage
    per_answer_s = (chosen.get("seconds") or 0.0) / n_answers if n_answers else None

    if ok:
        cost_inputs = {
            "watts": chosen["mean_w"],
            "query_gpu_seconds": round(per_answer_s, 4) if per_answer_s else None,
            "source": f"measured ({basis} basis, {profile.name}, "
                      f"coverage {chosen.get('coverage')})",
            "upper_bound": False,
        }
    elif fallback_tdp and profile.tdp_w is not None:
        cost_inputs = {
            "watts": float(profile.tdp_w),
            "query_gpu_seconds": round(per_answer_s, 4) if per_answer_s else None,
            "source": f"upper bound: {profile.name} nameplate {profile.tdp_w} W, "
                      f"coverage {chosen.get('coverage', 0.0)} < {min_coverage}",
            "upper_bound": True,
        }
    else:
        cost_inputs = {
            "watts": None,
            "query_gpu_seconds": round(per_answer_s, 4) if per_answer_s else None,
            "source": "unmeasured: no coverage and no device tdp_w",
            "upper_bound": False,
        }

    if not ok and on_uncovered != "ignore":
        msg = (f"power: cell {cell or '?'} {basis} coverage "
               f"{chosen.get('coverage', 0.0)} < {min_coverage} over {n_answers} answers; "
               f"{cost_inputs['source']}")
        if on_uncovered == "raise":
            raise ValueError(msg)
        if printer:
            printer("  " + msg)

    return {
        "cell": cell,
        "device": profile.as_dict(),
        "basis": basis,
        "log_path": os.path.abspath(log_path) if samples is None else None,
        "n_answers": n_answers,
        "window": window,
        "busy": busy,
        "cost_inputs": cost_inputs,
        "quality": {
            "ok": ok,
            "min_coverage": min_coverage,
            "idle_subtracted_w": floor,
            "busy_seconds": busy.get("covered_s") or 0.0,
            "log_rows": len(rows),
            "integrator": getattr(integrator, "__name__", str(integrator)),
            "measured_intervals": busy.get("measured_intervals"),
            "bracket_only_intervals": busy.get("bracket_only_intervals"),
            "inside_samples": busy.get("inside_samples"),
        },
    }


def measure_cell_serving_power(
    records: Sequence[Any],
    *,
    device: Any,
    priced: str = "serving",
    field_sets: Optional[Dict[str, Sequence[str]]] = None,
    latency_fields: Optional[Sequence[str]] = None,
    intervals_kwargs: Optional[Dict[str, Any]] = None,
    log_path: str = "out/power/power.csv",
    samples: Optional[Sequence[Tuple[datetime, float]]] = None,
    missing_log: str = "raise",
    tag: Optional[str] = None,
    constant_mode: str = "auto",
    constant_w: Optional[float] = None,
    constant_interval_s: float = 1.0,
    constant_upper_bound: bool = True,
    keep_alternates: bool = True,
    round_seconds: int = 3,
    printer: Optional[Callable[[str], None]] = print,
    cell: str = "",
    **power_kwargs: Any,
) -> Dict[str, Any]:
    """
    One cell measured on every latency sum at once, priced on the chosen one.

    `measure_cell_power` prices whatever single stretch it is handed. This front
    door runs it once per named field set off one read of the log, so the report
    can carry the full serving cost and the generation-only cost side by side
    and a reader can see how much of the bill is retrieval. Only the ``priced``
    set reaches ``cost_inputs``.

    A ``constant`` sampler device has no log by construction: its watts are one
    cited whole-machine figure, so the window is synthesised from that number
    and the result is labelled declared, never measured.

    Args:
        priced: the field set that becomes ``cost_inputs``.
        field_sets: name -> latency fields; None uses `SERVING_FIELD_SETS`.
        latency_fields: replaces the priced set's fields, leaving the rest.
        intervals_kwargs: `serving_intervals` knobs (``missing``,
            ``max_latency_ms``, ``default_latency_ms``, ...), shared by every set.
        log_path / samples / tag: the log, pre-loaded rows, or the tag to slice
            by. The log is read once and reused across the sets.
        missing_log: a sampled device with no log on disk ``raise``s, naming the
            meter command; ``ignore`` scores it on no rows, which prices the
            cell off the nameplate. A synthesised ``constant`` never reads it.
        constant_mode: ``auto`` synthesises only for a ``constant`` sampler with
            no rows handed in; ``always`` forces it; ``never`` reads the log.
        constant_w: the declared figure; None takes the device's ``constant_w``.
        constant_interval_s: spacing of the synthesised rows.
        constant_upper_bound: True marks the declared figure as an upper bound,
            which is what a nameplate sum is. Set False for a wall-meter reading.
        keep_alternates: keep every non-priced measurement under ``alternates``.
        round_seconds: decimals on the reported second totals.
        printer / cell: coverage warnings, priced set only.
        power_kwargs: every other `measure_cell_power` knob, forwarded unchanged
            (``basis``, ``subtract_idle``, ``min_coverage``, ``pad_start_s``, ...).

    Returns:
        The priced `measure_cell_power` dict, plus ``priced``,
        ``latency_field_sets``, ``serving_seconds`` (per set: per-answer seconds,
        totals, watts, coverage) and ``alternates``.
    """
    sets: Dict[str, Tuple[str, ...]] = {
        str(k): tuple(v) for k, v in (field_sets or SERVING_FIELD_SETS).items()
    }
    if latency_fields is not None:
        sets[priced] = tuple(latency_fields)
    if priced not in sets:
        raise ValueError(f"priced must name a field set, got {priced!r} of {sorted(sets)}")
    if constant_mode not in ("auto", "always", "never"):
        raise ValueError("constant_mode must be 'auto', 'always' or 'never'")
    for taken in ("intervals_fn", "intervals_kwargs", "latency_fields", "log_path",
                  "samples", "tag", "records", "device", "cell", "printer"):
        if taken in power_kwargs:
            raise ValueError(f"{taken} is this function's own argument, not a "
                             f"measure_cell_power passthrough")
    iv_kwargs = dict(intervals_kwargs or {})
    if "latency_fields" in iv_kwargs:
        raise ValueError("latency_fields belongs in field_sets or latency_fields, "
                         "not intervals_kwargs")

    profile = resolve_device(device, where="measure_cell_serving_power(device=)")
    spans = serving_intervals(records, latency_fields=sets[priced], **iv_kwargs)

    rows: Optional[Sequence[Tuple[datetime, float]]] = samples
    declared_w: Optional[float] = None
    synthesised = constant_mode == "always" or (
        constant_mode == "auto" and rows is None and profile.sampler == "constant")
    if synthesised:
        pick = constant_w if constant_w is not None else profile.constant_w
        if pick is None:
            raise ValueError(f"device {profile.name!r}: constant_mode={constant_mode!r} "
                             f"needs constant_w, on the device or here")
        declared_w = float(pick)
        pad = max(float(power_kwargs.get("pad_start_s", 0.0)),
                  float(power_kwargs.get("pad_end_s", 0.0))) + constant_interval_s
        if spans:
            rows = constant_samples(min(iv.start for iv in spans),
                                    max(iv.end for iv in spans), declared_w,
                                    interval_s=constant_interval_s, pad_s=pad)
        else:
            rows = []
    if rows is None:
        rows = require_power_log(log_path, device=profile, tag=tag,
                                 missing=missing_log,
                                 where="measure_cell_serving_power")

    logged_path = None if (samples is not None or synthesised) else os.path.abspath(log_path)
    out_by_set: Dict[str, Dict[str, Any]] = {}
    for name, fields in sets.items():
        is_priced = name == priced
        kwargs = dict(power_kwargs)
        if not is_priced:
            kwargs["on_uncovered"] = "ignore"
        one = measure_cell_power(
            records,
            device=profile,
            samples=rows,
            intervals_fn=serving_intervals,
            intervals_kwargs={**iv_kwargs, "latency_fields": fields},
            printer=printer if is_priced else None,
            cell=cell if is_priced else f"{cell}[{name}]",
            **kwargs,
        )
        one["latency_fields"] = list(fields)
        one["log_path"] = logged_path
        out_by_set[name] = one

    def _seconds(m: Dict[str, Any]) -> Dict[str, Any]:
        chosen = m.get(str(m.get("basis"))) or {}
        ci = m.get("cost_inputs") or {}
        return {
            "fields": m.get("latency_fields"),
            "n_answers": m.get("n_answers"),
            "per_answer_s": ci.get("query_gpu_seconds"),
            "total_s": round(float(chosen.get("seconds") or 0.0), round_seconds),
            "covered_s": round(float(chosen.get("covered_s") or 0.0), round_seconds),
            "watts": ci.get("watts"),
            "energy_wh": chosen.get("energy_wh"),
            "coverage": chosen.get("coverage"),
            "upper_bound": ci.get("upper_bound"),
        }

    result = dict(out_by_set[priced])
    result["priced"] = priced
    result["latency_field_sets"] = {k: list(v) for k, v in sets.items()}
    result["serving_seconds"] = {k: _seconds(v) for k, v in out_by_set.items()}
    if keep_alternates:
        result["alternates"] = {k: v for k, v in out_by_set.items() if k != priced}

    cost_inputs = dict(result.get("cost_inputs") or {})
    cost_inputs["serving_fields"] = list(sets[priced])
    summed = "+".join(sets[priced])
    if synthesised and (result.get("quality") or {}).get("ok"):
        cost_inputs["upper_bound"] = bool(constant_upper_bound)
        cost_inputs["declared_constant_w"] = declared_w
        cost_inputs["source"] = (f"declared constant {declared_w} W ({profile.name}, "
                                 f"whole machine, not sampled), {priced}={summed}")
        result["quality"] = {**(result.get("quality") or {}), "declared_constant": True}
    elif synthesised:
        cost_inputs["declared_constant_w"] = declared_w
        cost_inputs["source"] = (f"{cost_inputs.get('source') or 'unmeasured'} "
                                 f"(declared constant {declared_w} W covered nothing), "
                                 f"{priced}={summed}")
        result["quality"] = {**(result.get("quality") or {}), "declared_constant": False}
    else:
        cost_inputs["source"] = f"{cost_inputs.get('source') or 'unmeasured'}, {priced}={summed}"
    result["cost_inputs"] = cost_inputs
    return result


def save_power_summary(
    measure: Dict[str, Any],
    path: str,
    *,
    indent: int = 2,
    make_dirs: bool = True,
) -> str:
    """Write one cell's measurement beside its dump."""
    if make_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(measure, fh, indent=indent, default=str)
    return path
