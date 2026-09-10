"""
Device power profiles: how to read watts on the machine that served a cell.

Four ship. `omen` reads the GPU board alone and cannot see a CPU embedder;
`omen-system` scales whole-machine watts off a CPU utilisation counter, between a
measured idle floor and a cited base power; `omen-battery` measures the whole
machine off the ACPI battery's own discharge rate, which needs the charger out;
`spark` sums the two power channels tegrastats prints for the GB10. Any other
machine is a `DeviceProfile(...)` or a `device:` mapping in conditions.yaml.
A profile with no working sampler and no
`tdp_w` reports energy as unknown; no wattage is ever invented, and every number
in a profile's `notes` names where it came from.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

DEVICE_KEYS: Tuple[str, ...] = (
    "name", "label", "sampler", "argv", "pattern", "channels", "unit",
    "gpu_index", "tdp_w", "idle_w", "interval_s", "timeout_s", "constant_w",
    "load_w", "util_scale", "util_max", "util_exponent", "notes",
)

SAMPLERS: Tuple[str, ...] = ("nvidia-smi", "tegrastats", "sysfs", "command",
                             "utilization", "constant")

UNITS: Dict[str, float] = {"W": 1.0, "mW": 0.001, "uW": 1e-6, "kW": 1000.0}

UTIL_NUMBER_PATTERN: str = r"([-+]?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class DeviceProfile:
    """
    One machine's power sampler.

    Args:
        name: key used in conditions.yaml (`device: omen`).
        sampler: ``nvidia-smi`` | ``tegrastats`` | ``sysfs`` | ``command`` |
            ``utilization`` | ``constant``. ``command`` has no default argv: it
            runs whatever reads watts on that machine and parses the number out.
            ``utilization`` reads a busy fraction instead of watts and scales it
            between `idle_w` and `load_w`, for a machine that publishes no watts.
        argv: exact command, ``{index}`` substituted. Empty uses the default.
        pattern: regex whose group is a watt number; None parses the first float.
        channels: names of the power channels summed into one number, for a
            sampler that prints a table of them.
        unit: unit of the parsed numbers, converted through `UNITS`. Ignored by
            ``utilization``, whose reading is a fraction, not a power.
        tdp_w: nameplate, used only as an upper bound when sampling yields
            nothing, and as the ceiling of the ``utilization`` model. None
            reports unknown energy as unknown.
        idle_w: measured idle floor, subtracted only on request, and the
            intercept of the ``utilization`` model.
        constant_w: the only value the ``constant`` sampler returns, and the
            figure `power_meter` synthesises from on request.
        load_w: watts at utilisation 1.0, the slope end of the ``utilization``
            model. Pair it with the operating point the number was defined at:
            a CPU's base power is quoted for all cores at base clock, which is
            exactly what 100% ``% Processor Utility`` means.
        util_scale: multiplier from the parsed reading to a fraction. 0.01 for
            a percent counter, 1.0 for a sampler that already prints 0 to 1.
        util_max: cap on the fraction. None leaves it uncapped on purpose, so
            turbo above base clock can read past 1.0 and bill more than
            `load_w`, which is what the silicon actually does.
        util_exponent: shape of the curve, 1.0 linear. Raise it only with a
            measured fit to cite.
        notes: provenance of tdp_w, idle_w and load_w, carried into summaries.
    """

    name: str
    label: str = ""
    sampler: str = "nvidia-smi"
    argv: Tuple[str, ...] = ()
    pattern: Optional[str] = None
    channels: Tuple[str, ...] = ()
    unit: str = "W"
    gpu_index: int = 0
    tdp_w: Optional[float] = None
    idle_w: float = 0.0
    interval_s: float = 1.0
    timeout_s: float = 10.0
    constant_w: Optional[float] = None
    load_w: Optional[float] = None
    util_scale: float = 0.01
    util_max: Optional[float] = None
    util_exponent: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(
                f"device {self.name!r}: sampler must be one of {list(SAMPLERS)}, "
                f"got {self.sampler!r}"
            )
        if self.unit not in UNITS:
            raise ValueError(
                f"device {self.name!r}: unit must be one of {sorted(UNITS)}, got {self.unit!r}"
            )
        if self.sampler == "constant" and self.constant_w is None:
            raise ValueError(f"device {self.name!r}: sampler 'constant' needs constant_w")
        if self.sampler == "command" and not self.argv:
            raise ValueError(f"device {self.name!r}: sampler 'command' needs argv, "
                             f"the exact command that prints watts")
        if self.sampler == "utilization":
            if self.load_w is None:
                raise ValueError(
                    f"device {self.name!r}: sampler 'utilization' needs load_w, the "
                    f"watts this machine draws at utilisation 1.0, cited in `notes`"
                )
            if self.load_w <= self.idle_w:
                raise ValueError(
                    f"device {self.name!r}: load_w {self.load_w} must exceed idle_w "
                    f"{self.idle_w}, or busy work prices at or below doing nothing"
                )
        if self.util_scale <= 0:
            raise ValueError(f"device {self.name!r}: util_scale must be > 0, "
                             f"got {self.util_scale}")
        if self.util_max is not None and self.util_max <= 0:
            raise ValueError(f"device {self.name!r}: util_max must be > 0 or None, "
                             f"got {self.util_max}")
        if self.util_exponent <= 0:
            raise ValueError(f"device {self.name!r}: util_exponent must be > 0, "
                             f"got {self.util_exponent}")
        if self.tdp_w is not None and self.tdp_w <= 0:
            raise ValueError(f"device {self.name!r}: tdp_w must be > 0, got {self.tdp_w}")
        if self.idle_w < 0:
            raise ValueError(f"device {self.name!r}: idle_w must be >= 0, got {self.idle_w}")

    def command(self) -> Tuple[str, ...]:
        argv = self.argv or _DEFAULT_ARGV.get(self.sampler, ())
        return tuple(a.format(index=self.gpu_index) for a in argv)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "label": self.label, "sampler": self.sampler,
            "command": list(self.command()), "channels": list(self.channels),
            "unit": self.unit, "gpu_index": self.gpu_index, "tdp_w": self.tdp_w,
            "idle_w": self.idle_w, "interval_s": self.interval_s,
            "constant_w": self.constant_w, "load_w": self.load_w,
            "util_scale": self.util_scale, "util_max": self.util_max,
            "util_exponent": self.util_exponent, "notes": self.notes,
        }


_DEFAULT_ARGV: Dict[str, Tuple[str, ...]] = {
    "nvidia-smi": ("nvidia-smi", "--query-gpu=power.draw",
                   "--format=csv,noheader,nounits", "-i", "{index}"),
    "tegrastats": ("tegrastats", "--interval", "1000"),
    "sysfs": ("cat", "/sys/class/hwmon/hwmon0/power1_input"),
    "command": (),
    "utilization": ("typeperf", r"\Processor Information(_Total)\% Processor Utility",
                    "-sc", "1"),
    "constant": (),
}

DEVICES: Dict[str, DeviceProfile] = {
    "omen": DeviceProfile(
        name="omen",
        label="HP OMEN Transcend 14-fb0xxx, RTX 4070 laptop GPU board only",
        sampler="nvidia-smi",
        tdp_w=65.0,
        idle_w=11.3,
        notes="GPU board alone: power.draw excludes the CPU, so a CPU embedder "
              "bills nothing here. tdp_w 65 W is the board's power.max_limit "
              "(enforced.power.limit read back 41.88 W); idle_w 11.3 W is the "
              "mean of 12 samples taken on the idle machine, 11.19 to 11.79 W",
    ),
    "omen-system": DeviceProfile(
        name="omen-system",
        label="HP OMEN Transcend 14-fb0xxx, whole machine, utilisation-scaled",
        sampler="utilization",
        idle_w=39.1,
        load_w=56.3,
        util_scale=0.01,
        util_max=None,
        tdp_w=180.0,
        constant_w=43.7,
        interval_s=5.0,
        notes="Whole machine, and the only profile that bills a CPU embedder at "
              "all. HP publishes no watts for the Transcend 14, so they are modelled "
              "as idle_w + utilisation x (load_w - idle_w) off `% Processor "
              "Utility`, which is frequency-aware and so reads past 100% on "
              "turbo, billing past load_w exactly as the silicon does. "
              "idle_w 39.1 W is the settled whole-machine idle measured off the "
              "ACPI battery's discharge rate, panel dark and the app's "
              "containers up. "
              "load_w 56.3 W = 45 W Intel Processor Base Power for the Core "
              "Ultra 9 185H + 11.3 W measured GPU idle, and the two operating "
              "points match the counter: base power is quoted for all cores at "
              "base clock, which is what 100% utility means. Cross-check: idle "
              "utility read 8.96 to 13.43% on three back-to-back polls, where "
              "the model returns 40.6 to 41.4 W against the 39.1 W settled "
              "idle; a 22-worker all-core load sat at 17 to 41% "
              "`LoadPercentage` (a different counter, coarser) and measured "
              "43.7 W mean, 60.0 W peak, which the model brackets at 42.0 to "
              "46.1 W. tdp_w 180 W (115 W CPU max "
              "turbo + 65 W GPU max) caps the model and prices a failed run; "
              "constant_w 43.7 W is that measured loaded mean, for a run with no "
              "meter log. `typeperf` costs ~0.3 s of the 1 s it samples, so the "
              "meter contributes well under a point of the utilisation it reads. "
              "Counter names are English-locale",
    ),
    "omen-battery": DeviceProfile(
        name="omen-battery",
        label="HP OMEN Transcend 14-fb0xxx, whole machine, battery discharge rate",
        sampler="command",
        argv=("powershell", "-NoProfile", "-NonInteractive", "-Command",
              r"(Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus).DischargeRate"),
        pattern=r"([1-9]\d*)",
        unit="mW",
        tdp_w=180.0,
        idle_w=39.1,
        interval_s=15.0,
        notes="The one whole-machine measurement the Transcend 14 gives with nothing "
              "installed: the ACPI battery reports its discharge rate in mW, "
              "which is every watt the machine draws, CPU embedder included. "
              "Unplug first. On AC the rate reads 0, and the pattern rejects 0 "
              "on purpose so a plugged-in run fails coverage loudly instead of "
              "pricing the cell at 0 W. The firmware latches the value: 25 "
              "consecutive 1 s reads returned one identical number, so it "
              "refreshes every 15 to 30 s, which is why interval_s is 15 and why "
              "a per-answer busy slice can land inside a single latched reading. "
              "Price a battery run on the window basis. idle_w 39.1 W is the "
              "settled whole-machine idle read off this same rate with the panel dark "
              "and the app's containers up; single reads spanned 27.2 to 52.5 W "
              "as background load moved, so trust the settled value, not one poll",
    ),
    "spark": DeviceProfile(
        name="spark",
        label="NVIDIA DGX Spark, GB10",
        sampler="tegrastats",
        pattern=r"{channel}\s+(\d+)mW",
        channels=("VDD_GPU_SOC", "VDD_CPU_CV"),
        unit="mW",
        tdp_w=None,
        notes="GB10 publishes no nvidia-smi power.draw; the two power channels "
              "tegrastats prints are summed. "
              "tdp_w unset on purpose: pass a cited nameplate to price a failed run",
    ),
}


def register_device(profile: DeviceProfile, *, overwrite: bool = False) -> DeviceProfile:
    """Add a profile. Collisions raise, so two machines cannot share one name."""
    if profile.name in DEVICES and not overwrite:
        raise ValueError(f"device {profile.name!r} already registered; pass overwrite=True")
    DEVICES[profile.name] = profile
    return profile


def device_from_mapping(
    data: Mapping[str, Any],
    *,
    name: Optional[str] = None,
    base: Optional[DeviceProfile] = None,
    strict_keys: bool = True,
    where: str = "device",
) -> DeviceProfile:
    """
    Build a profile from a ``device:`` mapping.

    Args:
        data: may name ``base:`` to override a shipped profile, e.g.
            ``{base: omen, gpu_index: 1}``.
        strict_keys: an unknown key raises, so a typo in `tdp_w` cannot leave
            the fallback unset.
    """
    spec = dict(data)
    base_name = spec.pop("base", None)
    parent = base
    if base_name is not None:
        parent = resolve_device(str(base_name), where=f"{where}.base")
    stray = [k for k in spec if k not in DEVICE_KEYS]
    if stray and strict_keys:
        raise ValueError(
            f"{where}: unknown device key(s) {sorted(stray)}; accepted: {sorted(DEVICE_KEYS)}"
        )
    for k in stray:
        spec.pop(k)
    if spec.get("argv") is not None:
        spec["argv"] = tuple(str(a) for a in spec["argv"])
    if spec.get("channels") is not None:
        spec["channels"] = tuple(str(c) for c in spec["channels"])
    if parent is None:
        spec.setdefault("name", str(name or spec.get("name") or "device"))
        return DeviceProfile(**spec)
    spec.setdefault("name", str(name or parent.name))
    return replace(parent, **spec)


def resolve_device(
    device: Any,
    *,
    registry: Optional[Mapping[str, DeviceProfile]] = None,
    where: str = "device",
    required: bool = True,
) -> Optional[DeviceProfile]:
    """Name, mapping, `DeviceProfile` or None -> profile. Unknown names raise."""
    if device is None:
        if required:
            raise ValueError(f"{where}: no device declared")
        return None
    if isinstance(device, DeviceProfile):
        return device
    if isinstance(device, Mapping):
        return device_from_mapping(device, where=where)
    reg = DEVICES if registry is None else registry
    key = str(device).strip()
    if key not in reg:
        raise ValueError(f"{where}: unknown device {key!r}; known: {sorted(reg)}")
    return reg[key]


def parse_power_w(
    text: str,
    profile: DeviceProfile,
    *,
    require_all_channels: bool = True,
) -> float:
    """One watt number out of one reading; NaN when nothing parsed."""
    scale = UNITS[profile.unit]
    if profile.channels:
        total = 0.0
        seen = 0
        for channel in profile.channels:
            pat = (profile.pattern or r"{channel}\s+(\d+)mW").format(
                channel=re.escape(channel))
            m = re.search(pat, text)
            if m is None:
                if require_all_channels:
                    raise ValueError(
                        f"device {profile.name!r}: power channel {channel!r} absent "
                        f"from sampler output; fix `channels:` rather than price a "
                        f"partial sum"
                    )
                continue
            total += float(m.group(1))
            seen += 1
        return (total * scale) if seen else float("nan")
    if profile.pattern:
        m = re.search(profile.pattern, text)
        return float(m.group(1)) * scale if m else float("nan")
    try:
        return float(text.strip().splitlines()[0].split(",")[0]) * scale
    except Exception:
        return float("nan")


def parse_utilization_line(
    line: str,
    *,
    pattern: str = UTIL_NUMBER_PATTERN,
    field_index: int = -1,
    field_sep: str = ",",
    strip_chars: str = "\" \t\r",
) -> float:
    """The raw number out of one line, unscaled and unclamped; NaN when absent.

    Args:
        pattern: regex whose group 1 is the number, applied to the chosen field.
        field_index: which separated field on the line, -1 for the last.
        field_sep: field separator; "" reads the whole line as one field.
        strip_chars: trimmed off the field before matching, quotes by default.
    """
    if field_sep:
        fields = line.split(field_sep)
        try:
            field = fields[field_index]
        except IndexError:
            return float("nan")
    else:
        field = line
    field = field.strip(strip_chars) if strip_chars else field
    m = re.search(pattern, field)
    if m is None:
        return float("nan")
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return float("nan")


def parse_utilization(
    text: str,
    *,
    pattern: str = UTIL_NUMBER_PATTERN,
    line_index: Optional[int] = None,
    field_index: int = -1,
    field_sep: str = ",",
    strip_chars: str = "\" \t\r",
    scale: float = 0.01,
    max_fraction: Optional[float] = None,
    floor_fraction: Optional[float] = 0.0,
) -> float:
    """One busy fraction out of one utilisation reading; NaN when nothing parsed.

    Takes a named field of a named line rather than the first number in the
    text, because a counter sampler wraps the value in noise that carries
    digits: `typeperf` opens with ``"(PDH-CSV 4.0)"``, stamps each row with a
    date, and closes with ``[Exiting, please wait...]`` and ``The command
    completed successfully.``, so neither the first line nor the last holds the
    reading.

    Args:
        pattern: regex whose group 1 is the number, applied to the chosen field.
        line_index: which non-empty line holds the reading, counting from 0 and
            negative from the end. None walks the lines backwards and takes the
            newest one whose field parses, which skips a trailer, a preamble or
            a locale warning and takes the last row of a multi-row sample.
        field_index: which separated field on that line, -1 for the last.
        field_sep: field separator; "" reads the whole line as one field.
        strip_chars: trimmed off the field before matching, quotes by default.
        scale: multiplier to a fraction, 0.01 for a percent counter.
        max_fraction: cap after scaling. None leaves turbo readings above 1.0
            intact, since the machine really does draw more than base power.
        floor_fraction: lower clamp. None keeps a negative reading, which is a
            broken counter worth surfacing rather than a 0% machine.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return float("nan")
    if line_index is None:
        candidates = reversed(lines)
    else:
        try:
            candidates = iter([lines[line_index]])
        except IndexError:
            return float("nan")
    raw = float("nan")
    for line in candidates:
        raw = parse_utilization_line(line, pattern=pattern, field_index=field_index,
                                     field_sep=field_sep, strip_chars=strip_chars)
        if raw == raw:
            break
    if raw != raw:
        return float("nan")
    frac = raw * scale
    if floor_fraction is not None:
        frac = max(floor_fraction, frac)
    if max_fraction is not None:
        frac = min(max_fraction, frac)
    return frac


def utilization_watts(
    fraction: float,
    *,
    idle_w: float,
    load_w: float,
    exponent: float = 1.0,
    ceiling_w: Optional[float] = None,
    floor_w: Optional[float] = None,
) -> float:
    """Busy fraction -> watts: ``idle_w + f**exponent * (load_w - idle_w)``.

    Args:
        fraction: busy fraction; may exceed 1.0, which bills above `load_w`.
        idle_w: watts at fraction 0, the measured floor.
        load_w: watts at fraction 1, the operating point the fraction is
            defined at (all cores at base clock, for a base-power figure).
        exponent: curve shape, 1.0 linear. Raise it only with a measured fit.
        ceiling_w: hard cap, normally the nameplate. None does not cap.
        floor_w: hard floor. None falls back to `idle_w`.
    """
    if fraction != fraction:
        return float("nan")
    watts = float(idle_w) + (float(fraction) ** float(exponent)) * (
        float(load_w) - float(idle_w))
    low = float(idle_w) if floor_w is None else float(floor_w)
    watts = max(low, watts)
    if ceiling_w is not None:
        watts = min(float(ceiling_w), watts)
    return watts


def sample_utilization_w(
    profile: DeviceProfile,
    *,
    runner: Optional[Callable[..., Any]] = None,
    timeout_s: Optional[float] = None,
    pattern: Optional[str] = None,
    scale: Optional[float] = None,
    max_fraction: Optional[float] = None,
    exponent: Optional[float] = None,
    idle_w: Optional[float] = None,
    load_w: Optional[float] = None,
    ceiling_w: Optional[float] = None,
    **parse_kwargs: Any,
) -> float:
    """One live sample in watts off a utilisation counter; NaN when silent.

    Every term of the model defaults to the profile and stays overridable here,
    so a one-off machine needs an argument, not a second profile.

    Args:
        runner: injected `subprocess.run`, for testing without the hardware.
        ceiling_w: cap on the modelled watts, `profile.tdp_w` by default.
        parse_kwargs: passed through to `parse_utilization`, e.g. `field_index`.
    """
    run = runner or subprocess.run
    try:
        out = run(list(profile.command()), capture_output=True, text=True,
                  timeout=timeout_s if timeout_s is not None else profile.timeout_s)
    except Exception:
        return float("nan")
    frac = parse_utilization(
        getattr(out, "stdout", "") or "",
        pattern=(profile.pattern or UTIL_NUMBER_PATTERN) if pattern is None else pattern,
        scale=profile.util_scale if scale is None else scale,
        max_fraction=profile.util_max if max_fraction is None else max_fraction,
        **parse_kwargs,
    )
    return utilization_watts(
        frac,
        idle_w=profile.idle_w if idle_w is None else idle_w,
        load_w=(profile.load_w if load_w is None else load_w),
        exponent=profile.util_exponent if exponent is None else exponent,
        ceiling_w=profile.tdp_w if ceiling_w is None else ceiling_w,
    )


def sample_power_w(
    profile: DeviceProfile,
    *,
    runner: Optional[Callable[..., Any]] = None,
    timeout_s: Optional[float] = None,
    require_all_channels: bool = True,
) -> float:
    """One live sample in watts; NaN when the sampler is missing or silent.

    Args:
        runner: injected `subprocess.run`, for testing without the hardware.
    """
    if profile.sampler == "constant":
        return float(profile.constant_w)
    if profile.sampler == "utilization":
        return sample_utilization_w(profile, runner=runner, timeout_s=timeout_s)
    run = runner or subprocess.run
    try:
        out = run(list(profile.command()), capture_output=True, text=True,
                  timeout=timeout_s if timeout_s is not None else profile.timeout_s)
    except Exception:
        return float("nan")
    return parse_power_w(getattr(out, "stdout", "") or "", profile,
                         require_all_channels=require_all_channels)


def probe_device(
    profile: DeviceProfile,
    *,
    samples: int = 3,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Check the sampler works before a cell is answered, not after."""
    reads = []
    for _ in range(max(1, samples)):
        try:
            reads.append(sample_power_w(profile, runner=runner))
        except ValueError as exc:
            return {"device": profile.name, "ok": False, "error": str(exc),
                    "command": list(profile.command())}
    valid = [w for w in reads if w == w]
    return {
        "device": profile.name,
        "label": profile.label,
        "command": list(profile.command()),
        "ok": bool(valid),
        "samples": len(reads),
        "valid_samples": len(valid),
        "mean_w": round(sum(valid) / len(valid), 3) if valid else None,
        "fallback_tdp_w": profile.tdp_w,
        "error": None if valid else "sampler returned no parseable watts",
    }
