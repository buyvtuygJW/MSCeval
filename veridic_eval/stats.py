"""
Statistics - bootstrap BCa confidence intervals and paired cell deltas.

Pinned tooling: no framework ships this end-to-end, so the
standard published practice applies - a small scipy wrapper over the per-item
0/1 columns produced by the retrieval / faithfulness / abstention / CFCA
modules. BCa is used because binary-metric bootstrap distributions are skewed.

  * per-cell CI:  scipy.stats.bootstrap((scores,), np.mean, n_resamples=9999,
                  confidence_level=0.95, method="BCa", random_state=42)
  * paired delta: same call with (a, b), statistic = mean(a)-mean(b),
                  paired=True; significant when both CI ends fall one side of 0.
  * optional:     deepsig.aso on the same paired contrasts, under the same
                  Bonferroni family (num_comparisons).
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import settings


def _clean(scores: List[float]) -> np.ndarray:
    return np.asarray([s for s in scores if s is not None], dtype=float)


def bootstrap_ci(
    scores: List[float],
    n_resamples: Optional[int] = None,
    confidence_level: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Per-cell mean + BCa 95% CI. Degenerate inputs fall back gracefully."""
    arr = _clean(scores)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "method": None}
    mean = float(np.mean(arr))
    if n < 2 or np.allclose(arr, arr[0]):
        # BCa is undefined for constant / singleton samples.
        return {"n": n, "mean": round(mean, 6), "ci_low": round(mean, 6),
                "ci_high": round(mean, 6), "method": "degenerate"}

    from scipy.stats import bootstrap  # lazy

    res = bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_resamples or settings.bootstrap_resamples,
        confidence_level=confidence_level or settings.confidence_level,
        method="BCa",
        random_state=seed if seed is not None else settings.random_state,
    )
    return {
        "n": n,
        "mean": round(mean, 6),
        "ci_low": round(float(res.confidence_interval.low), 6),
        "ci_high": round(float(res.confidence_interval.high), 6),
        "method": "BCa",
    }


def _align(a: Dict[str, float], b: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Item-level pairing preserved: common query ids only, same order."""
    keys = [k for k in a.keys() if k in b and a[k] is not None and b[k] is not None]
    return (np.asarray([a[k] for k in keys], dtype=float),
            np.asarray([b[k] for k in keys], dtype=float),
            keys)


def paired_delta_ci(
    a: Dict[str, float],
    b: Dict[str, float],
    n_resamples: Optional[int] = None,
    confidence_level: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Paired bootstrap CI on mean(a) - mean(b) over common query ids.
    ``significant`` is True when both CI ends fall one side of 0.
    """
    xa, xb, keys = _align(a, b)
    n = len(keys)
    if n == 0:
        return {"n": 0, "delta": None, "ci_low": None, "ci_high": None,
                "significant": None, "method": None}
    delta = float(np.mean(xa) - np.mean(xb))
    if n < 2 or np.allclose(xa - xb, (xa - xb)[0]):
        return {"n": n, "delta": round(delta, 6), "ci_low": round(delta, 6),
                "ci_high": round(delta, 6), "significant": False, "method": "degenerate"}

    from scipy.stats import bootstrap  # lazy

    def _diff(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = bootstrap(
        (xa, xb),
        _diff,
        paired=True,
        n_resamples=n_resamples or settings.bootstrap_resamples,
        confidence_level=confidence_level or settings.confidence_level,
        method="BCa",
        random_state=seed if seed is not None else settings.random_state,
    )
    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    return {
        "n": n,
        "delta": round(delta, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "significant": bool(low > 0 or high < 0),
        "method": "BCa",
    }


def paired_permutation_p(
    a: Dict[str, float],
    b: Dict[str, float],
    n_resamples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[float]:
    """
    Two-sided paired permutation test on mean(a - b) over common query ids
    (scipy, dependency-free). Complements the bootstrap CI as the paired
    test per query. Returns the p-value, or None if fewer than 2 paired items
    or the two columns are identical (a zero difference on every pair leaves
    nothing to test). A constant *non-zero* difference is a real signal and
    gets a p-value: every sign flip moves the statistic.
    """
    xa, xb, keys = _align(a, b)
    if len(keys) < 2 or np.allclose(xa - xb, 0.0):
        return None

    from scipy.stats import permutation_test  # lazy

    res = permutation_test(
        (xa, xb),
        lambda x, y: np.mean(x - y),
        permutation_type="samples",
        n_resamples=n_resamples or settings.bootstrap_resamples,
        alternative="two-sided",
        random_state=seed if seed is not None else settings.random_state,
    )
    return float(res.pvalue)


def bonferroni(p: Optional[float], num_comparisons: int) -> Optional[float]:
    """Bonferroni-corrected p-value: min(1, p * m). None-safe."""
    if p is None or num_comparisons <= 0:
        return p
    return min(1.0, p * num_comparisons)


def aso_significance(
    a: Dict[str, float],
    b: Dict[str, float],
    num_comparisons: int = 1,
    seed: Optional[int] = None,
) -> Optional[float]:
    """
    Optional Almost Stochastic Order score (deepsig.aso, Dror et al., ACL 2019).
    num_comparisons is the caller's Bonferroni family, the same one the
    permutation p uses; 1 is uncorrected, never a family invented here.
    Returns the ASO violation ratio epsilon_min (<0.5 => a stochastically
    dominant over b), or None if deepsig is not installed / disabled.
    """
    if not settings.enable_deepsig:
        return None
    xa, xb, keys = _align(a, b)
    if len(keys) < 2:
        return None
    try:
        from deepsig import aso  # lazy, GPL-3.0 optional dep

        return float(
            aso(xa, xb, confidence_level=settings.confidence_level,
                num_comparisons=num_comparisons, seed=seed or settings.random_state)
        )
    except Exception:
        return None


# --------------------------------------------------- ratio deltas (CFCA)
_ZERO_DEN_POLICIES = ("warn", "raise", "ignore")


def _const(arr: np.ndarray) -> bool:
    """A column that cannot move under resampling."""
    return arr.size < 2 or bool(np.allclose(arr, arr.flat[0]))


def _pick(col: Dict[str, float], keys: List[str]) -> np.ndarray:
    return np.asarray([col[k] for k in keys], dtype=float)


def _align_ratio(
    num_a: Dict[str, float],
    den_a: Dict[str, float],
    num_b: Dict[str, float],
    den_b: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Four columns on one index: query ids present and non-null in all four."""
    cols = (num_a, den_a, num_b, den_b)
    keys = [k for k in num_a if all(k in c and c[k] is not None for c in cols)]
    return (_pick(num_a, keys), _pick(den_a, keys),
            _pick(num_b, keys), _pick(den_b, keys), keys)


def _ratio(num, den, axis=-1) -> np.ndarray:
    """mean(num) / mean(den) per draw, +inf where the denominator mean is 0."""
    m_num = np.asarray(np.mean(num, axis=axis), dtype=float)
    m_den = np.asarray(np.mean(den, axis=axis), dtype=float)
    val = np.full(m_num.shape, np.inf, dtype=float)
    np.divide(m_num, m_den, out=val, where=(m_den != 0.0))
    return val


def _zero_den_note(zero: int, draws: int) -> str:
    return (f"{zero} of {draws} resample draws had a zero all-pass denominator, "
            "where cost per faithful cited answer is undefined")


def _emit(msg: str, policy: str, warn: Optional[Callable[[str], None]]) -> None:
    if policy == "ignore":
        return
    if policy == "raise":
        raise ValueError(msg)
    if warn is not None:
        warn(msg)
    else:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def paired_ratio_delta_ci(
    num_a: Dict[str, float],
    den_a: Dict[str, float],
    num_b: Dict[str, float],
    den_b: Dict[str, float],
    *,
    n_resamples: Optional[int] = None,
    confidence_level: Optional[float] = None,
    seed: Optional[int] = None,
    on_zero_denominator: str = "warn",
    warn: Optional[Callable[[str], None]] = None,
    round_to: int = 8,
) -> Dict:
    """
    Paired BCa CI on a difference of two ratios: mean(num_a) / mean(den_a)
    minus mean(num_b) / mean(den_b).

    Each cell gets its own paired bootstrap — questions are resampled with the
    (numerator, denominator) pair kept together, so a resampled question moves
    cost and all-pass jointly — and a BCa interval on its ratio. The delta
    interval is the interval difference [lo_a - hi_b, hi_a - lo_b], which is
    conservative (coverage at least nominal) and keeps each cell's own
    uncertainty visible: a question that costs more to answer than its
    neighbours widens its cell's interval, and therefore the delta interval,
    instead of disappearing into a mean or being cancelled by cross-cell
    correlation.

    CFCA is cost / P(faithful . cited . right-version), so its delta is a
    difference of ratios and not a mean of per-question numbers.

    Args:
        num_a / den_a: per-question numerator and denominator for the cell.
        num_b / den_b: the same two columns for the reference cell.
        n_resamples / confidence_level / seed: bootstrap knobs; None takes the
            pinned settings (9999, 0.95, 42).
        on_zero_denominator: "warn" | "raise" | "ignore" for draws where no
            sampled answer passes both gates, which leaves the ratio undefined.
        warn: sink for that warning; None routes to ``warnings.warn``.
        round_to: decimals on every emitted number. The degenerate branch emits
            one value for the point and both ends, so a checker reading the
            point as inside its own interval finds it there exactly.

    Returns ``n``, ``delta``, ``ci_low``, ``ci_high``, ``significant``,
    ``method``, plus ``zero_denominator_draws``, ``resample_draws``, ``note``.
    """
    if on_zero_denominator not in _ZERO_DEN_POLICIES:
        raise ValueError(
            f"on_zero_denominator must be one of {list(_ZERO_DEN_POLICIES)}, "
            f"got {on_zero_denominator!r}"
        )
    xan, xad, xbn, xbd, keys = _align_ratio(num_a, den_a, num_b, den_b)
    n = len(keys)
    out: Dict = {"n": n, "delta": None, "ci_low": None, "ci_high": None,
                 "significant": None, "method": None,
                 "zero_denominator_draws": 0, "resample_draws": 0, "note": None}
    if n == 0:
        out["note"] = "no question is scored in both cells"
        return out

    ra, rb = float(_ratio(xan, xad)), float(_ratio(xbn, xbd))
    if not (np.isfinite(ra) and np.isfinite(rb)):
        out["note"] = ("a cell has no answer that passes both gates, so its "
                       "ratio is undefined and no delta is reported")
        return out
    delta = round(ra - rb, round_to)

    flat = None
    if n < 2:
        flat = "one paired question only, so there is nothing to resample"
    elif all(_const(c) for c in (xan, xad, xbn, xbd)):
        flat = "no spread to resample: all four columns are constant"
    if flat is not None:
        out.update({"delta": delta, "ci_low": delta, "ci_high": delta,
                    "significant": bool(delta > 0 or delta < 0),
                    "method": "degenerate", "note": flat})
        return out

    from scipy.stats import bootstrap  # lazy

    seen = {"draws": 0, "zero": 0}

    def _counted(num, den, axis):
        m_num = np.asarray(np.mean(num, axis=axis), dtype=float)
        m_den = np.asarray(np.mean(den, axis=axis), dtype=float)
        bad = m_den == 0.0
        seen["draws"] += int(bad.size)
        seen["zero"] += int(np.count_nonzero(bad))
        val = np.full(m_num.shape, np.inf, dtype=float)
        np.divide(m_num, m_den, out=val, where=~bad)
        return val

    b_resamples = n_resamples or settings.bootstrap_resamples
    b_level = confidence_level or settings.confidence_level
    b_seed = seed if seed is not None else settings.random_state

    def _cell_ci(point, num, den):
        """Paired BCa interval on one cell's ratio; point interval when the
        cell has no resampling variance."""
        if _const(num) and _const(den):
            return point, point
        r = bootstrap(
            (num, den),
            _counted,
            paired=True,
            n_resamples=b_resamples,
            confidence_level=b_level,
            method="BCa",
            random_state=b_seed,
        )
        return (float(r.confidence_interval.low),
                float(r.confidence_interval.high))

    la, ha = _cell_ci(ra, xan, xad)
    lb, hb = _cell_ci(rb, xbn, xbd)
    low, high = la - hb, ha - lb

    notes: List[str] = []
    if seen["zero"]:
        notes.append(_zero_den_note(seen["zero"], seen["draws"]))
        _emit("CFCA delta: " + notes[-1], on_zero_denominator, warn)
    out.update({"delta": delta, "method": "BCa",
                "zero_denominator_draws": seen["zero"],
                "resample_draws": seen["draws"]})
    if np.isfinite(low) and np.isfinite(high):
        out["ci_low"], out["ci_high"] = round(low, round_to), round(high, round_to)
        out["significant"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    else:
        notes.append("the resampled interval is unbounded, so none is reported")
    out["note"] = "; ".join(notes) or None
    return out


def paired_ratio_permutation_p(
    num_a: Dict[str, float],
    den_a: Dict[str, float],
    num_b: Dict[str, float],
    den_b: Dict[str, float],
    *,
    n_resamples: Optional[int] = None,
    seed: Optional[int] = None,
    exact_max_draws: int = 20000,
    tol: float = 1e-12,
) -> Optional[float]:
    """
    Two-sided paired permutation p for a difference of two ratios.

    The exchangeable unit is the question: one draw swaps a question's
    (numerator, denominator) pair between the two cells, so cost and all-pass
    stay together. Every one of the 2**n sign patterns is enumerated while that
    count stays under `exact_max_draws`, else `n_resamples` of them are sampled
    and the p-value takes the +1 correction. A draw whose denominator vanishes
    counts as extreme, so the p-value errs high rather than low.

    Returns None when fewer than two questions pair or the observed statistic is
    undefined.
    """
    xan, xad, xbn, xbd, keys = _align_ratio(num_a, den_a, num_b, den_b)
    n = len(keys)
    if n < 2:
        return None
    obs = float(_ratio(xan, xad)) - float(_ratio(xbn, xbd))
    if not np.isfinite(obs):
        return None
    if 2 ** n <= exact_max_draws:
        flip = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
        divisor, offset = float(flip.shape[0]), 0.0
    else:
        rng = np.random.default_rng(
            seed if seed is not None else settings.random_state
        )
        draws = int(n_resamples or settings.bootstrap_resamples)
        flip = rng.random((draws, n)) < 0.5
        divisor, offset = float(draws) + 1.0, 1.0
    t = (_ratio(np.where(flip, xbn, xan), np.where(flip, xbd, xad), axis=1)
         - _ratio(np.where(flip, xan, xbn), np.where(flip, xad, xbd), axis=1))
    with np.errstate(invalid="ignore"):
        extreme = ~np.isfinite(t) | (np.abs(t) >= abs(obs) - tol)
    return float((offset + np.count_nonzero(extreme)) / divisor)
