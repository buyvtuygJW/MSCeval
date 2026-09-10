"""CFCA cost calculator for the rescoped methodology. All prices GBP.

Formulas implemented verbatim from methodnow-rescoped.md:
    Cost(c) = C_onetime(c)/A + C_query(c) + (U * C_update(c) + M * index_GB(c) * p_store) / Q
    CFCA(c) = Cost(c) / P_hat_c          (P_hat_c = all-pass rate of faithful_i * cited_i)

GPU compute is priced by METERED ELECTRICITY (methodology option ii):
    p_gpu (GBP/GPU-hour) = device_watts / 1000 * kwh_gbp
Hardware purchase is a one-time capital cost reported in the write-up, never
part of Cost(c). Token prices are GBP per 1M tokens; convert USD sheets once at a
cited FX rate before passing them in.

Every number is an explicit, overridable argument; defaults are 0.0 (= free/local),
so you only fill in what you actually pay.

Usage examples:
    veridic-eval cost --watts 43.7 --kwh-gbp 0.26 --onetime-gpu-hours 3.5 --A 5000 \
        --query-gpu-seconds 2.1 --p-hat 0.82
    veridic-eval cost --p-in-mtok 0.12 --p-out-mtok 0.48 \
        --query-in-mtok 0.004 --query-out-mtok 0.0006 --p-hat 0.77
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Mapping, Optional, Sequence


def electricity_gpu_hour(watts: float, kwh_gbp: float) -> float:
    """GBP per GPU-hour from device power draw and the cited Ofgem unit rate."""
    return (watts / 1000.0) * kwh_gbp


def cost_per_answer(
    *,
    # --- unit prices, GBP (0.0 = free/local; set only what you pay, each cited) ---
    p_gpu_hour: float = 0.0,      # GBP/GPU-hour, from electricity_gpu_hour()
    p_embed_mtok: float = 0.0,    # GBP per 1M embedding tokens (API)
    p_in_mtok: float = 0.0,       # GBP per 1M generator prompt tokens (API)
    p_out_mtok: float = 0.0,      # GBP per 1M generator completion tokens (API)
    p_store_gb_month: float = 0.0,  # GBP per GB-month of vector storage; local disk -> 0
    # --- C_onetime: build the condition once (ingest+embed+index, or fine-tune) ---
    onetime_gpu_hours: float = 0.0,
    onetime_embed_mtok: float = 0.0,
    A: float = 1.0,               # answers the one-time cost amortises over
    # --- C_query: one answer ---
    query_gpu_seconds: float = 0.0,   # serving compute per answer (measured wall-clock)
    query_embed_mtok: float = 0.0,    # query embedding tokens
    query_in_mtok: float = 0.0,       # prompt tokens per answer
    query_out_mtok: float = 0.0,      # completion tokens per answer
    # --- recurring: document additions (re-embed + re-index) and storage, spread over Q queries ---
    U: float = 0.0,               # documents added in the period
    update_gpu_hours: float = 0.0,   # indexing compute per added document
    update_embed_mtok: float = 0.0,  # embedding tokens per added document
    M: float = 0.0,               # storage months in the period
    index_gb: float = 0.0,        # index size in GB
    Q: float = 1.0,               # queries in the period
) -> dict[str, float]:
    """Return the cost breakdown per answer, in GBP. Pure arithmetic, no I/O."""
    c_onetime = onetime_gpu_hours * p_gpu_hour + onetime_embed_mtok * p_embed_mtok
    c_query = (
        (query_gpu_seconds / 3600.0) * p_gpu_hour
        + query_embed_mtok * p_embed_mtok
        + query_in_mtok * p_in_mtok
        + query_out_mtok * p_out_mtok
    )
    c_update = update_gpu_hours * p_gpu_hour + update_embed_mtok * p_embed_mtok
    recurring = (U * c_update + M * index_gb * p_store_gb_month) / Q
    total = c_onetime / A + c_query + recurring
    return {
        "C_onetime": c_onetime,
        "C_onetime/A": c_onetime / A,
        "C_query": c_query,
        "C_update": c_update,
        "recurring/Q": recurring,
        "Cost_per_answer": total,
    }


def cfca(cost_per_answer_gbp: float, p_hat: float) -> float | None:
    """CFCA = Cost / P_hat. Returns None when P_hat <= 0, where the ratio is undefined."""
    if p_hat <= 0.0:
        return None
    return cost_per_answer_gbp / p_hat


#: USD per 1 GBP, ECB reference rate via api.frankfurter.dev, retrieved 2026-08-24.
GBP_USD: float = 1.3634
GBP_USD_SOURCE: str = "ECB via api.frankfurter.dev, GBP->USD 1.3634, retrieved 2026-08-24"


def gbp_from_usd(usd: float, *, gbp_usd: float = GBP_USD) -> float:
    """A USD figure in GBP, at ``gbp_usd`` dollars to the pound."""
    if gbp_usd <= 0.0:
        raise ValueError(f"gbp_usd must be positive, got {gbp_usd!r}")
    return float(usd) / gbp_usd


def per_answer_cost_gbp(
    cost_inputs: Mapping[str, Mapping[str, float]],
    *,
    usd_cost: Optional[Mapping[str, float]] = None,
    gbp_usd: float = GBP_USD,
    cost_fn: Callable[..., Dict[str, float]] = cost_per_answer,
    total_key: str = "Cost_per_answer",
    cells: Optional[Sequence[str]] = None,
    round_to: Optional[int] = 8,
) -> Dict[str, float]:
    """``{cell: GBP cost per answer}``, the CFCA numerator the report scores on.

    One pound figure per cell: the measured electricity and token quantities
    priced by ``cost_fn``, plus any API bill declared in USD, converted once at
    a cited rate. Pure arithmetic, no network: the rate is a number in, so a run
    is reproducible from the report.

    Args:
        cost_inputs: ``{cell: cost_per_answer kwargs}``, already GBP-priced.
        usd_cost: ``{cell: USD per answer}`` billed by a provider, added after
            conversion; a local cell declares 0.0 and the term vanishes.
        gbp_usd: dollars to the pound; the module default carries its date.
        cost_fn / total_key: the formula and the key its total sits under.
        cells: cell order and membership; None takes both mappings, quantities
            first, so a cell that only declares a USD bill still gets a figure.
        round_to: decimals on each figure; None keeps full precision.
    """
    billed = dict(usd_cost or {})
    names = list(cells) if cells is not None else (
        list(cost_inputs) + [n for n in billed if n not in cost_inputs]
    )
    out: Dict[str, float] = {}
    for name in names:
        quantities = cost_inputs.get(name)
        total = float(cost_fn(**quantities)[total_key]) if quantities else 0.0
        total += gbp_from_usd(billed[name], gbp_usd=gbp_usd) if billed.get(name) else 0.0
        out[name] = total if round_to is None else round(total, round_to)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="veridic-eval cost", description="CFCA cost calculator, GBP. No silent defaults: every quantity is required; pass 0 explicitly only for a component that does not apply.")
    ap.add_argument("--watts", type=float, default=None, help="device power draw under load: mean_w from `veridic-eval power` during the job; if polling is unavailable use 1.0*TDP (full nameplate) for any job and report the result as an upper bound (no derate borrowed from Patel et al., ASPLOS 2024, arXiv:2308.12908)")
    ap.add_argument("--kwh-gbp", type=float, default=None, help="Ofgem unit rate GBP/kWh, cite retrieval date (e.g. 0.26)")
    ap.add_argument("--p-gpu-hour", type=float, default=None, help="direct GBP/GPU-hour; overrides watts*kwh")
    ap.add_argument("--p-embed-mtok", type=float, required=True)
    ap.add_argument("--p-in-mtok", type=float, required=True)
    ap.add_argument("--p-out-mtok", type=float, required=True)
    ap.add_argument("--p-store-gb-month", type=float, required=True)
    ap.add_argument("--onetime-gpu-hours", type=float, required=True)
    ap.add_argument("--onetime-embed-mtok", type=float, required=True)
    ap.add_argument("--A", type=float, required=True, help="amortization denominator, must be > 0")
    ap.add_argument("--query-gpu-seconds", type=float, required=True)
    ap.add_argument("--query-embed-mtok", type=float, required=True)
    ap.add_argument("--query-in-mtok", type=float, required=True)
    ap.add_argument("--query-out-mtok", type=float, required=True)
    ap.add_argument("--U", type=float, required=True)
    ap.add_argument("--update-gpu-hours", type=float, required=True)
    ap.add_argument("--update-embed-mtok", type=float, required=True)
    ap.add_argument("--M", type=float, required=True)
    ap.add_argument("--index-gb", type=float, required=True)
    ap.add_argument("--Q", type=float, required=True, help="query volume denominator, must be > 0")
    ap.add_argument("--p-hat", type=float, default=None, help="all-pass rate; omit to print cost only")
    args = ap.parse_args(argv)

    if args.p_gpu_hour is None:
        if args.watts is None or args.kwh_gbp is None:
            ap.error("no GPU price given: pass --p-gpu-hour, or both --watts and --kwh-gbp")
        if args.watts <= 0.0 or args.kwh_gbp <= 0.0:
            ap.error("--watts and --kwh-gbp must be > 0")
    if args.A <= 0.0 or args.Q <= 0.0:
        ap.error("--A and --Q are denominators and must be > 0")
    negative = [name for name, value in vars(args).items() if name != "p_hat" and value is not None and value < 0.0]
    if negative:
        ap.error("negative values not allowed: " + ", ".join(sorted(negative)))

    p_gpu = args.p_gpu_hour if args.p_gpu_hour is not None else electricity_gpu_hour(args.watts, args.kwh_gbp)

    breakdown = cost_per_answer(
        p_gpu_hour=p_gpu,
        p_embed_mtok=args.p_embed_mtok,
        p_in_mtok=args.p_in_mtok,
        p_out_mtok=args.p_out_mtok,
        p_store_gb_month=args.p_store_gb_month,
        onetime_gpu_hours=args.onetime_gpu_hours,
        onetime_embed_mtok=args.onetime_embed_mtok,
        A=args.A,
        query_gpu_seconds=args.query_gpu_seconds,
        query_embed_mtok=args.query_embed_mtok,
        query_in_mtok=args.query_in_mtok,
        query_out_mtok=args.query_out_mtok,
        U=args.U,
        update_gpu_hours=args.update_gpu_hours,
        update_embed_mtok=args.update_embed_mtok,
        M=args.M,
        index_gb=args.index_gb,
        Q=args.Q,
    )

    print(f"p_gpu = £{p_gpu:.6f}/GPU-hour (electricity-metered)")
    for k, v in breakdown.items():
        print(f"{k:>16}: £{v:.6f}")
    if args.p_hat is not None:
        v = cfca(breakdown["Cost_per_answer"], args.p_hat)
        print(f"{'CFCA':>16}: " + (f"£{v:.6f}  (P_hat={args.p_hat})" if v is not None else "undefined (P_hat <= 0)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
