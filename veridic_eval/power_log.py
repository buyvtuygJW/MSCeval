"""
Concurrent GPU power poller for electricity pricing.

Run alongside any timed stage (indexing, fine-tuning, serving) on either
machine (DGX Spark / Omen 4070). Polls `nvidia-smi --query-gpu=power.draw`
and writes a CSV of (iso_ts, watts); on exit (Ctrl+C or --duration-s) it
prints and writes a summary with mean/peak watts and energy in Wh.

Every tunable is an explicit flag; no hidden assumptions:
  --interval-s        poll cadence (default 1.0 s)
  --duration-s        stop after N seconds (default: run until Ctrl+C)
  --out               CSV path (default power_log.csv; summary at <out>.summary.json)
  --gpu-index         nvidia-smi -i index (default 0)
  --smi               nvidia-smi executable (default "nvidia-smi" on PATH)
  --tag               free-text label written into the summary (e.g. "index_c3_spark")
  --tdp-fallback-w    watts used for energy ONLY when polling returns no valid
                      sample (e.g. unsupported device): pass 1.0*TDP (full
                      nameplate) for ANY job and report the result as an upper
                      bound (no derate is borrowed
                      from Patel et al., ASPLOS 2024, arXiv:2308.12908);
                      default None = no fallback, summary marks energy unknown.

Usage:
  veridic-eval power --tag finetune_c2 --out ft_c2_power.csv
  veridic-eval power --duration-s 600 --gpu-index 0 --tdp-fallback-w 65  # 1.0*TDP for the 4070 laptop board, upper bound
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone


def read_power_w(smi: str, gpu_index: int) -> float:
    """One power.draw sample in watts; NaN when unsupported/unparseable."""
    try:
        out = subprocess.run(
            [smi, f"--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def summarise(samples: list, elapsed_s: float, tag: str,
              tdp_fallback_w) -> dict:
    valid = [w for _, w in samples if not math.isnan(w)]
    mean_w = (sum(valid) / len(valid)) if valid else None
    peak_w = max(valid) if valid else None
    if mean_w is not None:
        energy_wh = mean_w * elapsed_s / 3600.0
        energy_basis = "measured (nvidia-smi mean W x wall-clock)"
    elif tdp_fallback_w is not None:
        energy_wh = float(tdp_fallback_w) * elapsed_s / 3600.0
        energy_basis = f"fallback TDP {tdp_fallback_w} W x wall-clock (no valid samples; upper bound)"
    else:
        energy_wh = None
        energy_basis = "unknown (no valid samples, no --tdp-fallback-w)"
    return {
        "tag": tag,
        "samples": len(samples),
        "valid_samples": len(valid),
        "elapsed_s": round(elapsed_s, 3),
        "mean_w": round(mean_w, 3) if mean_w is not None else None,
        "peak_w": round(peak_w, 3) if peak_w is not None else None,
        "energy_wh": round(energy_wh, 6) if energy_wh is not None else None,
        "energy_basis": energy_basis,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="veridic-eval power", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--out", default="power_log.csv")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--smi", default="nvidia-smi")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tdp-fallback-w", type=float, default=None)
    args = ap.parse_args(argv)

    samples = []
    t0 = time.monotonic()
    print(f"power_log: polling {args.smi} -i {args.gpu_index} every "
          f"{args.interval_s}s -> {args.out}  (Ctrl+C to stop)", flush=True)
    try:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["iso_ts", "watts"])
            while True:
                watts = read_power_w(args.smi, args.gpu_index)
                ts = datetime.now(timezone.utc).isoformat()
                samples.append((ts, watts))
                w.writerow([ts, "" if math.isnan(watts) else watts])
                fh.flush()
                if args.duration_s is not None and time.monotonic() - t0 >= args.duration_s:
                    break
                time.sleep(args.interval_s)
    except KeyboardInterrupt:
        pass

    summary = summarise(samples, time.monotonic() - t0, args.tag,
                        args.tdp_fallback_w)
    with open(args.out + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
