#!/usr/bin/env python3
"""Run the calibrated constrained-DCR sweep for the TR-B rebuild."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_REDUCTIONS = "0.35,0.40,0.45,0.50,0.55"
BASE_OUT = Path("results/path3_constrained_dcr_buffer70_sweep")


def _run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reductions", default=DEFAULT_REDUCTIONS)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--router", choices=("haversine", "osrm-grid", "osrm"), default="haversine")
    parser.add_argument("--osrm-grid-size", type=int, default=25)
    parser.add_argument("--base-out", default=str(BASE_OUT))
    args = parser.parse_args()

    base_out = Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    reductions = [float(r.strip()) for r in args.reductions.split(",") if r.strip()]

    for reduction in reductions:
        label = f"{int(round(reduction * 100)):02d}"
        out_dir = base_out / f"red_{label}"
        cmd = [
            sys.executable,
            "scripts/external_baselines_eval.py",
            "--out-dir",
            str(out_dir),
            "--seeds",
            args.seeds,
            "--top-k",
            "5",
            "--share-move-fraction",
            "0.50",
            "--robust-alpha",
            "0.80",
            "--robust-shortage-reduction",
            f"{reduction:.2f}",
            "--robust-mean-slack-weight",
            "0.001",
            "--robust-buffer-json",
            "results/path3_retrieval_buffers_top5/buffers.json",
            "--robust-buffer-alpha",
            "0.70",
            "--methods",
            "spatial_gate_robust_budget",
            "--router",
            args.router,
            "--osrm-grid-size",
            str(args.osrm_grid_size),
        ]
        if args.scenarios:
            cmd.extend(["--scenarios", args.scenarios])
        _run(cmd)

    print(f"\nCompleted calibrated constrained-DCR sweep under {base_out}")


if __name__ == "__main__":
    main()
