#!/usr/bin/env python3
"""Time-boxed contextual-DQN baseline scaffold.

The full Lin-style DQN baseline is intentionally treated as a time-boxed
external baseline rather than a new contribution. This script records the
checkpoint format used by ``ContextualDQNReposition`` and provides a reproducible
warm-start head for the simulator evaluation. Longer RL training can replace the
linear head by writing the same ``q_weights`` array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/path2_contextual_dqn")
    parser.add_argument(
        "--q-weights",
        default="1.6,-1.1,0.7,2.2,0.05,0.05",
        help=(
            "Comma-separated linear Q head over forecast_share, idle_share, "
            "pending_share, shortage, hour_sin, hour_cos."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_weights = np.asarray([float(x) for x in args.q_weights.split(",")], dtype=float)
    if q_weights.shape != (6,):
        raise ValueError("--q-weights must contain exactly six values")

    checkpoint = out_dir / "contextual_dqn_warmstart.npz"
    np.savez(checkpoint, q_weights=q_weights)
    metadata = {
        "checkpoint": str(checkpoint),
        "feature_order": [
            "forecast_share",
            "idle_share",
            "pending_share",
            "shortage",
            "hour_sin",
            "hour_cos",
        ],
        "scope": (
            "Path 2 lightweight contextual-DQN hook. This is a warm-start "
            "shortage-following head, not a fully converged RL contribution."
        ),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
