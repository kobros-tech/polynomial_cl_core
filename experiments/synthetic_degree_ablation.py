# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Paired-seed ablation on the binary sinusoid task: does growing the polynomial degree help?

Within each seed the three conditions share the same generated experiences, replay budget,
epochs and optimiser, and differ only in the degree schedule:

* ``expanding``:            degree 1, 2, ..., n_experiences (zero-initialised new coefficients).
* ``fixed_initial_degree``: degree 1 throughout; the literal no-expansion baseline.
* ``fixed_capacity``:       the final degree from experience 1 (same final capacity; isolates
                            *when* capacity becomes available rather than how much there is).

expanding vs fixed_initial_degree measures the value of adding capacity at all;
expanding vs fixed_capacity measures the value of adding it gradually.
"""

import argparse
import json
from pathlib import Path

from polynomial_cl.binary import make_config, make_experiences, run_continual
from polynomial_cl.metrics import mean_std_by_condition

METRICS = ("final_mean_accuracy", "final_current_accuracy", "final_mean_forgetting")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-experiences", type=int, default=10)
    p.add_argument(
        "--fixed-degree",
        type=int,
        default=None,
        help="degree of the fixed_capacity condition (default: n-experiences, i.e. the final\n"
        "degree of the expanding schedule)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--m-current", type=int, default=100)
    p.add_argument("--m-replay", type=int, default=100)
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument(
        "--output", type=Path, default=Path("results/synthetic_degree_ablation.json")
    )
    a = p.parse_args()
    if not a.seeds or a.n_experiences < 1 or a.epochs < 0:
        p.error("need at least one seed, n-experiences >= 1 and epochs >= 0")
    fixed_degree = a.n_experiences if a.fixed_degree is None else a.fixed_degree
    if fixed_degree < 1:
        p.error("fixed-degree must be positive")

    schedules = {
        "expanding": None,  # default schedule: 1, 2, ..., n_experiences
        "fixed_initial_degree": [1] * a.n_experiences,
        "fixed_capacity": [fixed_degree] * a.n_experiences,
    }
    rows = []
    for seed in a.seeds:
        experiences = make_experiences(n_experiences=a.n_experiences, seed=seed)
        for condition, degrees in schedules.items():
            config = make_config(
                "ordinary",
                seed=seed,
                epochs=a.epochs,
                m_current=a.m_current,
                m_replay=a.m_replay,
                capacity=a.capacity,
                degrees=degrees,
            )
            last = run_continual(experiences, config).records[-1]
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "final_mean_accuracy": float(last.mean_acc),
                    "final_current_accuracy": float(last.acc_current),
                    "final_mean_forgetting": float(last.mean_forgetting),
                }
            )
            print(
                f"seed={seed} condition={condition} "
                f"final_mean_accuracy={rows[-1]['final_mean_accuracy']:.4f} "
                f"final_mean_forgetting={rows[-1]['final_mean_forgetting']:.4f}"
            )

    summary = mean_std_by_condition(rows, METRICS)
    payload = {
        "protocol": {
            "seeds": a.seeds,
            "n_experiences": a.n_experiences,
            "fixed_degree": fixed_degree,
            "epochs": a.epochs,
            "m_current": a.m_current,
            "m_replay_per_old_experience": a.m_replay,
            "capacity_per_experience": a.capacity,
        },
        "runs": rows,
        "summary": summary,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
