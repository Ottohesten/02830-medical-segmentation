"""Power simulation for the user study (G2): how many participants for a given (assumed) effect size?

Settings in configs/study.yaml under "power". Writes results/<study name>/power_simulation.csv,
power_needed.csv and figures/power_simulation.png. The effect sizes are assumptions for planning, not results.

Usage: uv run python scripts/power_study.py --config configs/study.yaml
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segreview.config import REPO_ROOT
from segreview.power import participants_needed, simulate_power
from segreview.study import load_study_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    args = parser.parse_args()
    study, _ = load_study_config(args.config)
    cfg = study["power"]
    out = REPO_ROOT / "results" / study["name"]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    power = simulate_power(cfg["effect_sizes_dz"], cfg["n_participants"], cfg["n_simulations"], cfg["alpha"],
                           study["seed"])
    needed = participants_needed(power, cfg["target_power"])
    power.to_csv(out / "power_simulation.csv", index=False)
    needed.to_csv(out / "power_needed.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for d, g in power.groupby("effect_size_dz"):
        ax.plot(g.n_participants, g.power, marker="o", ms=3, lw=1.2, ls="--" if d == 0 else "-",
                color="0.6" if d == 0 else None, label=f"$d_z$ = {d:g}" if d else "no effect ($d_z$ = 0)")
    ax.axhline(cfg["target_power"], color="k", lw=.7, ls=":")
    ax.set_xlabel("participants (multiple of 4)")
    ax.set_ylabel(f"power (Wilcoxon, $\\alpha$ = {cfg['alpha']:g})")
    ax.set_xticks(cfg["n_participants"])
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Assumed effect sizes (simulation)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "power_simulation.png", dpi=200)
    print(needed.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
