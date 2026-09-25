"""How many participants does the user study (G2) need? A Monte Carlo power simulation for planning.

The study compares, per participant, the mean Dice gain per minute WITH the heatmap and WITHOUT it (3 scans
each), with a Wilcoxon signed-rank test on the per-participant differences (see study_analysis.py).
The effect is described as a standardised effect size

    d_z = mean of the per-participant differences / standard deviation of those differences.

We do not know d_z yet. After the pilot it can be estimated, e.g. an expected difference of 0.008 Dice per
minute with a spread of the per-participant differences of 0.010 gives d_z = 0.8.

Method: for each assumed d_z and number of participants n (a multiple of 4, because of the balancing), draw
n differences from a normal distribution with mean d_z and standard deviation 1, run the same two-sided
Wilcoxon test as the analysis, and repeat many times. The power is the share of repetitions with p < alpha.
d_z = 0 is included as a check: its "power" is the false-positive rate and must stay at or below alpha.

These are ASSUMED effect sizes, not study results.
"""

import numpy as np
import pandas as pd
from scipy import stats


def simulate_power(effect_sizes: list[float], n_participants: list[int], n_simulations: int, alpha: float,
                   seed: int) -> pd.DataFrame:
    """Power of the two-sided Wilcoxon signed-rank test for each (d_z, n).

    Output: one row per combination: effect_size_dz, n_participants, power, n_simulations, alpha.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in effect_sizes:
        for n in n_participants:
            diffs = rng.normal(d, 1.0, size=(n_simulations, n))
            p = stats.wilcoxon(diffs, axis=1, alternative="two-sided").pvalue
            rows.append({"effect_size_dz": d, "n_participants": n, "power": float(np.mean(p < alpha)),
                         "n_simulations": n_simulations, "alpha": alpha})
    return pd.DataFrame(rows)


def participants_needed(power: pd.DataFrame, target: float) -> pd.DataFrame:
    """Smallest simulated number of participants that reaches the target power, per effect size (NaN if none)."""
    rows = []
    for d, g in power[power.effect_size_dz > 0].groupby("effect_size_dz"):
        ok = g[g.power >= target].n_participants
        rows.append({"effect_size_dz": d, "target_power": target,
                     "participants_needed": int(ok.min()) if len(ok) else np.nan})
    return pd.DataFrame(rows)
