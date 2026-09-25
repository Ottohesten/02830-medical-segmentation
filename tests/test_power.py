"""Tests of the power simulation for planning the user study (src/segreview/power.py)."""

import numpy as np

from segreview.power import participants_needed, simulate_power


def test_power_behaves_as_expected():
    power = simulate_power([0, 0.8, 1.2], [4, 8, 16], n_simulations=4000, alpha=0.05, seed=0).set_index(
        ["effect_size_dz", "n_participants"]).power
    # With 4 participants the exact two-sided Wilcoxon p is at least 2/16 = 0.125: nothing can be detected.
    assert (power.xs(4, level="n_participants") == 0).all()
    # No effect: the false-positive rate stays at or below alpha (plus Monte Carlo noise).
    assert power.loc[(0, 16)] <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 4000)
    # More participants or a larger effect: more power.
    assert power.loc[(0.8, 8)] < power.loc[(0.8, 16)] and power.loc[(0.8, 16)] < power.loc[(1.2, 16)]


def test_participants_needed():
    power = simulate_power([1.0], [4, 8, 12, 16], n_simulations=2000, alpha=0.05, seed=1)
    needed = participants_needed(power, 0.8)
    assert needed.participants_needed.tolist() == [12]
    assert np.isnan(participants_needed(power, 0.9999).participants_needed.iloc[0])
