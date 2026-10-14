"""Tests of the user-study analysis (src/segreview/study_analysis.py) on SIMULATED sessions.

The sessions are made by tests/simulated_sessions.py in a temporary folder and marked "simulated". The numbers
checked here say nothing about the real study; they only show that the analysis computes what it should:
Dice before/after on the right grid, gain per minute, pairing per participant, the Wilcoxon test and its sign,
and that simulated output is labeled and kept out of results/.
"""

import json

import numpy as np
import pandas as pd
import pytest

from segreview.config import REPO_ROOT
from segreview.study import WITH, WITHOUT, assignment, study_dir
from segreview.study_analysis import (TruthCache, analyze, conclude, paired_tests, participant_table, scan_table,
                                      signed_rank_p, wilcoxon_paired)
from tests.conftest import manifest
from tests.simulated_sessions import simulate_sessions

TLX = {"mental": 60, "physical": 20, "temporal": 70, "performance": 30, "effort": 65, "frustration": 40}
TLX_WITH = {**TLX, "mental": 40, "frustration": 20}          # simulated: a little less workload with the heatmap


@pytest.fixture(scope="module")
def simulated(real_study, tmp_path_factory):
    """4 SIMULATED participants: they fix 80 % of the wrong voxels with the heatmap and 40 % without, 3 minutes
    per scan. P01 fixes everything with the heatmap (Dice after must then be exactly 1)."""
    import copy
    study, source = real_study
    sessions = tmp_path_factory.mktemp("simulated") / "sessions"
    truths = TruthCache(study, source)
    frac = lambda pid, cond: (1.0 if pid == "P01" else 0.8) if cond == WITH else 0.4
    simulate_sessions(study, source, sessions, ["P01", "P02", "P03", "P04"], frac, lambda pid, cond: 3.0,
                      seed=1, tlx=lambda pid, which: TLX_WITH if which == WITH else TLX, tlx_which=(WITH, WITHOUT),
                      truths=truths)
    simulate_sessions(study, source, sessions, ["T01"], lambda p, c: 0.0, lambda p, c: 1.0, seed=2,
                      truths=truths)                                                          # a test id
    return copy.deepcopy(study), source, sessions, truths


def test_scan_table_from_simulated_sessions(simulated):
    study, source, sessions, truths = simulated
    scans = scan_table(study, source, sessions, truths)
    m = manifest(study)
    assert sorted(scans.participant.unique()) == ["P01", "P02", "P03", "P04"]      # the test id T01 is ignored
    assert len(scans) == 24 and scans.simulated.all()
    for pid, g in scans.groupby("participant"):
        plan = {a["case_id"]: a["condition"] for a in assignment(pid, m["study_scans"], "P")}
        assert dict(zip(g.case_id, g.condition)) == plan
    # Dice before is the model's Dice on the viewer grid; it matches what prepare_study computed on the full scan.
    for r in scans.itertuples():
        assert r.dice_before == pytest.approx(m["facts"]["scans"][r.case_id]["dice_before"], abs=1e-4)
    assert np.allclose(scans.gain_per_min, (scans.dice_after - scans.dice_before) / 3.0)
    assert (scans[(scans.participant == "P01") & (scans.condition == WITH)].dice_after == 1.0).all()
    assert (scans.dice_after > scans.dice_before).all()


def test_analysis_output_is_labeled_simulated(simulated, tmp_path):
    study, source, sessions, truths = simulated
    result = analyze(study, source, sessions, tmp_path / "out", truths)
    files = sorted(p.name for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert files == ["SIMULATED_study_conditions.csv", "SIMULATED_study_exclusions.csv",
                     "SIMULATED_study_gain_per_min.png", "SIMULATED_study_order.csv",
                     "SIMULATED_study_participants.csv", "SIMULATED_study_scans.csv", "SIMULATED_study_tests.csv",
                     "SIMULATED_study_tlx.csv"]
    tests = pd.read_csv(tmp_path / "out" / "SIMULATED_study_tests.csv")
    assert tests.simulated.all() and tests.loc[tests.role == "primary", "outcome"].tolist() == ["gain_per_min"]
    assert tests.set_index("outcome").hypothesis.dropna().to_dict() == {"gain_per_min": "H1", "raw_tlx": "H2"}
    assert (tests.heatmap_display == "full").all() and (result["scans"].heatmap_display == "full").all()
    # Every simulated participant gains more with the heatmap: all differences positive, r = +1.
    primary = result["tests"].set_index("outcome").loc["gain_per_min"]
    assert primary.n == 4 and primary.r_rank_biserial == 1.0 and primary.mean_diff > 0
    assert primary.p == pytest.approx(0.125)       # exact two-sided p with 4 pairs, all positive: 2 / 2**4
    assert primary.ci_low <= primary.mean_diff <= primary.ci_high
    # With 4 participants p cannot go below 0.05: H1 is not supported, so H2 is not tested.
    assert primary.conclusion == "not supported"
    h2 = result["tests"].set_index("outcome").loc["raw_tlx"]
    assert h2.conclusion == "not tested (H1 not supported)" and h2.mean_diff == pytest.approx(-40 / 6)
    assert set(result["tlx"].raw_tlx) == {sum(TLX.values()) / 6, sum(TLX_WITH.values()) / 6}
    assert result["exclusions"].empty and result["scans"].included.all()
    conditions = result["conditions"].set_index("condition")
    assert (conditions.n_scans == 12).all() and (conditions.n_timeout == 0).all() and (conditions.n_unchanged == 0).all()
    assert conditions.loc[WITH, "median_raw_tlx"] == pytest.approx(sum(TLX_WITH.values()) / 6)
    order = result["order"].set_index("heatmap_first")
    assert order.n.sum() == 4 and order.loc[True, "n"] == 2     # types 2 and 3 (P03, P04) start with the heatmap


def test_simulated_results_never_go_to_results_folder(simulated):
    study, source, sessions, _ = simulated
    with pytest.raises(ValueError, match="never be written to results"):
        analyze(study, source, sessions, REPO_ROOT / "results" / "study_kits")


def test_real_and_simulated_sessions_are_not_mixed(simulated, tmp_path):
    import shutil
    study, source, sessions, _ = simulated
    mixed = tmp_path / "sessions"
    shutil.copytree(sessions, mixed)
    log_path = next((mixed / "P02").glob("*_log.json"))
    log = json.loads(log_path.read_text())
    del log["simulated"]                                  # pretend one scan is real
    log_path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="mixed"):
        analyze(study, source, mixed, tmp_path / "out")


def test_wrong_condition_in_a_log_is_caught(simulated, tmp_path):
    import shutil
    study, source, sessions, truths = simulated
    copy_dir = tmp_path / "sessions"
    shutil.copytree(sessions, copy_dir)
    log_path = next((copy_dir / "P03").glob("*_log.json"))
    log = json.loads(log_path.read_text())
    log["condition"] = WITH if log["condition"] == WITHOUT else WITHOUT
    log_path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="differs from the plan"):
        scan_table(study, source, copy_dir, truths)


def test_wilcoxon_sign_and_effect_size():
    """diff = with - without: positive differences mean better with the heatmap."""
    better = wilcoxon_paired(np.arange(1, 13) / 100)       # 12 participants, all better with the heatmap
    assert better["r_rank_biserial"] == 1.0 and better["p"] < 0.001 and better["mean_diff"] > 0
    worse = wilcoxon_paired(-np.arange(1, 13) / 100)
    assert worse["r_rank_biserial"] == -1.0 and worse["p"] == pytest.approx(better["p"])
    balanced = wilcoxon_paired(np.array([1, -1, 2, -2, 3, -3, 4, -4.0]))
    assert balanced["r_rank_biserial"] == 0.0 and balanced["p"] > 0.9
    none = wilcoxon_paired(np.zeros(8))                    # no differences at all
    assert none["n"] == 0 and np.isnan(none["p"])


def test_participant_means_and_pairing():
    """Participant means per condition are taken over that participant's own scans only."""
    scans = pd.DataFrame({
        "participant": ["P01"] * 4 + ["P02"] * 4,
        "condition": [WITH, WITH, WITHOUT, WITHOUT] * 2, "position": [3, 4, 1, 2] * 2,
        "gain_per_min": [0.02, 0.04, 0.01, 0.01, 0.00, 0.02, 0.03, 0.05],
        "dice_after": [.9] * 8, "dice_gain": [.1] * 8, "minutes": [2, 4, 3, 5, 1, 1, 1, 1]})
    part = participant_table(scans).set_index("participant")
    assert part.loc["P01", "gain_per_min_with"] == pytest.approx(0.03)
    assert part.loc["P01", "gain_per_min_diff"] == pytest.approx(0.02)
    assert part.loc["P02", "gain_per_min_diff"] == pytest.approx(-0.03)
    assert part.loc["P01", "minutes_diff"] == pytest.approx(-1.0)
    assert not part.heatmap_first.any()                                  # both started without the heatmap
    tests = paired_tests(part.reset_index()).set_index("outcome")
    assert tests.loc["gain_per_min", "n"] == 2 and tests.loc["dice_after", "n"] == 0   # equal Dice: no pairs left


def test_different_heatmap_displays_are_not_pooled(simulated, tmp_path):
    """Every participant must have seen the heatmap the same way (study.heatmap_display, locked before P01)."""
    import shutil
    study, source, sessions, truths = simulated
    copy_dir = tmp_path / "sessions"
    shutil.copytree(sessions, copy_dir)
    for log_path in (copy_dir / "P02").glob("*_log.json"):
        log = json.loads(log_path.read_text())
        log["heatmap_display"] = {"mode": "top", "top_percent": 10}
        log_path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="different ways"):
        analyze(study, source, copy_dir, tmp_path / "out", truths)
    del log["heatmap_display"]                           # a log without the setting cannot be analyzed either
    log_path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="no heatmap_display"):
        scan_table(study, source, copy_dir, truths)


def test_sign_flip_p_matches_the_exact_wilcoxon_test():
    """Without ties the sign-flip p-value is the textbook exact p-value (scipy's exact method); with ties it is the
    exact permutation p-value (scipy also enumerates all sign patterns up to 13 pairs)."""
    from scipy import stats
    rng = np.random.default_rng(3)
    for n in (6, 9, 16, 20):
        d = rng.normal(0.4, 1, n)
        assert signed_rank_p(d)[0] == pytest.approx(stats.wilcoxon(d, method="exact").pvalue, abs=1e-12)
    ties = np.array([5, 5, 10, -5, 15, 20, 5, 10, -10, 5, 5, 10]) / 6        # TLX-like differences, many ties
    assert signed_rank_p(ties)[0] == pytest.approx(stats.wilcoxon(ties).pvalue, abs=1e-12)
    assert signed_rank_p(ties)[1].startswith("exact")
    many = rng.normal(0.3, 1, 30)                                         # above the exact limit: random patterns
    p, method = signed_rank_p(many, seed=1)
    assert method.startswith("random") and p == pytest.approx(stats.wilcoxon(many, method="exact").pvalue, abs=0.002)
    assert signed_rank_p(many, seed=1)[0] == p                            # the same seed gives the same p


def test_conclusions_follow_the_fixed_sequence():
    """H1 decides whether H2 is tested; the direction comes from the sign of the rank-biserial r."""
    def table(p1, r1, p2, r2):
        return pd.DataFrame({"hypothesis": ["H1", "", "H2"], "expected_sign": [1, 0, -1], "p": [p1, 0.001, p2],
                             "r_rank_biserial": [r1, 0.9, r2]})
    assert list(conclude(table(0.01, 0.7, 0.03, -0.5), 0.05)) == ["supported", "exploratory", "supported"]
    assert list(conclude(table(0.01, 0.7, 0.30, -0.2), 0.05)) == ["supported", "exploratory", "not supported"]
    assert list(conclude(table(0.01, 0.7, 0.01, 0.8), 0.05)) == ["supported", "exploratory", "opposite direction"]
    assert list(conclude(table(0.20, 0.3, 0.001, -0.9), 0.05)) == ["not supported", "exploratory",
                                                                     "not tested (H1 not supported)"]
    assert list(conclude(table(0.01, -0.7, 0.001, -0.9), 0.05)) == ["opposite direction", "exploratory",
                                                                      "not tested (H1 not supported)"]


def test_bootstrap_interval_of_the_mean_difference():
    diff = np.array([0.01, 0.02, -0.005, 0.03, 0.015, 0.0, 0.02, 0.01])
    a = wilcoxon_paired(diff, n_bootstrap=5000, seed=7)
    assert a["ci_low"] < a["mean_diff"] < a["ci_high"] and a["n"] == 7 and a["n_participants"] == 8
    assert a == wilcoxon_paired(diff, n_bootstrap=5000, seed=7)          # reproducible with the seed


def test_incomplete_sessions_timeouts_and_unchanged_scans(simulated, tmp_path):
    """An incomplete session is excluded (and listed); timeouts and unchanged scans stay in and are counted."""
    import shutil
    import nibabel as nib
    study, source, sessions, truths = simulated
    copy_dir = tmp_path / "sessions"
    shutil.copytree(sessions, copy_dir)
    m = json.loads((study_dir(study) / "manifest.json").read_text())
    p03 = assignment("P03", m["study_scans"], "P")
    (copy_dir / "P03" / f"{p03[-1]['case_id']}_log.json").unlink()          # P03 stopped before the last scan
    p02 = assignment("P02", m["study_scans"], "P")
    first, second = p02[0]["case_id"], p02[1]["case_id"]
    model = nib.load(study_dir(study) / "scans" / first / "mask.nii.gz")
    nib.save(model, copy_dir / "P02" / f"{first}_mask.nii.gz")              # P02 saved the AI mask unchanged
    log_path = copy_dir / "P02" / f"{second}_log.json"
    log = json.loads(log_path.read_text())
    log.update(reason="timeout", duration_s=300.2)
    log_path.write_text(json.dumps(log))

    result = analyze(study, source, copy_dir, tmp_path / "out", truths)
    assert result["exclusions"].participant.tolist() == ["P03"]
    assert "5 of 6" in result["exclusions"].reason.iloc[0]
    assert not result["scans"][result["scans"].participant == "P03"].included.any()
    assert result["tests"].set_index("outcome").loc["gain_per_min", "n_participants"] == 3
    assert "P03" not in set(result["tlx"].participant)
    scans = result["scans"].set_index(["participant", "case_id"])
    assert scans.loc[("P02", first), "unchanged"] and scans.loc[("P02", first), "gain_per_min"] == 0
    assert scans.loc[("P02", second), "timeout"] and scans.loc[("P02", second), "minutes"] == pytest.approx(300.2 / 60)
    counts = result["conditions"].set_index("condition")
    assert counts.n_unchanged.sum() == 1 and counts.n_timeout.sum() == 1 and counts.n_scans.sum() == 18
