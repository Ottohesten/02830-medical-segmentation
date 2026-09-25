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
from segreview.study_analysis import (TruthCache, analyze, paired_tests, participant_table, scan_table,
                                      wilcoxon_paired)
from tests.conftest import manifest
from tests.simulated_sessions import simulate_sessions

TLX = {"mental": 60, "physical": 20, "temporal": 70, "performance": 30, "effort": 65, "frustration": 40}


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
                      seed=1, tlx=lambda pid, which: TLX, truths=truths)
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
    assert files == ["SIMULATED_study_gain_per_min.png", "SIMULATED_study_participants.csv",
                     "SIMULATED_study_scans.csv", "SIMULATED_study_tests.csv", "SIMULATED_study_tlx.csv"]
    tests = pd.read_csv(tmp_path / "out" / "SIMULATED_study_tests.csv")
    assert tests.simulated.all() and tests.loc[tests.primary, "outcome"].tolist() == ["gain_per_min"]
    # Every simulated participant gains more with the heatmap: all differences positive, r = +1.
    primary = result["tests"].set_index("outcome").loc["gain_per_min"]
    assert primary.n == 4 and primary.r_rank_biserial == 1.0 and primary.mean_diff > 0
    assert primary.p == pytest.approx(0.125)       # exact two-sided p with 4 pairs, all positive: 2 / 2**4
    assert set(result["tlx"].raw_tlx) == {sum(TLX.values()) / 6}


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
        "condition": [WITH, WITH, WITHOUT, WITHOUT] * 2,
        "gain_per_min": [0.02, 0.04, 0.01, 0.01, 0.00, 0.02, 0.03, 0.05],
        "dice_after": [.9] * 8, "dice_gain": [.1] * 8, "minutes": [2, 4, 3, 5, 1, 1, 1, 1]})
    part = participant_table(scans).set_index("participant")
    assert part.loc["P01", "gain_per_min_with"] == pytest.approx(0.03)
    assert part.loc["P01", "gain_per_min_diff"] == pytest.approx(0.02)
    assert part.loc["P02", "gain_per_min_diff"] == pytest.approx(-0.03)
    assert part.loc["P01", "minutes_diff"] == pytest.approx(-1.0)
    tests = paired_tests(part.reset_index()).set_index("outcome")
    assert tests.loc["gain_per_min", "n"] == 2 and tests.loc["dice_after", "n"] == 0   # equal Dice: no pairs left
