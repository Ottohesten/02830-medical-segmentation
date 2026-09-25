"""Tests of the study design: balancing of scans and conditions, and the scan selection."""

from collections import Counter

import pytest

from segreview.study import WITH, WITHOUT, assignment, participant_type

SCANS = ["s1", "s2", "s3", "s4", "s5", "s6"]


def test_first_two_participants_match_the_study_plan():
    # Person 1: scans 1-3 without, 4-6 with. Person 2: the other way round.
    p1 = {a["case_id"]: a["condition"] for a in assignment("P01", SCANS, "P")}
    p2 = {a["case_id"]: a["condition"] for a in assignment("P02", SCANS, "P")}
    assert [p1[s] for s in SCANS] == [WITHOUT] * 3 + [WITH] * 3
    assert [p2[s] for s in SCANS] == [WITH] * 3 + [WITHOUT] * 3


@pytest.mark.parametrize("n_participants", [4, 8, 12])
def test_balancing_over_multiples_of_four(n_participants):
    ids = [f"P{i:02d}" for i in range(1, n_participants + 1)]
    per_scan = Counter()
    first_condition = Counter()
    for pid in ids:
        a = assignment(pid, SCANS, "P")
        assert sorted(x["case_id"] for x in a) == sorted(SCANS)          # every scan exactly once
        assert Counter(x["condition"] for x in a) == {WITH: 3, WITHOUT: 3}
        assert [x["position"] for x in a] == [1, 2, 3, 4, 5, 6]
        # conditions come in two blocks, not mixed
        conds = [x["condition"] for x in a]
        assert conds[:3] == [conds[0]] * 3 and conds[3:] == [conds[3]] * 3 and conds[0] != conds[3]
        first_condition[conds[0]] += 1
        for x in a:
            per_scan[(x["case_id"], x["condition"])] += 1
    half = n_participants // 2
    assert all(per_scan[(s, c)] == half for s in SCANS for c in (WITH, WITHOUT))  # each scan with/without equally often
    assert first_condition[WITH] == first_condition[WITHOUT] == half              # order balanced


@pytest.mark.parametrize("bad", ["", "P", "X01", "P0", "Pab", "01"])
def test_bad_participant_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        participant_type(bad, "P")


def test_selected_scans_meet_the_criteria(real_study):
    import pandas as pd
    from segreview.study import study_dir
    study, _ = real_study
    root = study_dir(study)
    from tests.conftest import manifest
    m = manifest(study)
    table = pd.read_csv(root / "candidates.csv").set_index("case_id")
    sel = study["selection"]
    chosen = m["study_scans"] + m["practice_scans"]
    assert len(m["study_scans"]) == sel["n_scans"] and len(set(chosen)) == len(chosen)
    for cid in chosen:
        r = table.loc[cid]
        assert sel["dice_min"] <= r.dice <= sel["dice_max"]
        assert sel["error_ml_min"] <= r.error_ml <= sel["error_ml_max"]
        assert r.axial_slices <= sel["max_axial_slices"] and r.n_sides_found == 2
        assert m["facts"]["scans"][cid]["gt_kept"] >= sel["min_gt_in_crop"]


def test_test_ids_get_the_same_balancing_as_participant_ids():
    for n in (1, 2, 3, 4, 7):
        assert assignment(f"T{n:02d}", SCANS, ("P", "T")) == assignment(f"P{n:02d}", SCANS, "P")
    with pytest.raises(ValueError):
        participant_type("T01", "P")              # a test id is only accepted where test ids are allowed


def test_clean_test_data_deletes_only_test_ids(temp_study):
    from segreview.study import clean_test_data, study_dir
    root = study_dir(temp_study)
    folders = ["P01", "P12", "T01", "T15", "TX", "T01a", "t02", "PT01", "demo"]
    for name in folders:
        (root / "sessions" / name).mkdir(parents=True)
        (root / "sessions" / name / "case_00001_log.json").write_text("{}")
    (root / "queue_edits").mkdir()
    (root / "queue_edits" / "case_00001_mask.nii.gz").write_bytes(b"x")

    listed = clean_test_data(temp_study, dry_run=True)
    assert all(p.exists() for p in listed)                                  # a dry run deletes nothing
    deleted = clean_test_data(temp_study)
    assert sorted(p.name for p in deleted) == ["T01", "T15", "queue_edits"]
    left = sorted(p.name for p in (root / "sessions").iterdir())
    assert left == sorted(set(folders) - {"T01", "T15"})
    assert (root / "sessions" / "P01" / "case_00001_log.json").exists()     # participant data untouched
    assert (root / "scans").exists() and (root / "manifest.json").exists()


def test_prefixes_that_could_overlap_are_refused(temp_study):
    from segreview.study import clean_test_data
    for participant, test in (("P", "P"), ("P", "PT"), ("TP", "T"), ("P", "")):
        temp_study["study"]["participant_prefix"], temp_study["study"]["test_prefix"] = participant, test
        with pytest.raises(ValueError):
            clean_test_data(temp_study, dry_run=True)
