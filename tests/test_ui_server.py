"""Tests of the review server: files, heatmap and ground-truth access, saving masks and logs, resuming."""

import gzip
import json
import urllib.error
import urllib.request

import nibabel as nib
import numpy as np
import pytest

from segreview.study import WITH, WITHOUT, assignment, study_dir
from tests.conftest import manifest


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, body, content_type="application/octet-stream"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def status_of(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_page_settings_and_niivue_are_served(server):
    base, study = server
    assert b"niivue.umd.js" in get(base + "/")[1]
    assert get(base + "/ui/vendor/niivue/niivue.umd.js")[0] == 200
    settings = json.loads(get(base + "/api/settings")[1])
    assert settings["time_limit_min"] == study["study"]["time_limit_min"]
    assert status_of(base + "/ui/../configs/study.yaml") in (403, 404)


def test_plan_follows_balancing(server):
    base, study = server
    m = manifest(study)
    plan = json.loads(get(base + "/api/study/P01")[1])
    assert [s["case_id"] for s in plan["scans"]] == [a["case_id"] for a in assignment("P01", m["study_scans"], "P")]
    assert not any(s["done"] for s in plan["scans"] + plan["practice"])
    assert status_of(base + "/api/study/nonsense") == 400


def test_heatmap_only_in_with_condition(server):
    base, study = server
    m = manifest(study)
    for a in assignment("P01", m["study_scans"], "P"):
        url = f"{base}/files/P01/scans/{a['case_id']}/heatmap.nii.gz"
        assert status_of(url) == (200 if a["condition"] == WITH else 403)
        assert status_of(f"{base}/files/P01/scans/{a['case_id']}/ct.nii.gz") == 200
    assert status_of(f"{base}/files/P01/scans/{m['practice_scans'][0]}/heatmap.nii.gz") == 200
    assert status_of(f"{base}/files/demo/queue/{m['queue_scans'][0]}/heatmap.nii.gz") == 200
    assert status_of(f"{base}/files/P01/scans/case_99999/ct.nii.gz") == 404
    assert status_of(f"{base}/files/P01/scans/{m['study_scans'][0]}/secret.nii.gz") == 404


def test_ground_truth_only_for_practice_and_queue(server):
    """The answer is available after the practice and in the demo, never for a scan that is measured."""
    base, study = server
    m = manifest(study)
    for cid in m["study_scans"]:
        assert status_of(f"{base}/files/P01/scans/{cid}/truth.nii.gz") == 403
        assert not (study_dir(study) / "scans" / cid / "truth.nii.gz").exists()
    assert status_of(f"{base}/files/P01/scans/{m['practice_scans'][0]}/truth.nii.gz") == 200
    assert status_of(f"{base}/files/demo/queue/{m['queue_scans'][0]}/truth.nii.gz") == 200


def test_guide_image_is_served(server):
    base, _ = server
    status, body = get(base + "/files/guide/example.png")
    assert status == 200 and body[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(get(base + "/api/settings")[1])["guide_image"] is True
    assert status_of(base + "/files/guide/other.png") == 404


def test_saved_files_have_the_scan_grid(server):
    """ct, mask, heatmap (and truth) of every prepared scan share one voxel grid (so the drawing lines up)."""
    _, study = server
    m = manifest(study)
    root = study_dir(study)
    for sub, ids in (("scans", m["study_scans"] + m["practice_scans"]), ("queue", m["queue_scans"])):
        for cid in ids:
            names = ["ct", "mask", "heatmap"] + (["truth"] if cid in m["practice_scans"] or sub == "queue" else [])
            imgs = [nib.load(root / sub / cid / f"{n}.nii.gz") for n in names]
            assert len({i.shape for i in imgs}) == 1
            assert all(np.allclose(imgs[0].affine, i.affine) for i in imgs)


def test_save_log_and_resume(server):
    base, study = server
    m = manifest(study)
    root = study_dir(study)
    first = assignment("P03", m["study_scans"], "P")[0]["case_id"]
    original = nib.load(root / "scans" / first / "mask.nii.gz")
    edited = np.asanyarray(original.dataobj).copy()
    edited[:5, :5, :2] = 1
    raw = nib.Nifti1Image(edited, original.affine).to_bytes()          # uncompressed, as NiiVue sends it

    code, reply = post(f"{base}/api/save/study/P03/{first}", raw)
    assert code == 200, reply
    saved = nib.load(root / "sessions" / "P03" / f"{first}_mask.nii.gz")
    assert np.array_equal(np.asanyarray(saved.dataobj), edited)

    code, _ = post(f"{base}/api/log/study/P03/{first}", json.dumps({"reason": "done", "duration_s": 12.3}).encode(),
                   "application/json")
    assert code == 200
    log = json.loads((root / "sessions" / "P03" / f"{first}_log.json").read_text())
    assert log["participant"] == "P03" and log["case_id"] == first and log["mode"] == "study"

    plan = json.loads(get(base + "/api/study/P03")[1])
    assert [s["done"] for s in plan["scans"]] == [True, False, False, False, False, False]


def test_bad_uploads_are_rejected(server):
    base, study = server
    m = manifest(study)
    cid = m["study_scans"][0]
    wrong_shape = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)).to_bytes()
    assert post(f"{base}/api/save/study/P01/{cid}", wrong_shape)[0] == 400
    assert post(f"{base}/api/save/study/P01/{cid}", b"not a nifti")[0] == 400
    assert post(f"{base}/api/save/study/P01/case_99999", gzip.compress(b"x"))[0] == 404
    assert post(f"{base}/api/save/study/Z01/{cid}", b"x")[0] == 400
    assert not (study_dir(study) / "sessions" / "P01" / f"{cid}_mask.nii.gz").exists()
