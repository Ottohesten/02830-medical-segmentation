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
    test_plan = json.loads(get(base + "/api/study/T01")[1])         # test ids work and get the same order
    assert [s["case_id"] for s in test_plan["scans"]] == [s["case_id"] for s in plan["scans"]]


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


GOOD_TLX = {"mental": 55, "physical": 10, "temporal": 70, "performance": 25, "effort": 60, "frustration": 30}


def test_tlx_after_session_is_saved_with_the_raw_score(temp_study):
    from tests.conftest import running_server
    temp_study["study"]["tlx"] = "after_session"
    with running_server(temp_study) as base:
        plan = json.loads(get(base + "/api/study/P05")[1])
        assert [t["which"] for t in plan["tlx"]] == ["session"] and not plan["tlx"][0]["done"]
        assert plan["tlx"][0]["after_position"] == len(plan["scans"])
        code, reply = post(f"{base}/api/tlx/P05/session", json.dumps({"scales": GOOD_TLX}).encode(), "application/json")
        assert code == 200 and reply["raw_tlx"] == pytest.approx(sum(GOOD_TLX.values()) / 6)
        saved = json.loads((study_dir(temp_study) / "sessions" / "P05" / "tlx_session.json").read_text())
        assert saved["scales"] == GOOD_TLX and saved["raw_tlx"] == pytest.approx(41.6667, abs=1e-3)
        assert json.loads(get(base + "/api/study/P05")[1])["tlx"][0]["done"]


# The study config asks the TLX after each block (the default), so "with_heatmap" is a valid questionnaire
# and "session" is not.
@pytest.mark.parametrize("scales, which, pid", [
    ({**GOOD_TLX, "mental": 7}, WITH, "P05"),             # not a step of 5
    ({**GOOD_TLX, "effort": 105}, WITH, "P05"),           # out of range
    ({k: v for k, v in GOOD_TLX.items() if k != "effort"}, WITH, "P05"),   # a scale missing
    (GOOD_TLX, "session", "P05"),                         # not a questionnaire of this study (after_each_block)
    (GOOD_TLX, WITH, "X05"),                              # not a valid id
])
def test_bad_tlx_is_rejected(server, scales, which, pid):
    base, study = server
    code, _ = post(f"{base}/api/tlx/{pid}/{which}", json.dumps({"scales": scales}).encode(), "application/json")
    assert code == 400
    assert not (study_dir(study) / "sessions" / pid).exists()


def test_tlx_after_each_block_is_the_default(temp_study):
    from segreview.study import tlx_schedule
    from tests.conftest import running_server
    assert temp_study["study"]["tlx"] == "after_each_block"
    m = manifest(temp_study)
    with running_server(temp_study) as base:
        plan = json.loads(get(base + "/api/study/P02")[1])
        # One questionnaire after each block, named after its condition.
        conditions = [s["condition"] for s in plan["scans"]]
        assert [(t["which"], t["after_position"]) for t in plan["tlx"]] == [(conditions[0], 3), (conditions[3], 6)]
        code, _ = post(f"{base}/api/tlx/P02/{WITH}", json.dumps({"scales": GOOD_TLX}).encode(), "application/json")
        assert code == 200
    assert tlx_schedule(temp_study, assignment("P02", m["study_scans"], "P")) == [
        {"which": t["which"], "after_position": t["after_position"]} for t in plan["tlx"]]


def test_tlx_scale_names_match_the_page():
    """The page (ui/app.js) and the server must use the same six scale keys."""
    import re
    from segreview.config import REPO_ROOT
    from segreview.study import TLX_SCALES
    js = (REPO_ROOT / "ui" / "app.js").read_text()
    block = js[js.index("const TLX = ["):js.index("];", js.index("const TLX = ["))]
    assert re.findall(r'^\s*\["(\w+)"', block, flags=re.M) == TLX_SCALES


# ---------- heatmap display modes and the session record ----------

def served_heatmap(base, participant, file_set, case_id):
    """The heatmap as the server hands it out, read like NiiVue reads it (values 0-1)."""
    data = get(f"{base}/files/{participant}/{file_set}/{case_id}/heatmap.nii.gz")[1]
    return nib.Nifti1Image.from_bytes(gzip.decompress(data)).get_fdata(dtype=np.float32)


def test_full_display_serves_the_prepared_heatmap_unchanged(server):
    base, study = server
    m = manifest(study)
    assert json.loads(get(base + "/api/settings")[1])["heatmap_display"]["mode"] == "full"
    cid = m["practice_scans"][0]
    assert get(f"{base}/files/P01/scans/{cid}/heatmap.nii.gz")[1] == \
        (study_dir(study) / "scans" / cid / "heatmap.nii.gz").read_bytes()


@pytest.mark.parametrize("mode", ["top", "contour"])
def test_top_and_contour_displays_are_served(temp_study, mode):
    from segreview.heatmap_display import check_display, display_map, top_voxels
    from segreview.uncertainty import uncertainty_region
    from tests.conftest import running_server
    temp_study["study"]["heatmap_display"] = {"mode": mode, "top_percent": 10, "contour_margin_mm": 3}
    m = manifest(temp_study)
    folder = study_dir(temp_study) / "scans" / m["practice_scans"][0]
    heat_img, mask_img = nib.load(folder / "heatmap.nii.gz"), nib.load(folder / "mask.nii.gz")
    heat, mask = heat_img.get_fdata(dtype=np.float32), np.asanyarray(mask_img.dataobj) > 0
    spacing = heat_img.header.get_zooms()[:3]
    with running_server(temp_study) as base:
        assert json.loads(get(base + "/api/settings")[1])["heatmap_display"]["mode"] == mode
        shown = served_heatmap(base, "P01", "scans", m["practice_scans"][0])
        # Still no heatmap at all in the without-heatmap condition.
        without = next(a for a in assignment("P01", m["study_scans"], "P") if a["condition"] == WITHOUT)
        assert status_of(f"{base}/files/P01/scans/{without['case_id']}/heatmap.nii.gz") == 403
    expected = display_map(heat, mask, spacing, 5, check_display(temp_study["study"]["heatmap_display"]))
    assert np.allclose(shown, expected, atol=0.5 / 255)
    region = uncertainty_region(mask, spacing, 5)
    kept = top_voxels(heat, mask, spacing, 5, 10)
    assert 0.10 <= kept.sum() / region.sum() < 0.11             # about 10 % of the region (ties may add a little)
    if mode == "top":
        assert np.array_equal(shown > 0, kept) and np.allclose(shown[kept], heat[kept], atol=0.5 / 255)
    else:
        assert shown.any() and set(np.unique(np.round(shown, 3))) == {0.0, 1.0}
        assert not (shown.astype(bool) & kept).any()             # the uncertain voxels themselves stay clear


def test_bad_display_setting_stops_the_server(temp_study):
    from segreview.ui_server import make_server
    temp_study["study"]["heatmap_display"] = {"mode": "brightest"}
    with pytest.raises(ValueError, match="heatmap_display.mode"):
        make_server(temp_study, host="127.0.0.1", port=0)


def test_session_records_the_settings_and_refuses_a_changed_display(temp_study):
    from tests.conftest import running_server
    path = study_dir(temp_study) / "sessions" / "P04" / "session.json"
    with running_server(temp_study) as base:
        code, record = post(f"{base}/api/session/P04", b"")
        assert code == 200 and record["settings"]["heatmap_display"]["mode"] == "full"
        assert record["settings"]["heatmap"] == "entropy" and record["settings"]["tlx"] == "after_each_block"
        assert post(f"{base}/api/session/P04", b"")[0] == 200            # resumed with the same settings
        assert len(json.loads(path.read_text())["starts"]) == 2
        assert post(f"{base}/api/session/Q04", b"")[0] == 400
        assert post(f"{base}/api/session/T04", b"")[0] == 200            # test ids get a record too
        # The server adds the display to every scan log, whatever the page sends.
        cid = assignment("P04", manifest(temp_study)["study_scans"], "P")[0]["case_id"]
        post(f"{base}/api/log/study/P04/{cid}", json.dumps({"reason": "done", "heatmap_display": "fake"}).encode(),
             "application/json")
        log = json.loads((path.parent / f"{cid}_log.json").read_text())
        assert log["heatmap_display"] == record["settings"]["heatmap_display"]
    before = path.read_text()
    temp_study["study"]["heatmap_display"] = {"mode": "top"}
    with running_server(temp_study) as base:
        code, reply = post(f"{base}/api/session/P04", b"")
        assert code == 409 and "started with" in reply["error"]
        assert post(f"{base}/api/session/P08", b"")[0] == 200            # a new participant can start
    assert path.read_text() == before
