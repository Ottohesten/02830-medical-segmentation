"""End-to-end test of the review page in a real browser (headless Google Chrome), without clicking.

The page is opened with ?selftest=1: it loads the first demo-queue scan in NiiVue, paints a 10 x 10 square
into the middle slice of the drawing and saves it through the normal save path. The test then checks that
the saved mask equals the original mask plus exactly that square, on the scan's own grid. That covers
loading, the voxel orientation between file, NiiVue and server, and saving.
Skipped if Chrome is not installed.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from segreview.study import study_dir
from tests.conftest import manifest

CHROME_CANDIDATES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


@pytest.mark.skipif(find_chrome() is None, reason="Google Chrome not installed")
def test_paint_and_save_in_browser(server, tmp_path):
    base, study = server
    root = study_dir(study)
    case_id = manifest(study)["queue_scans"][0]
    cmd = [find_chrome(), "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--no-first-run", f"--user-data-dir={tmp_path / 'chrome'}", "--remote-debugging-port=0",
           f"{base}/?selftest=1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log_file = root / "queue_edits" / f"{case_id}_log.json"
    try:
        t0 = time.time()
        while time.time() - t0 < 180 and not log_file.exists():
            time.sleep(1)
    finally:
        proc.kill()
    assert log_file.exists(), "the page did not save within 180 s"

    ct = nib.load(root / "queue" / case_id / "ct.nii.gz")
    original = np.asanyarray(nib.load(root / "queue" / case_id / "mask.nii.gz").dataobj) > 0
    saved_img = nib.load(root / "queue_edits" / f"{case_id}_mask.nii.gz")
    saved = np.asanyarray(saved_img.dataobj) > 0
    nx, ny, nz = ct.shape
    expected = original.copy()
    expected[nx // 2 - 5:nx // 2 + 5, ny // 2 - 5:ny // 2 + 5, nz // 2] = True   # same square as in ui/app.js
    assert saved.shape == original.shape and np.allclose(saved_img.affine, ct.affine)
    assert np.array_equal(saved, expected)
    log = json.loads(log_file.read_text())
    assert log["reason"] == "selftest" and log["heatmap_available"] and log["events"][-1]["type"] == "end"
