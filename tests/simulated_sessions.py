"""SIMULATED user-study sessions, only for testing the analysis (tests/test_study_analysis.py).

Nothing here is data. A simulated "participant" corrects each study scan by fixing a chosen fraction of the
wrong voxels (voxels where the model's mask differs from the ground truth), picked at random, in a chosen
number of minutes. Every log and questionnaire is marked "simulated": true, the files are only written to a
temporary test folder, and the analysis refuses to write simulated results to results/.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from segreview.study import TLX_SCALES, assignment, study_dir
from segreview.study_analysis import TruthCache


def simulate_sessions(study: dict, source: dict, sessions_dir: Path, participants: list[str], fraction_fixed,
                      minutes, seed: int = 0, tlx=None, tlx_which: tuple = ("session",), truths=None) -> None:
    """Write SIMULATED sessions for the given participant ids into sessions_dir.

    fraction_fixed(pid, condition) -> share (0-1) of the wrong voxels the participant fixes.
    minutes(pid, condition) -> time spent on each scan.
    tlx(pid, which) -> dict of the six scales, for each questionnaire in tlx_which; tlx=None: no questionnaires.
    truths: a TruthCache to reuse (optional).
    """
    root = study_dir(study)
    manifest = json.loads((root / "manifest.json").read_text())
    truths = truths or TruthCache(study, source)
    rng = np.random.default_rng(seed)
    for pid in participants:
        out = sessions_dir / pid
        out.mkdir(parents=True, exist_ok=True)
        for a in assignment(pid, manifest["study_scans"], ("P", "T")):
            model_img = nib.load(root / "scans" / a["case_id"] / "mask.nii.gz")
            model = np.asanyarray(model_img.dataobj) > 0
            truth, _ = truths.get(a["case_id"])
            wrong = np.flatnonzero(model != truth)
            fixed = rng.choice(wrong, size=int(round(fraction_fixed(pid, a["condition"]) * len(wrong))), replace=False)
            corrected = model.ravel().copy()
            corrected[fixed] = truth.ravel()[fixed]
            nib.save(nib.Nifti1Image(corrected.reshape(model.shape).astype(np.uint8), model_img.affine),
                     out / f"{a['case_id']}_mask.nii.gz")
            log = {"simulated": True, "condition": a["condition"], "position": a["position"], "reason": "done",
                   "duration_s": 60.0 * minutes(pid, a["condition"]), "strokes": 0, "undo_count": 0,
                   "heatmap_visible_s": 0.0, "events": []}
            (out / f"{a['case_id']}_log.json").write_text(json.dumps(log))
        if tlx is not None:
            for which in tlx_which:
                scales = tlx(pid, which)
                (out / f"tlx_{which}.json").write_text(json.dumps({
                    "simulated": True, "participant": pid, "which": which, "scales": scales,
                    "raw_tlx": sum(scales[k] for k in TLX_SCALES) / 6}))
