"""User study (G2): choosing the study scans, preparing their files, and assigning scans to participants.

Pipeline step 5 (guided correction). The review interface (ui/, served by ui_server.py) shows each
participant a CT scan with the model's kidney mask and lets them fix it with a brush, with or without
the uncertainty heatmap.

Balancing (from the project plan): the 6 study scans are split into two blocks, A = scans 1-3 and B = scans 4-6.
Every participant corrects all 6 scans, one block WITH the heatmap and the other WITHOUT, so nobody sees
a scan twice. Two things are swapped between participants:
- which block gets the heatmap (A with / B without, or the other way round), so every scan is corrected
  with the heatmap by half of the participants and without it by the other half;
- which condition comes first, so learning and tiredness affect both conditions equally.
That gives 4 participant types, cycled by participant number (P01 -> type 0, P02 -> type 1, ...).
With a multiple of 4 participants every combination occurs equally often.

Test runs (trying the interface, demos) use ids with the test prefix instead, e.g. T01. They get the same
balancing, but they are never analysed, and scripts/clean_test_data.py can delete them. Real participant ids
(P01, ...) can never be deleted by that script.
"""

import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from segreview.config import REPO_ROOT, load_config, results_dir
from segreview.data import list_cases, load_canonical, load_organ_mask
from segreview.guide import make_guide_image, pick_guide_slice
from segreview.io import QUANT_LEVELS, load_map, voxel_spacing
from segreview.metrics import dice

WITH, WITHOUT = "with_heatmap", "without_heatmap"
GUIDE_IMAGE = "example.png"

# NASA-TLX, raw version (Raw TLX): six scales from 0 to 100 in steps of 5, no pairwise weighting.
# The score is the mean of the six. For "performance" 0 = perfect and 100 = failure, so for every scale a
# higher value means a higher workload.
TLX_SCALES = ["mental", "physical", "temporal", "performance", "effort", "frustration"]
TLX_MODES = ("after_session", "after_each_block")


def tlx_schedule(study: dict, scans: list[dict]) -> list[dict]:
    """When a participant fills in the NASA-TLX (study.tlx in the config).

    after_session:    once, after the last study scan ("which" = "session").
    after_each_block: after the last scan of each block, i.e. once per condition ("which" = the condition),
                      so the workload WITH and WITHOUT the heatmap can be compared per participant.
    Input: study config, the participant's scans in order (from assignment()).
    Output: list of {"which", "after_position"}.
    """
    mode = study["study"]["tlx"]
    if mode not in TLX_MODES:
        raise ValueError(f"study.tlx must be one of {TLX_MODES}, not '{mode}'")
    if mode == "after_session":
        return [{"which": "session", "after_position": scans[-1]["position"]}]
    ends = [s for i, s in enumerate(scans) if i + 1 == len(scans) or scans[i + 1]["condition"] != s["condition"]]
    return [{"which": s["condition"], "after_position": s["position"]} for s in ends]


def raw_tlx(scales: dict) -> float:
    """Raw TLX score: the mean of the six scales (0-100). Checks that every scale is 0..100 in steps of 5."""
    if sorted(scales) != sorted(TLX_SCALES):
        raise ValueError(f"TLX needs exactly the scales {TLX_SCALES}")
    for name, v in scales.items():
        if not isinstance(v, int) or not 0 <= v <= 100 or v % 5:
            raise ValueError(f"TLX scale '{name}' must be 0-100 in steps of 5 (got {v!r})")
    return sum(scales.values()) / len(scales)

# The 4 participant types: (block that gets the heatmap, condition shown first).
PARTICIPANT_TYPES = [("B", WITHOUT), ("A", WITHOUT), ("B", WITH), ("A", WITH)]


def load_study_config(path: Path) -> tuple[dict, dict]:
    """Read the study config and the config it takes the model output from.

    Output: (study config with absolute paths, source config).
    """
    with open(path) as f:
        study = yaml.safe_load(f)
    study["paths"] = {k: (REPO_ROOT / v if not Path(v).is_absolute() else Path(v)) for k, v in study["paths"].items()}
    source = load_config(REPO_ROOT / study["source_config"])
    return study, source


def study_dir(study: dict) -> Path:
    """Root folder of this study, e.g. data/study/study_kits."""
    return study["paths"]["study_dir"] / study["name"]


def id_pattern(prefix: str) -> re.Pattern:
    """The ids of one kind: the prefix followed by digits only, e.g. P01 or T12."""
    return re.compile(re.escape(prefix) + r"\d+")


def id_prefixes(study: dict) -> tuple[str, str]:
    """(participant prefix, test prefix) from the study config. They must differ, and no id may fit both."""
    participant, test = study["study"]["participant_prefix"], study["study"]["test_prefix"]
    if not participant or not test or participant.startswith(test) or test.startswith(participant):
        raise ValueError(f"participant_prefix '{participant}' and test_prefix '{test}' must be different and "
                         "neither may start with the other")
    return participant, test


def participant_type(participant_id: str, prefix: str | tuple[str, ...]) -> int:
    """Balancing type (0-3) from an id like 'P07' (number 7 -> type (7 - 1) % 4 = 2).

    prefix: the allowed id prefix, or several (e.g. ("P", "T") for participant and test ids).
    """
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
    match = next((p for p in prefixes if id_pattern(p).fullmatch(participant_id)), None)
    if match is None:
        raise ValueError(f"Participant id must look like {prefixes[0]}01, {prefixes[0]}02, ... (got '{participant_id}')")
    number = int(participant_id[len(match):])
    if number < 1:
        raise ValueError("Participant numbers start at 1")
    return (number - 1) % len(PARTICIPANT_TYPES)


def assignment(participant_id: str, study_scans: list[str], prefix: str | tuple[str, ...]) -> list[dict]:
    """The ordered list of scans and conditions for one participant (see the module docstring).

    Input: participant id, the study scans in their fixed order (first half = block A), id prefix(es).
    Output: list of {"case_id", "condition", "position"} in the order the participant sees them.
    """
    half = len(study_scans) // 2
    blocks = {"A": study_scans[:half], "B": study_scans[half:]}
    heatmap_block, first = PARTICIPANT_TYPES[participant_type(participant_id, prefix)]
    other = "A" if heatmap_block == "B" else "B"
    with_part = [(c, WITH) for c in blocks[heatmap_block]]
    without_part = [(c, WITHOUT) for c in blocks[other]]
    ordered = with_part + without_part if first == WITH else without_part + with_part
    return [{"case_id": c, "condition": cond, "position": i + 1} for i, (c, cond) in enumerate(ordered)]


def candidate_table(study: dict, source: dict) -> pd.DataFrame:
    """Per clean scan of the source config: Dice, wrong volume, number of axial slices, sides found.

    Uses the ground truth (via dice.csv and the label files) ONLY to pick suitable study scans.
    """
    dice_df = pd.read_csv(results_dir(source) / "dice.csv")
    dice_df = dice_df[dice_df.variant == "clean"]
    scores = pd.read_csv(results_dir(source) / "scores.csv")
    scores = scores[scores.variant == "clean"][["case_id", "n_sides_found", study["queue"]["score"]]]
    df = dice_df.merge(scores, on="case_id")
    rows = []
    for case in list_cases(source):
        if case.case_id not in set(df.case_id):
            continue
        img = load_canonical(case.image_path)
        voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        r = df[df.case_id == case.case_id].iloc[0]
        overlap = r.dice * (r.pred_voxels + r.truth_voxels) / 2
        rows.append({"case_id": case.case_id, "split": r.split, "dice": r.dice,
                     "error_ml": (r.pred_voxels + r.truth_voxels - 2 * overlap) * voxel_ml,
                     "axial_slices": img.shape[2], "n_sides_found": r.n_sides_found,
                     "score": r[study["queue"]["score"]]})
    return pd.DataFrame(rows)


def select_scans(study: dict, table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pick the study scans and practice scans from the candidates (criteria in study.selection).

    Eligible scans are drawn at random with the study seed; the first n_scans become the study scans
    (their order defines blocks A and B), the next n_practice the practice scans.
    Output: (study scan ids, practice scan ids).
    """
    sel = study["selection"]
    ok = table[(table.dice >= sel["dice_min"]) & (table.dice <= sel["dice_max"])
               & (table.error_ml >= sel["error_ml_min"]) & (table.error_ml <= sel["error_ml_max"])
               & (table.axial_slices <= sel["max_axial_slices"])]
    if sel["require_both_sides"]:
        ok = ok[ok.n_sides_found == 2]
    ids = sorted(ok.case_id)
    order = np.random.default_rng(study["seed"]).permutation(len(ids))
    drawn = [ids[i] for i in order]
    needed = sel["n_scans"] + sel["n_practice"]
    if len(drawn) < needed:
        raise ValueError(f"Only {len(drawn)} scans meet the selection criteria, {needed} needed.")
    return drawn[:sel["n_scans"]], drawn[sel["n_scans"]:needed]


def prepare_scan(study: dict, source: dict, case, out_dir: Path, with_truth: bool = False) -> dict:
    """Write the files the viewer needs for one scan: ct.nii.gz, mask.nii.gz, heatmap.nii.gz.

    All three are on the same canonical (RAS) grid, cut to the slices with predicted kidney plus
    selection.crop_margin_mm, so the viewer loads quickly and the participant is not lost in the scan.
    The cut uses only the prediction, so it does not reveal where the ground truth is; it is checked
    afterwards that it keeps at least selection.min_gt_in_crop of the ground truth.

    with_truth=True also writes truth.nii.gz (the ground-truth organ mask). Only the practice scan (the
    answer is shown after the practice) and the demo queue get it; study scans never do, so the server
    cannot hand out the answer for a scan that is measured.

    Output: dict with scan facts (crop, Dice of the model before correction, GT share kept).
    """
    img = load_canonical(case.image_path)
    mask = load_map(source, "clean", case.case_id, "mask")
    heat = load_map(source, "clean", case.case_id, study["study"]["heatmap"])
    truth, ignore = load_organ_mask(source, case)
    spacing = voxel_spacing(source, "clean", case.case_id)

    z = np.flatnonzero(mask.any(axis=(0, 1)))
    margin = int(np.ceil(study["selection"]["crop_margin_mm"] / spacing[2]))
    z0, z1 = max(0, z.min() - margin), min(mask.shape[2], z.max() + 1 + margin)
    gt_kept = float(truth[:, :, z0:z1].sum() / max(truth.sum(), 1))

    # Shift the affine so the cropped volume stays in the same place in world coordinates.
    affine = img.affine.copy()
    affine[:3, 3] = img.affine[:3, :3] @ np.array([0, 0, z0]) + img.affine[:3, 3]
    ct = np.asanyarray(img.dataobj)[:, :, z0:z1]
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.clip(np.round(ct), -1024, 3071).astype(np.int16), affine), out_dir / "ct.nii.gz")
    nib.save(nib.Nifti1Image(mask[:, :, z0:z1].astype(np.uint8), affine), out_dir / "mask.nii.gz")
    # Heatmap values are 0-1; stored as 0-255 with a scale factor, like the other saved maps.
    heat_img = nib.Nifti1Image(np.round(np.clip(heat[:, :, z0:z1], 0, 1) * QUANT_LEVELS).astype(np.uint8), affine)
    heat_img.header.set_slope_inter(1.0 / QUANT_LEVELS, 0.0)
    nib.save(heat_img, out_dir / "heatmap.nii.gz")
    truth_file = out_dir / "truth.nii.gz"
    if with_truth:
        nib.save(nib.Nifti1Image(truth[:, :, z0:z1].astype(np.uint8), affine), truth_file)
    elif truth_file.exists():
        truth_file.unlink()
    return {"case_id": case.case_id, "crop_z": [int(z0), int(z1)], "gt_kept": round(gt_kept, 4),
            "dice_before": round(dice(mask, truth, ignore), 4)}


def prepare_study(study: dict, source: dict) -> dict:
    """Select the scans, prepare their files and write the manifest (manifest.json).

    Study and practice scans go to <study dir>/scans/, the demo queue (the highest-scoring test scans
    by the locked G1 measure) to <study dir>/queue/, the example image for the instruction screen to
    <study dir>/guide/. The run stops with an error if a crop would cut away ground truth.
    """
    table = candidate_table(study, source)
    root = study_dir(study)
    cases = {c.case_id: c for c in list_cases(source)}

    study_ids, practice_ids = select_scans(study, table)
    facts = {"scans": {}, "queue": {}}
    for cid in study_ids + practice_ids:
        facts["scans"][cid] = prepare_scan(study, source, cases[cid], root / "scans" / cid,
                                           with_truth=cid in practice_ids)
    lost = [c for c, f in facts["scans"].items() if f["gt_kept"] < study["selection"]["min_gt_in_crop"]]
    if lost:
        raise ValueError(f"The crop cuts away ground truth in {lost}; increase selection.crop_margin_mm.")

    queue = table[table.split != "dev"].sort_values("score", ascending=False).head(study["queue"]["n_scans"])
    queue_ids = list(queue.case_id)
    for cid in queue_ids:
        facts["queue"][cid] = prepare_scan(study, source, cases[cid], root / "queue" / cid, with_truth=True)

    # Example image for the instruction screen, from a development scan used nowhere else in the study.
    guide_case, guide_slice = pick_guide_slice(study, source, exclude=set(study_ids + practice_ids + queue_ids))
    make_guide_image(study, source, guide_case, guide_slice, root / "guide" / GUIDE_IMAGE)

    manifest = {"study_scans": study_ids, "practice_scans": practice_ids, "queue_scans": queue_ids,
                "queue_scores": {r.case_id: float(r.score) for r in queue.itertuples()},
                "heatmap": study["study"]["heatmap"], "guide": {"case_id": guide_case, "slice": guide_slice},
                "facts": facts}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    table.to_csv(root / "candidates.csv", index=False)
    return manifest


def load_manifest(study: dict) -> dict:
    """Read the manifest written by prepare_study."""
    path = study_dir(study) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/prepare_study.py first.")
    return json.loads(path.read_text())


def clean_test_data(study: dict, dry_run: bool = False) -> list[Path]:
    """Delete what test runs saved: sessions/<test id>/ and queue_edits/ (the demo queue, never study data).

    Safety: a session folder is only deleted if its name is a test id (test prefix + digits, e.g. T03) AND not
    a participant id. Everything else in sessions/ (P01, ...) is left alone. Symbolic links are removed as
    links; what they point to is never touched.

    Input: study config; dry_run=True only lists what would be deleted.
    Output: the deleted (or, with dry_run, the would-be deleted) paths.
    """
    participant, test = id_prefixes(study)
    root = study_dir(study)
    targets = []
    sessions = root / "sessions"
    if sessions.is_dir():
        for folder in sorted(sessions.iterdir()):
            if id_pattern(test).fullmatch(folder.name) and not id_pattern(participant).fullmatch(folder.name):
                targets.append(folder)
    if (root / "queue_edits").exists():
        targets.append(root / "queue_edits")
    if not dry_run:
        for t in targets:
            if t.is_symlink() or t.is_file():
                t.unlink()
            else:
                shutil.rmtree(t)
    return targets
