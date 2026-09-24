"""Prepare the user study (G2): pick the study, practice and demo-queue scans and write their viewer files.

Writes data/study/<study name>/scans/<case>/{ct,mask,heatmap}.nii.gz, .../queue/<case>/..., manifest.json
and candidates.csv (every scan with the values the selection criteria use). The practice scan and the demo
queue also get truth.nii.gz (the ground truth, shown as the answer); study scans never do. guide/example.png
is the example image for the instruction screen.

Usage: uv run python scripts/prepare_study.py --config configs/study.yaml
"""

import argparse
from pathlib import Path

from segreview.study import load_study_config, prepare_study, study_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    args = parser.parse_args()
    study, source = load_study_config(args.config)
    manifest = prepare_study(study, source)
    print(f"Study scans (block A = first half, B = second half): {manifest['study_scans']}")
    print(f"Practice: {manifest['practice_scans']}   Queue (demo): {manifest['queue_scans']}")
    for cid, f in manifest["facts"]["scans"].items():
        print(f"  {cid}: Dice before {f['dice_before']}, slices {f['crop_z']}, ground truth kept {f['gt_kept']}")
    print(f"Guide image from {manifest['guide']['case_id']}, slice {manifest['guide']['slice']}")
    print(f"-> {study_dir(study)}")


if __name__ == "__main__":
    main()
