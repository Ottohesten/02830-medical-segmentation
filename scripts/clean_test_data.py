"""Delete the data of test runs of the review interface: sessions of test ids (T01, T02, ...) and the demo-queue edits.

Real participant data (sessions/P01, ...) is never deleted: a session folder is only removed if its name is a test
id (study.test_prefix followed by digits) and not a participant id (study.participant_prefix followed by digits).

Usage: uv run python scripts/clean_test_data.py --config configs/study.yaml [--dry-run]
"""

import argparse
from pathlib import Path

from segreview.study import clean_test_data, load_study_config, study_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be deleted")
    args = parser.parse_args()
    study, _ = load_study_config(args.config)
    targets = clean_test_data(study, dry_run=args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    for t in targets:
        print(f"{verb}: {t.relative_to(study_dir(study))}")
    if not targets:
        print("No test data found.")
    kept = sorted(p.name for p in (study_dir(study) / "sessions").glob("*") if p not in targets)
    if kept:
        print(f"Kept (not test ids): {', '.join(kept)}")


if __name__ == "__main__":
    main()
