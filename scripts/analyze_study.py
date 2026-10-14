"""Analyze the user study (G2): Dice gain per minute with vs without the uncertainty heatmap (Wilcoxon, paired
per participant), plus NASA-TLX. See src/segreview/study_analysis.py for the method.

Carries out docs/study/analysis_plan.md: H1 (gain per minute, primary), H2 (raw NASA-TLX, tested only if H1 is
supported), everything else exploratory. Run it ONCE, after the last participant; not during the study.

Reads data/study/<study name>/sessions/ (real participant ids only; test ids like T01 are ignored; incomplete
sessions are listed as excluded) and writes results/<study name>/study_*.csv and a figure. Writes nothing if no
participant has been run yet.

Usage: uv run python scripts/analyze_study.py --config configs/study.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from segreview.config import REPO_ROOT
from segreview.study import load_study_config, study_dir
from segreview.study_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    args = parser.parse_args()
    study, source = load_study_config(args.config)
    out_dir = REPO_ROOT / "results" / study["name"]
    result = analyze(study, source, study_dir(study) / "sessions", out_dir)
    if result is None:
        print("No participant sessions yet (only ids like P01 count; test ids like T01 are ignored). Nothing written.")
        return
    scans, exclusions = result["scans"], result["exclusions"]
    included = scans[scans.included]
    print(f"{included.participant.nunique()} complete participants, {len(included)} corrected scans")
    for r in exclusions.itertuples():
        print(f"  excluded {r.participant}: {r.reason}")
    if result["tests"].empty:
        print("No complete session yet: no tests.")
    else:
        columns = ["outcome", "hypothesis", "n", "mean_with", "mean_without", "mean_diff", "ci_low", "ci_high",
                   "r_rank_biserial", "p", "conclusion"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(result["tests"][columns].round(4).to_string(index=False))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
