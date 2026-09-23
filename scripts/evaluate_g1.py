"""Evaluate how well each uncertainty score ranks the scans (G1). Uses ground truth via dice.csv.

Needs results/<config name>/scores.csv and dice.csv. Writes g1_scores_<set>.csv, g1_metrics_<set>.csv
and figures/g1_curves_<set>.png + g1_scatter_<set>.png, for set = clean and (if any) perturbed.

Usage: uv run python scripts/evaluate_g1.py --config configs/local_all.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.pipeline import step_evaluate


def main():
    args = config_arg_parser(__doc__).parse_args()
    step_evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
