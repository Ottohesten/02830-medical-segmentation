"""Compare ground-truth definitions of the organ (evaluation only; needs saved predictions).

For every clean scan: Dice under each definition in the dataset file (e.g. KiTS: kidney + tumour + cyst,
kidney + cyst), and the share of each ground-truth label that the model calls organ (e.g. how much of
the tumour TotalSegmentator calls kidney or kidney cyst). Writes results/<config name>/label_check.csv,
label_check_summary.csv and figures/label_check.png.

Usage: uv run python scripts/check_labels.py --config configs/kits_pilot.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.pipeline import step_label_check


def main():
    args = config_arg_parser(__doc__).parse_args()
    if step_label_check(load_config(args.config)) is None:
        print("The dataset has only one organ definition; nothing to compare.")


if __name__ == "__main__":
    main()
