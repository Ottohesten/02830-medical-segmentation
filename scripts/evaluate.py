"""Compare saved predictions with the ground truth: Dice per scan and variant (evaluation only).

Writes results/<config name>/dice.csv. Ground truth is only read here and in evaluate_g1.py,
never in the prediction or uncertainty code.

Usage: uv run python scripts/evaluate.py --config configs/local.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.pipeline import step_dice


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    df = step_dice(cfg)
    print(df.groupby("variant")["dice"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
