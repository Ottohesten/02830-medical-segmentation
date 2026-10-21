"""Predict clean and configured perturbed scans with the second model (e.g. the 6 mm network).

The disagreement between the main and the second model is used as an uncertainty score and heatmap.

Usage: uv run python scripts/run_second_model.py --config configs/local_all.yaml [--overwrite]
"""

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.pipeline import step_second_model


def main():
    """Read the config and run missing second-model predictions for every configured variant."""
    parser = config_arg_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="run again even if output exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    step_second_model(cfg, setup_compute(cfg), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
