"""Predict the clean scans with the second model (uncertainty.second_model_resolution, e.g. 6 mm).

The disagreement between the main and the second model is used as an uncertainty score and heatmap.

Usage: uv run python scripts/run_second_model.py --config configs/local_all.yaml [--overwrite]
"""

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.pipeline import step_second_model


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="run again even if output exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    step_second_model(cfg, setup_compute(cfg), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
