"""Run the pretrained model on every scan chosen by the config (pipeline step 2).

Predicts the clean scans and, if the config lists perturbations, a degraded copy of each scan per
perturbation. Saves mask + organ probability to data/predictions/<config name>/<variant>/ and the
runtime to results/<config name>/inference_times.csv. Already saved predictions are skipped,
so an interrupted run can simply be restarted.

Usage: uv run python scripts/run_inference.py --config configs/local.yaml [--overwrite]
"""

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.pipeline import step_inference


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="predict again even if output exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    step_inference(cfg, setup_compute(cfg), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
