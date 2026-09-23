"""Run the whole G1 evaluation with one command.

inference (clean + perturbations) -> TTA -> second model -> Dice -> label check (if several organ
definitions) -> uncertainty scores + heatmaps -> evaluation + figures.
Steps that are already done (saved predictions, TTA maps) are skipped, so this can be rerun safely.

Usage: uv run python scripts/run_g1.py --config configs/local_all.yaml
"""

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.pipeline import (step_dice, step_evaluate, step_inference, step_label_check, step_second_model,
                                step_tta, step_uncertainty)


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    device = setup_compute(cfg)
    step_inference(cfg, device)
    step_tta(cfg, device)
    step_second_model(cfg, device)
    step_dice(cfg)
    step_label_check(cfg)
    step_uncertainty(cfg)
    step_evaluate(cfg)


if __name__ == "__main__":
    main()
