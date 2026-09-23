"""Test-time augmentation on the clean scans (G1 uncertainty; runs only if uncertainty.tta.n_passes > 0).

Each pass runs the network on a slightly shifted, rescaled and noisy copy of the scan (no mirroring,
because the model was trained without it). Saves per-voxel vote counts and standard deviation maps.

Usage: uv run python scripts/run_tta.py --config configs/local.yaml [--overwrite]
"""

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.pipeline import step_tta


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="run again even if output exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    step_tta(cfg, setup_compute(cfg), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
