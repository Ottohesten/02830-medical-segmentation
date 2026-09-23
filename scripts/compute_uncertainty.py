"""Compute the GT-free uncertainty scores and the voxel-wise entropy heatmaps (G1, pipeline step 3).

Reads only the saved model output (mask, probability, TTA maps). Writes results/<config name>/scores.csv
and data/predictions/<config name>/<variant>/<case>_entropy.nii.gz (the heatmap for the review interface).

Usage: uv run python scripts/compute_uncertainty.py --config configs/local.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.pipeline import step_uncertainty


def main():
    args = config_arg_parser(__doc__).parse_args()
    step_uncertainty(load_config(args.config))


if __name__ == "__main__":
    main()
