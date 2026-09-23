"""Download the pretrained TotalSegmentator weights into paths.models_dir (pipeline step 2).

Downloads the main model (model.resolution) and, if configured, the second model
(uncertainty.second_model_resolution). Also reports which nnU-Net folds are included, because
fold disagreement as an uncertainty measure needs more than one fold.

Usage: uv run python scripts/download_model.py --config configs/local.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.weights import available_folds, download_weights, model_spec


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    resolutions = [cfg["model"]["resolution"], cfg["uncertainty"].get("second_model_resolution")]
    for resolution in filter(None, resolutions):
        spec = model_spec(cfg, resolution)
        folder = download_weights(cfg, resolution)
        print(f"Task {spec.task_id} ({resolution}, {spec.spacing_mm} mm), "
              f"organ {cfg['dataset']['totalseg_classes']} = channels {spec.organ_channels}")
        print(f"  weights: {folder}")
        print(f"  folds with a checkpoint: {available_folds(folder)}")


if __name__ == "__main__":
    main()
