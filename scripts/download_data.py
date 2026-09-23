"""Download the CT dataset with ground truth into paths.raw_dir (pipeline step 1).

Usage: uv run python scripts/download_data.py --config configs/local.yaml
"""

from segreview.config import config_arg_parser, load_config
from segreview.data import download_dataset, list_cases


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    folder = download_dataset(cfg)
    print(f"Dataset in {folder}; {len(list_cases(cfg))} scans selected by the config.")


if __name__ == "__main__":
    main()
