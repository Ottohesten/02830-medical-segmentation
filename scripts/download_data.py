"""Download the CT dataset with ground truth into paths.raw_dir (pipeline step 1).

Usage: uv run python scripts/download_data.py --config configs/local.yaml
       uv run python scripts/download_data.py --config configs/kits100.yaml --cases case_00183 case_00002

--cases downloads only the listed scans (and the dataset's full case list), e.g. the 2 scans of the G3 smoke test,
which are not among the config's own selection. Only for datasets with download format 'files' (KiTS23).
"""

from segreview.config import config_arg_parser, load_config
from segreview.data import download_cases, download_dataset, list_cases


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--cases", nargs="+", metavar="CASE_ID",
                        help="download only these scans instead of the config's selection")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cases:
        folder = download_cases(cfg, args.cases)
        print(f"{len(args.cases)} scans in {folder}.")
        return
    folder = download_dataset(cfg)
    print(f"Dataset in {folder}; {len(list_cases(cfg))} scans selected by the config.")


if __name__ == "__main__":
    main()
