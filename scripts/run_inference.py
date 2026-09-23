"""Run the pretrained model on the scans chosen by the config (pipeline step 2).

For each scan: predict the organ, save mask + organ probability to paths.predictions_dir/<config name>/,
and log the runtime to results/inference_<config name>_<date>.csv.
Scans that already have a saved prediction are skipped, so an interrupted run can simply be restarted.

Usage: uv run python scripts/run_inference.py --config configs/local.yaml [--overwrite]
"""

import time
from datetime import date

import pandas as pd

from segreview.config import config_arg_parser, load_config, setup_compute
from segreview.data import list_cases, load_canonical
from segreview.inference import OrganSegmenter
from segreview.io import mask_path, prob_path, save_prediction


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="predict again even if output exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = setup_compute(cfg)

    cases = list_cases(cfg)
    todo = [c for c in cases if args.overwrite
            or not (mask_path(cfg, c.case_id).exists() and prob_path(cfg, c.case_id).exists())]
    print(f"Config '{cfg['name']}': device={device}, resolution={cfg['model']['resolution']}, "
          f"{len(cases)} scans ({len(todo)} to predict)")
    if not todo:
        return

    segmenter = OrganSegmenter(cfg, device)
    rows = []
    for case in todo:
        image = load_canonical(case.image_path)
        start = time.perf_counter()
        pred = segmenter.predict(image)
        seconds = time.perf_counter() - start
        save_prediction(cfg, case.case_id, pred)
        print(f"  {case.case_id}: shape {image.shape}, {seconds:.1f} s, {int(pred.mask.sum())} organ voxels")
        rows.append({"case_id": case.case_id, "seconds": round(seconds, 2), "device": str(device),
                     "resolution": cfg["model"]["resolution"], "shape": "x".join(map(str, image.shape))})

    out = cfg["paths"]["results_dir"] / f"inference_{cfg['name']}_{date.today().isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():  # a restarted run on the same day adds to the log instead of replacing it
        df = pd.concat([pd.read_csv(out), df]).drop_duplicates("case_id", keep="last")
    df.to_csv(out, index=False)
    print(f"Runtime log: {out}")


if __name__ == "__main__":
    main()
