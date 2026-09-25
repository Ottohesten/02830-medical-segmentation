"""Dataset access: download CT scans with ground truth and load one scan at a time (pipeline step 1).

Datasets are interchangeable. Each one is described by a small YAML file in configs/datasets/
(where the images and labels are, which label values form the organ, and which TotalSegmentator
classes match it). This module only reads that description, so adding a dataset needs a new
YAML file, not new code, as long as the scans are NIfTI files with one label file per scan.

Two download formats are supported:
- tar:   one archive with everything (MSD Spleen).
- files: one image and one label file per scan, fetched separately (KiTS23). Only the selected
         scans are downloaded, so a subset fits on a small disk.

Which scans are used (dataset.n_cases, dataset.selection):
- first:  the first n in sorted order.
- random: n scans drawn at random with the config's seed, from the FULL list of the dataset's scans.
  The draw is a random permutation cut after n, so a larger n with the same seed contains the
  smaller selection (the 20-scan pilot is part of a later 100-scan run).
"""

import tarfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import requests
from tqdm import tqdm

CASE_LIST_FILE = "case_list.txt"   # full list of the dataset's scan ids, written at download time


@dataclass
class Case:
    """One CT scan and its ground-truth label file on disk."""
    case_id: str
    image_path: Path
    label_path: Path


def dataset_dir(cfg: dict) -> Path:
    """Return the folder the dataset lives in (paths.raw_dir / dataset.root)."""
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["root"]


def _stem(path: Path) -> str:
    """File name without .nii.gz / .nii (e.g. spleen_10.nii.gz -> spleen_10)."""
    return path.name.removesuffix(".gz").removesuffix(".nii")


def select_case_ids(cfg: dict, all_ids: list[str]) -> list[str]:
    """Pick the scans to use from the full list of scan ids (see module docstring).

    Input: config (dataset.n_cases, dataset.selection, seed), all scan ids of the dataset.
    Output: the selected ids, sorted.
    """
    ds = cfg["dataset"]
    ids = sorted(all_ids)
    n = ds["n_cases"] if ds["n_cases"] is not None else len(ids)
    selection = ds.get("selection", "first")
    if selection == "first":
        return ids[:n]
    if selection == "random":
        order = np.random.default_rng(cfg["seed"]).permutation(len(ids))
        return sorted(ids[i] for i in order[:n])
    raise ValueError(f"Unknown selection '{selection}'")


def _download_file(url: str, target: Path) -> None:
    """Download one file in chunks (never the whole file in memory). Writes to a temporary name first,
    so an interrupted download never leaves a broken file that looks finished."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(partial, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name, leave=False) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    partial.rename(target)


def _download_tar(cfg: dict, target: Path) -> None:
    """Download and unpack a .tar archive; the archive is deleted afterwards to save disk space."""
    raw_dir = cfg["paths"]["raw_dir"]
    url = cfg["dataset"]["download"]["url"]
    tar_path = raw_dir / Path(url).name
    _download_file(url, tar_path)
    print(f"Unpacking {tar_path.name} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir, filter="data")
    tar_path.unlink()


def _download_files(cfg: dict, target: Path) -> None:
    """Download only the selected scans, one image + one label file each.

    The dataset's full list of scan ids comes from download.list_url (a JSON file listing, as the
    Hugging Face API returns it) and is saved to case_list.txt, so the random selection is always
    drawn from the same full list.
    """
    dl = cfg["dataset"]["download"]
    listing = requests.get(dl["list_url"], timeout=60)
    listing.raise_for_status()
    all_ids = sorted(_stem(Path(f["path"])) for f in listing.json() if f["type"] == "file")
    target.mkdir(parents=True, exist_ok=True)
    (target / CASE_LIST_FILE).write_text("\n".join(all_ids) + "\n")

    for case_id in tqdm(select_case_ids(cfg, all_ids), desc="scans"):
        for template in (dl["image_file"], dl["label_file"]):
            rel = template.format(case_id=case_id)
            if not (target / rel).exists():
                _download_file(f"{dl['base_url']}/{rel}", target / rel)


def download_dataset(cfg: dict) -> Path:
    """Download the dataset (or the selected part of it) if it is not already there.

    Input: the loaded config (uses dataset.download and paths.raw_dir).
    Output: the folder with the dataset.
    """
    target = dataset_dir(cfg)
    fmt = cfg["dataset"]["download"]["format"]
    if fmt == "tar":
        if target.is_dir() and any(target.iterdir()):
            print(f"Dataset already present: {target}")
        else:
            _download_tar(cfg, target)
    elif fmt == "files":
        _download_files(cfg, target)   # skips files that are already there
    else:
        raise NotImplementedError(f"No download for format '{fmt}'. Download it manually into {target}.")
    return target


def list_cases(cfg: dict, splits: list[str] | None = None) -> list[Case]:
    """List the scans to use (selection from the config), with their image and label paths.

    Image files are found with dataset.image_glob. The matching label file is dataset.label_path,
    where {stem} is replaced by the image file name without extension and {parent} by the folder
    the image is in. Hidden files (starting with '.', e.g. macOS '._' metadata files) are skipped.
    If the dataset has a case_list.txt (partial downloads), the selection is drawn from that full list.

    Input: the loaded config, and optionally a list of splits to keep (e.g. ["dev"]); only those scans
           then have to be downloaded.
    Output: a list of Case objects, sorted by id.
    """
    ds = cfg["dataset"]
    root = dataset_dir(cfg)
    images = sorted(p for p in root.glob(ds["image_glob"]) if not p.name.startswith("."))
    local = {}
    for image in images:
        names = {"stem": _stem(image), "parent": image.parent.name}
        local[names[ds["case_id"]]] = Case(names[ds["case_id"]], image, root / ds["label_path"].format(**names))

    case_list = root / CASE_LIST_FILE
    all_ids = case_list.read_text().split() if case_list.exists() else list(local)
    if not all_ids:
        raise FileNotFoundError(f"No scans matching {root / ds['image_glob']}. Run scripts/download_data.py first.")
    selected = select_case_ids(cfg, all_ids)
    if splits is not None:
        selected = [i for i in selected if case_split(cfg, i) in splits]
    missing = [i for i in selected if i not in local]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected scans are not downloaded (e.g. {missing[0]}). "
                                f"Run scripts/download_data.py with this config.")
    return [local[i] for i in selected]


def case_split(cfg: dict, case_id: str) -> str:
    """Which split a scan belongs to: 'dev' if listed in dataset.split.dev, else 'test'.

    Datasets without a split return 'all'. The development set is where every choice is made; the
    test set is only evaluated once, after the choices are locked (see evaluation.splits).
    """
    split = cfg["dataset"].get("split")
    if not split:
        return "all"
    return "dev" if case_id in split["dev"] else "test"


def evaluation_cases(cfg: dict) -> list[Case]:
    """The scans whose ground truth may be used now: those in the splits listed in evaluation.splits.

    Keeping the test set out of evaluation.splits until all choices are locked makes it impossible to
    look at test-set Dice by accident. Datasets without a split: all selected scans.
    """
    allowed = cfg["evaluation"].get("splits")
    if not cfg["dataset"].get("split") or allowed is None:
        return list_cases(cfg)
    return list_cases(cfg, splits=allowed)


def load_canonical(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file and reorient it to the closest canonical (RAS) orientation.

    TotalSegmentator does the same before inference, so the model sees scans in the
    orientation it was trained on. We reorient the ground truth the same way so that
    prediction and ground truth line up voxel by voxel. Reorienting only flips or swaps
    axes; it never interpolates, so the ground truth keeps its original resolution.
    """
    return nib.as_closest_canonical(nib.load(path))


def organ_definition(cfg: dict, name: str | None = None) -> dict:
    """The ground-truth definition of the organ: which label values count as organ, and which are ignored.

    A dataset file lists one or more definitions under organ_definitions; organ_definition picks
    the one in use. Example for KiTS: "kidney + tumor + cyst" or "kidney + cyst".
    - organ_labels:  label values that together form the organ (a union).
    - ignore_labels: label values left out of the Dice computation entirely (optional). Useful when
                     it is unclear whether a structure should count as organ or not.

    Input: config, and optionally a definition name (default: the one the config selects).
    Output: dict with organ_labels and ignore_labels.
    """
    ds = cfg["dataset"]
    d = ds["organ_definitions"][name or ds["organ_definition"]]
    return {"organ_labels": d["organ_labels"], "ignore_labels": d.get("ignore_labels", [])}


def load_label_map(case: Case) -> np.ndarray:
    """The raw ground-truth label values of one scan (canonical orientation, original resolution)."""
    return np.asanyarray(load_canonical(case.label_path).dataobj)


def organ_from_labels(labels: np.ndarray, definition: dict) -> tuple[np.ndarray, np.ndarray]:
    """Turn a label map into (organ mask, ignore mask) for one organ definition. Evaluation only."""
    return np.isin(labels, definition["organ_labels"]), np.isin(labels, definition["ignore_labels"])


def load_organ_mask(cfg: dict, case: Case, definition: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth organ mask and ignore mask of one scan, for the config's organ definition.

    Used ONLY for evaluation.

    Output: (organ, ignore), boolean arrays. ignore is all False if the definition ignores nothing.
    """
    return organ_from_labels(load_label_map(case), organ_definition(cfg, definition))
