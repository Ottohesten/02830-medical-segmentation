"""Dataset access: download CT scans with ground truth and load one scan at a time (pipeline step 1).

Datasets are interchangeable. Each one is described by a small YAML file in configs/datasets/
(where the images and labels are, which label values form the organ, and which TotalSegmentator
classes match it). This module only reads that description, so adding a dataset needs a new
YAML file, not new code, as long as the scans are NIfTI files with one label file per scan.

The first dataset is MSD Task09 Spleen (see configs/datasets/msd_spleen.yaml).
"""

import tarfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import requests
from tqdm import tqdm


@dataclass
class Case:
    """One CT scan and its ground-truth label file on disk."""
    case_id: str
    image_path: Path
    label_path: Path


def dataset_dir(cfg: dict) -> Path:
    """Return the folder the dataset lives in (paths.raw_dir / dataset.root)."""
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["root"]


def download_dataset(cfg: dict) -> Path:
    """Download and unpack the dataset if it is not already there.

    Only .tar archives are handled here. The archive is deleted after unpacking to save disk space.

    Input: the loaded config (uses dataset.download and paths.raw_dir).
    Output: the folder with the unpacked dataset.
    """
    target = dataset_dir(cfg)
    if target.is_dir() and any(target.iterdir()):
        print(f"Dataset already present: {target}")
        return target

    download = cfg["dataset"]["download"]
    if download["format"] != "tar":
        raise NotImplementedError(f"No automatic download for format '{download['format']}'. "
                                  f"Download it manually into {target}.")
    raw_dir = cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / Path(download["url"]).name

    # Stream the download to disk in chunks so the whole file never sits in memory.
    with requests.get(download["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tar_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Unpacking {tar_path.name} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir, filter="data")
    tar_path.unlink()
    return target


def _stem(path: Path) -> str:
    """File name without .nii.gz / .nii (e.g. spleen_10.nii.gz -> spleen_10)."""
    return path.name.removesuffix(".gz").removesuffix(".nii")


def list_cases(cfg: dict) -> list[Case]:
    """List the scans to use, in sorted order, limited to dataset.n_cases.

    Image files are found with dataset.image_glob. The matching label file is dataset.label_path,
    where {stem} is replaced by the image file name without extension and {parent} by the folder
    the image is in. Hidden files (starting with '.', e.g. macOS '._' metadata files) are skipped.

    Input: the loaded config.
    Output: a list of Case objects.
    """
    ds = cfg["dataset"]
    root = dataset_dir(cfg)
    images = sorted(p for p in root.glob(ds["image_glob"]) if not p.name.startswith("."))
    if not images:
        raise FileNotFoundError(f"No scans matching {root / ds['image_glob']}. Run scripts/download_data.py first.")

    cases = []
    for image in images:
        names = {"stem": _stem(image), "parent": image.parent.name}
        label = root / ds["label_path"].format(**names)
        cases.append(Case(names[ds["case_id"]], image, label))
    n = ds["n_cases"]
    return cases if n is None else cases[:n]


def load_canonical(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file and reorient it to the closest canonical (RAS) orientation.

    TotalSegmentator does the same before inference, so the model sees scans in the
    orientation it was trained on. We reorient the ground truth the same way so that
    prediction and ground truth line up voxel by voxel. Reorienting only flips or swaps
    axes; it never interpolates, so the ground truth keeps its original resolution.
    """
    return nib.as_closest_canonical(nib.load(path))


def load_organ_mask(cfg: dict, case: Case) -> np.ndarray:
    """Load the ground-truth organ mask of one scan (canonical orientation, original resolution).

    The organ can be a union of several label values (dataset.organ_labels), e.g. kidney,
    tumour and cyst together form "the kidney" in KiTS. Used ONLY for evaluation.

    Output: boolean array, True inside the organ.
    """
    labels = np.asanyarray(load_canonical(case.label_path).dataobj)
    return np.isin(labels, cfg["dataset"]["organ_labels"])
