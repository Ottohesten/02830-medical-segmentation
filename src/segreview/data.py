"""Dataset access: download the CT scans with ground truth and load one scan at a time.

We use the Medical Segmentation Decathlon (MSD) Task09 Spleen dataset: 41 abdominal CT
scans from Memorial Sloan Kettering with a manual spleen mask for each. It was chosen
because the pretrained model (TotalSegmentator) was trained on data from a different
hospital (University Hospital Basel), so the model has not seen these scans.

This is step 1 of the pipeline: CT scan + ground truth in, (image, mask) arrays out.
"""

import tarfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import requests
from tqdm import tqdm


@dataclass
class Case:
    """One CT scan and its ground-truth mask on disk."""
    case_id: str
    image_path: Path
    label_path: Path


def dataset_dir(cfg: dict) -> Path:
    """Return the folder the dataset is extracted to (e.g. data/raw/Task09_Spleen)."""
    name = Path(cfg["dataset"]["url"]).stem  # "Task09_Spleen.tar" -> "Task09_Spleen"
    return cfg["paths"]["raw_dir"] / name


def download_dataset(cfg: dict) -> Path:
    """Download and unpack the dataset if it is not already there.

    The .tar file is deleted after unpacking to save disk space.

    Input: the loaded config (uses dataset.url and paths.raw_dir).
    Output: the folder with the unpacked dataset.
    """
    target = dataset_dir(cfg)
    if (target / "imagesTr").is_dir():
        print(f"Dataset already present: {target}")
        return target

    raw_dir = cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = cfg["dataset"]["url"]
    tar_path = raw_dir / Path(url).name

    # Stream the download to disk in chunks so the whole file never sits in memory.
    with requests.get(url, stream=True, timeout=60) as r:
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


def list_cases(cfg: dict) -> list[Case]:
    """List the scans to use, in sorted order, limited to dataset.n_cases.

    Only the training split (imagesTr/labelsTr) has ground truth, so that is the split we use.
    Files starting with '._' are macOS metadata files that ship inside the archive and are skipped.

    Input: the loaded config.
    Output: a list of Case objects.
    """
    root = dataset_dir(cfg)
    images = sorted(p for p in (root / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
    if not images:
        raise FileNotFoundError(f"No scans in {root / 'imagesTr'}. Run scripts/download_data.py first.")
    cases = [Case(p.name.removesuffix(".nii.gz"), p, root / "labelsTr" / p.name) for p in images]
    n = cfg["dataset"]["n_cases"]
    return cases if n is None else cases[:n]


def load_canonical(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file and reorient it to the closest canonical (RAS) orientation.

    TotalSegmentator does the same before inference, so the model sees scans in the
    orientation it was trained on. We reorient the ground truth the same way so that
    prediction and ground truth line up voxel by voxel.
    """
    return nib.as_closest_canonical(nib.load(path))
