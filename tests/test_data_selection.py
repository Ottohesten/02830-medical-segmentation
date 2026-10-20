"""Tests of which scans a config selects and of downloading single scans (src/segreview/data.py), on a tiny fake
dataset in a temporary folder (no network: the download functions are replaced)."""

import pytest

from segreview import data
from segreview.data import (CASE_LIST_FILE, dataset_dir, download_cases, list_cases, select_case_ids,
                            selected_case_ids)

IDS = [f"case_{i:05d}" for i in range(10)]


def make_cfg(tmp_path, n_cases=3, selection="random") -> dict:
    """A config like configs/kits100.yaml with configs/datasets/kits23.yaml, for a dataset in tmp_path."""
    return {"seed": 42, "paths": {"raw_dir": tmp_path},
            "dataset": {"root": "kits", "image_glob": "images/*.nii.gz", "label_path": "segmentations/{stem}.nii.gz",
                        "case_id": "stem", "n_cases": n_cases, "selection": selection,
                        "download": {"format": "files", "list_url": "https://example.org/list",
                                     "base_url": "https://example.org/files",
                                     "image_file": "images/{case_id}.nii.gz",
                                     "label_file": "segmentations/{case_id}.nii.gz"}}}


def put_scans(cfg, case_ids) -> None:
    """Empty image and label files, as if the scans were downloaded (list_cases only looks at the names)."""
    for sub in ("images", "segmentations"):
        (dataset_dir(cfg) / sub).mkdir(parents=True, exist_ok=True)
        for case_id in case_ids:
            (dataset_dir(cfg) / sub / f"{case_id}.nii.gz").touch()


def write_case_list(cfg) -> None:
    dataset_dir(cfg).mkdir(parents=True, exist_ok=True)
    (dataset_dir(cfg) / CASE_LIST_FILE).write_text("\n".join(IDS) + "\n")


def test_selection_from_the_case_list_needs_no_downloaded_scans(tmp_path):
    cfg = make_cfg(tmp_path)
    write_case_list(cfg)
    chosen = selected_case_ids(cfg)
    assert chosen == select_case_ids(cfg, IDS)       # the same seeded draw from the full list
    assert len(chosen) == 3
    with pytest.raises(FileNotFoundError):           # list_cases still needs every selected scan on disk
        list_cases(cfg)


def test_list_cases_gives_the_same_selection_with_its_files(tmp_path):
    cfg = make_cfg(tmp_path)
    write_case_list(cfg)
    put_scans(cfg, selected_case_ids(cfg) + ["case_99999"])     # an extra scan on disk changes nothing
    cases = list_cases(cfg)
    assert [c.case_id for c in cases] == selected_case_ids(cfg)
    assert all(c.label_path == dataset_dir(cfg) / "segmentations" / f"{c.case_id}.nii.gz" for c in cases)


def test_without_a_case_list_the_selection_comes_from_the_scans_on_disk(tmp_path):
    cfg = make_cfg(tmp_path, n_cases=None, selection="first")
    put_scans(cfg, IDS[:4])
    assert selected_case_ids(cfg) == IDS[:4]
    with pytest.raises(FileNotFoundError):
        selected_case_ids(make_cfg(tmp_path / "empty"))


def test_download_cases_fetches_only_the_given_scans_and_the_full_list(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    listing = [{"type": "file", "path": f"images/{c}.nii.gz"} for c in IDS] + [{"type": "directory", "path": "x"}]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return listing

    fetched = []

    def fake_download(url, target):
        fetched.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    monkeypatch.setattr(data.requests, "get", lambda url, timeout: Response())
    monkeypatch.setattr(data, "_download_file", fake_download)

    download_cases(cfg, ["case_00007", "case_00002"])
    assert sorted(fetched) == sorted(f"https://example.org/files/{sub}/{c}.nii.gz"
                                     for c in ("case_00007", "case_00002") for sub in ("images", "segmentations"))
    assert (dataset_dir(cfg) / CASE_LIST_FILE).read_text().split() == IDS
    fetched.clear()
    download_cases(cfg, ["case_00007"])              # already there: nothing is fetched again
    assert fetched == []
    with pytest.raises(ValueError):                  # not in the dataset
        download_cases(cfg, ["case_99999"])
    cfg["dataset"]["download"]["format"] = "tar"
    with pytest.raises(ValueError):                  # single scans only from 'files' datasets
        download_cases(cfg, ["case_00001"])
