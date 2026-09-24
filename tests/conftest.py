"""Shared test setup: a temporary copy of the study folder, so tests never write into real participant data.

The prepared scans are not copied (they are large); the temporary folder links to them instead.
Tests that need the prepared study are skipped if scripts/prepare_study.py has not been run.
"""

import copy
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from segreview.config import REPO_ROOT
from segreview.study import load_study_config, study_dir

STUDY_CONFIG = REPO_ROOT / "configs" / "study.yaml"


@pytest.fixture(scope="session")
def real_study():
    study, source = load_study_config(STUDY_CONFIG)
    if not (study_dir(study) / "manifest.json").exists():
        pytest.skip("study not prepared (run scripts/prepare_study.py)")
    return study, source


@pytest.fixture()
def temp_study(real_study, tmp_path):
    """Study config whose folder is a temporary directory linking to the real prepared scans."""
    study, _ = real_study
    real = study_dir(study)
    fake = copy.deepcopy(study)
    fake["paths"]["study_dir"] = tmp_path / "study"
    root = study_dir(fake)
    root.mkdir(parents=True)
    for sub in ("scans", "queue", "guide"):
        if (real / sub).exists():
            (root / sub).symlink_to(real / sub)
    (root / "manifest.json").write_text((real / "manifest.json").read_text())
    return fake


@contextmanager
def running_server(study):
    """The review server for a study config, on a free port in a background thread. Yields the base url."""
    from segreview.ui_server import make_server

    srv = make_server(study, host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def server(temp_study):
    """The review server with the study config as it is. Yields (base url, study config)."""
    with running_server(temp_study) as base:
        yield base, temp_study


@pytest.fixture()
def quick_server(temp_study):
    """Like server, but with a 4-second time limit, so a test can let the time run out."""
    temp_study["study"]["time_limit_min"] = 4 / 60
    with running_server(temp_study) as base:
        yield base, temp_study


def manifest(study) -> dict:
    return json.loads((study_dir(study) / "manifest.json").read_text())
