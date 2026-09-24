"""Shared test setup: a temporary copy of the study folder, so tests never write into real participant data.

The prepared scans are not copied (they are large); the temporary folder links to them instead.
Tests that need the prepared study are skipped if scripts/prepare_study.py has not been run.
"""

import copy
import json
import threading
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
    fake["paths"]["study_dir"] = tmp_path
    root = study_dir(fake)
    root.mkdir(parents=True)
    for sub in ("scans", "queue"):
        (root / sub).symlink_to(real / sub)
    (root / "manifest.json").write_text((real / "manifest.json").read_text())
    return fake


@pytest.fixture()
def server(temp_study):
    """The review server on a free port in a background thread. Yields (base url, study config)."""
    from segreview.ui_server import make_server

    srv = make_server(temp_study, host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", temp_study
    srv.shutdown()
    srv.server_close()


def manifest(study) -> dict:
    return json.loads((study_dir(study) / "manifest.json").read_text())
