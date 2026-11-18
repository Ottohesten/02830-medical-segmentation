"""Tests of the G3 planning formulas (src/segreview/g3_estimates.py) with small hand-checked examples; the web server
is replaced by a fake one (no network)."""

import pytest

from segreview import g3_estimates
from segreview.g3_estimates import evaluation_hours, folder_bytes, remote_size, run_hours


def test_run_hours_counts_training_and_validation_steps():
    # 2,500 steps at 0.5 s = 1,250 s, plus 10 epochs x 50 validation steps at 0.2 s = 100 s
    assert run_hours(2500, 0.5, 0.2) == pytest.approx((1250 + 100) / 3600)
    assert run_hours(500, 1.0, 1.0, steps_per_epoch=100, validation_steps_per_epoch=10) == pytest.approx(550 / 3600)


def test_evaluation_hours():
    assert evaluation_hours(models=3, scans=10, predict_s=12.0) == pytest.approx(3 * 10 * 12 / 3600)


class FakeResponse:
    def __init__(self, status, headers):
        self.status_code, self.headers = status, headers
        self.is_redirect = status in (301, 302, 307, 308)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_remote_size_reads_the_size_headers(monkeypatch):
    answers = {("hf", False): FakeResponse(302, {"X-Linked-Size": "1234"}),          # Hugging Face: size in redirect
               ("plain", False): FakeResponse(200, {"Content-Length": "99"}),
               ("moved", False): FakeResponse(302, {}),
               ("moved", True): FakeResponse(200, {"Content-Length": "7"})}
    monkeypatch.setattr(g3_estimates.requests, "head",
                        lambda url, allow_redirects, timeout: answers[(url, allow_redirects)])
    assert remote_size("hf") == 1234
    assert remote_size("plain") == 99
    assert remote_size("moved") == 7


def test_folder_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"12345")
    (tmp_path / "y.bin").write_bytes(b"123")
    assert folder_bytes(tmp_path) == 8
    assert folder_bytes(tmp_path / "missing") == 0
