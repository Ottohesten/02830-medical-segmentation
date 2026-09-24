"""Small local web server for the review interface (G2). Standard library only.

It serves the page in ui/ (with NiiVue from ui/vendor), the prepared scan files, the participant's
scan list, and it stores what the participant produces: the corrected mask and a log per scan.

Endpoints:
  GET  /                                        the interface (ui/index.html)
  GET  /ui/<file>                               static files (JavaScript, CSS, NiiVue)
  GET  /api/settings                            viewer settings and time limit from the study config
  GET  /api/study/<participant>                 the participant's practice + study scans, in order, with 'done' flags
  GET  /api/queue                               demo queue: scans ranked by the G1 score
  GET  /files/<participant>/<set>/<case>/<name>.nii.gz   ct, mask or heatmap (set = scans or queue)
  POST /api/save/<mode>/<participant>/<case>    body = the corrected mask as NIfTI bytes (from NiiVue)
  POST /api/log/<mode>/<participant>/<case>     body = JSON log of the scan (times, events)

The heatmap of a study scan is only handed out if that participant has the scan in the WITH-heatmap
condition, so the "without" condition cannot show it by mistake. Only known scan ids and file names are
accepted, and the server only listens on this computer (127.0.0.1).

Saved files (data/study/<study name>/):
  sessions/<participant>/<case>_mask.nii.gz and <case>_log.json   study scans
  sessions/<participant>/practice/...                             practice scans
  queue_edits/<case>_mask.nii.gz and <case>_log.json              demo queue
"""

import gzip
import json
import mimetypes
import time
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nibabel as nib

from segreview.config import REPO_ROOT
from segreview.study import WITH, assignment, load_manifest, participant_type, study_dir

UI_DIR = REPO_ROOT / "ui"
FILE_NAMES = {"ct", "mask", "heatmap"}
MODES = {"study", "practice", "queue"}


class StudyServer:
    """Everything the request handler needs: config, manifest, and where to read and write files."""

    def __init__(self, study: dict):
        self.study = study
        self.root = study_dir(study)
        self.manifest = load_manifest(study)
        self.prefix = study["study"]["participant_prefix"]

    def output_dir(self, mode: str, participant: str) -> Path:
        """Where a scan's corrected mask and log are written."""
        if mode == "queue":
            return self.root / "queue_edits"
        base = self.root / "sessions" / participant
        return base / "practice" if mode == "practice" else base

    def plan(self, participant: str) -> dict:
        """Practice + study scans for one participant, with the condition and whether each is done."""
        scans = assignment(participant, self.manifest["study_scans"], self.prefix)
        for s in scans:
            s["done"] = (self.output_dir("study", participant) / f"{s['case_id']}_log.json").exists()
        practice = [{"case_id": c, "condition": WITH, "position": i + 1,
                     "done": (self.output_dir("practice", participant) / f"{c}_log.json").exists()}
                    for i, c in enumerate(self.manifest["practice_scans"])]
        return {"participant": participant, "type": participant_type(participant, self.prefix),
                "practice": practice, "scans": scans}

    def heatmap_allowed(self, participant: str, file_set: str, case_id: str) -> bool:
        """The heatmap is shown in the demo queue, in practice, and for study scans in the WITH condition."""
        if file_set == "queue" or case_id in self.manifest["practice_scans"]:
            return True
        return any(s["case_id"] == case_id and s["condition"] == WITH
                   for s in assignment(participant, self.manifest["study_scans"], self.prefix))

    def known_case(self, file_set: str, case_id: str) -> bool:
        if file_set == "queue":
            return case_id in self.manifest["queue_scans"]
        return case_id in self.manifest["study_scans"] + self.manifest["practice_scans"]


class Handler(BaseHTTPRequestHandler):
    """Handles one HTTP request. 'app' is the StudyServer (bound with functools.partial)."""

    def __init__(self, *args, app: StudyServer, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # keep the terminal quiet; errors are still reported
        pass

    # --- helpers -------------------------------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    # --- GET -----------------------------------------------------------------------------------
    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if not parts:
                return self._static(UI_DIR / "index.html")
            if parts[0] == "ui":
                target = (UI_DIR / "/".join(parts[1:])).resolve()
                if UI_DIR.resolve() not in target.parents:
                    return self._error(HTTPStatus.FORBIDDEN, "outside ui/")
                return self._static(target)
            if parts[:2] == ["api", "settings"]:
                s = self.app.study
                return self._json({"viewer": s["viewer"], "time_limit_min": s["study"]["time_limit_min"],
                                   "participant_prefix": self.app.prefix})
            if parts[:2] == ["api", "study"] and len(parts) == 3:
                return self._json(self.app.plan(parts[2]))
            if parts[:2] == ["api", "queue"]:
                m = self.app.manifest
                return self._json([{"case_id": c, "score": m["queue_scores"][c], "rank": i + 1}
                                   for i, c in enumerate(m["queue_scans"])])
            if parts[0] == "files" and len(parts) == 5:
                return self._file(*parts[1:])
            return self._error(HTTPStatus.NOT_FOUND, "unknown path")
        except ValueError as e:  # e.g. a malformed participant id
            return self._error(HTTPStatus.BAD_REQUEST, str(e))

    def _static(self, path: Path) -> None:
        if not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "no such file")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), ctype)

    def _file(self, participant: str, file_set: str, case_id: str, filename: str) -> None:
        name = filename.removesuffix(".nii.gz")
        if file_set not in {"scans", "queue"} or name not in FILE_NAMES or not self.app.known_case(file_set, case_id):
            return self._error(HTTPStatus.NOT_FOUND, "unknown scan or file")
        if name == "heatmap" and not self.app.heatmap_allowed(participant, file_set, case_id):
            return self._error(HTTPStatus.FORBIDDEN, "no heatmap in this condition")
        self._send(HTTPStatus.OK, (self.app.root / file_set / case_id / filename).read_bytes(), "application/gzip")

    # --- POST ----------------------------------------------------------------------------------
    def do_POST(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if len(parts) != 5 or parts[0] != "api" or parts[1] not in {"save", "log"} or parts[2] not in MODES:
            return self._error(HTTPStatus.NOT_FOUND, "unknown path")
        _, kind, mode, participant, case_id = parts
        try:
            if mode != "queue":
                participant_type(participant, self.app.prefix)  # validates the id
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        file_set = "queue" if mode == "queue" else "scans"
        if not self.app.known_case(file_set, case_id):
            return self._error(HTTPStatus.NOT_FOUND, "unknown scan")
        out = self.app.output_dir(mode, participant)
        out.mkdir(parents=True, exist_ok=True)
        body = self._body()
        if kind == "save":
            return self._save_mask(body, out, file_set, case_id)
        log = json.loads(body)
        log.update({"case_id": case_id, "participant": participant, "mode": mode,
                    "server_received_unix": time.time()})
        (out / f"{case_id}_log.json").write_text(json.dumps(log, indent=2))
        return self._json({"ok": True})

    def _save_mask(self, body: bytes, out: Path, file_set: str, case_id: str) -> None:
        """Check that the uploaded NIfTI has the scan's shape, then store it gzipped."""
        data = body if body[:2] == b"\x1f\x8b" else gzip.compress(body)
        tmp = out / f"{case_id}_mask.upload.nii.gz"
        tmp.write_bytes(data)
        try:
            shape = nib.load(tmp).shape
        except Exception as e:  # not a readable NIfTI
            tmp.unlink()
            return self._error(HTTPStatus.BAD_REQUEST, f"not a NIfTI image: {e}")
        expected = nib.load(self.app.root / file_set / case_id / "ct.nii.gz").shape
        if tuple(shape[:3]) != tuple(expected[:3]):
            tmp.unlink()
            return self._error(HTTPStatus.BAD_REQUEST, f"mask shape {shape} differs from scan {expected}")
        tmp.rename(out / f"{case_id}_mask.nii.gz")
        return self._json({"ok": True, "shape": list(shape)})


def make_server(study: dict, host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    """Create (but do not start) the server for a study config. Port 0 picks a free port (for tests)."""
    app = StudyServer(study)
    host = host or study["server"]["host"]
    port = study["server"]["port"] if port is None else port
    return ThreadingHTTPServer((host, port), partial(Handler, app=app))
