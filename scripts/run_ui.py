"""Start the review interface (G2) on this computer and open it in the browser.

Needs the prepared study files (scripts/prepare_study.py). Works without internet: NiiVue is in ui/vendor.
Stop the server with Ctrl+C.

Usage: uv run python scripts/run_ui.py --config configs/study.yaml [--no-browser]
"""

import argparse
import threading
import webbrowser
from pathlib import Path

from segreview.study import load_study_config
from segreview.ui_server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()
    study, _ = load_study_config(args.config)
    server = make_server(study)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"Review interface running at {url}  (stop with Ctrl+C)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
