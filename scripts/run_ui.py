"""Start the review interface (G2) on this computer and open it in the browser.

Needs the prepared study files (scripts/prepare_study.py). Works without internet: NiiVue is in ui/vendor.
Which browser opens is set by server.browser in the study config: "chrome" (the study only runs in Google
Chrome; the default), "default" (the computer's default browser) or "none". Stop the server with Ctrl+C.

Usage: uv run python scripts/run_ui.py --config configs/study.yaml [--no-browser]
"""

import argparse
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from segreview.study import load_study_config
from segreview.ui_server import make_server

BROWSERS = ("chrome", "default", "none")


def open_browser(url: str, choice: str) -> None:
    """Open the interface in the chosen browser (see the module docstring). Prints a hint if that fails."""
    if choice == "none":
        return
    if choice == "default":
        webbrowser.open(url)
        return
    # Chrome. On macOS "open -a" starts the app even if another browser is the default.
    if sys.platform == "darwin":
        if subprocess.run(["open", "-a", "Google Chrome", url], capture_output=True).returncode == 0:
            return
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
            if shutil.which(name):
                subprocess.Popen([name, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
    print(f"Could not start Google Chrome. Open {url} in Chrome yourself.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="study config, e.g. configs/study.yaml")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser (overrides server.browser)")
    args = parser.parse_args()
    study, _ = load_study_config(args.config)
    choice = "none" if args.no_browser else study["server"].get("browser", "chrome")
    if choice not in BROWSERS:
        raise SystemExit(f"server.browser must be one of {BROWSERS}, not '{choice}'")
    server = make_server(study)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"Review interface running at {url}  (stop with Ctrl+C)")
    threading.Timer(0.5, open_browser, args=(url, choice)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
