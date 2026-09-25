"""Drive a headless Google Chrome from the tests, through the Chrome DevTools Protocol (CDP).

The browser tests use this to act like a participant: click buttons, press keys and drag with the
"mouse". Input sent with Input.dispatchMouseEvent goes through Chrome's normal input pipeline, so the
page receives the same pointer events as from a real trackpad drag.

Only a few CDP commands are used: Runtime.evaluate (run JavaScript in the page), Input.dispatchMouseEvent,
Input.dispatchKeyEvent and Page.captureScreenshot. Protocol reference: https://chromedevtools.github.io/devtools-protocol/
"""

import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket

CHROME_CANDIDATES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]

# Key codes CDP needs for the keys the tests press (key name -> (code, windowsVirtualKeyCode)).
KEYS = {"ArrowUp": ("ArrowUp", 38), "ArrowDown": ("ArrowDown", 40), "Enter": ("Enter", 13)}


def find_chrome() -> str | None:
    """Path of the Chrome executable, or None if Chrome is not installed."""
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


class Browser:
    """A headless Chrome with one page, controlled over a CDP websocket.

    Input: a folder for Chrome's temporary profile, and the window size in CSS pixels. The scale factor 2
    gives a Retina-like screen, as on the MacBooks used in the study.
    """

    def __init__(self, profile_dir: Path, width: int = 1440, height: int = 900, scale: int = 2):
        profile_dir.mkdir(parents=True, exist_ok=True)
        # On a Mac, headless Chrome draws WebGL with the real graphics card (Metal), like in the study.
        # Elsewhere (e.g. a Linux server without a GPU) it uses SwiftShader, a much slower software renderer.
        gpu = [] if sys.platform == "darwin" else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        self.proc = subprocess.Popen(
            [find_chrome(), "--headless=new", *gpu,
             "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile_dir}",
             "--remote-debugging-port=0", f"--window-size={width},{height}", f"--force-device-scale-factor={scale}",
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = profile_dir / "DevToolsActivePort"     # Chrome writes the chosen port here
        t0 = time.time()
        while not (port_file.exists() and port_file.read_text().strip()):
            if time.time() - t0 > 30:
                raise RuntimeError("Chrome did not start")
            time.sleep(0.2)
        port = int(port_file.read_text().split()[0])
        pages = []
        while not pages:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as r:
                pages = [t for t in json.loads(r.read()) if t["type"] == "page"]
        # No Origin header: Chrome only accepts DevTools connections without one (or from allowed origins).
        self.ws = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        self._id = 0

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.kill()
            self.proc.wait()

    def send(self, method: str, **params) -> dict:
        """Send one CDP command and wait for its reply (events that arrive in between are skipped)."""
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression: str):
        """Evaluate JavaScript in the page and return the value (promises are awaited)."""
        r = self.send("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JavaScript error: {r['exceptionDetails']}")
        return r["result"].get("value")

    def goto(self, url: str) -> None:
        self.send("Page.navigate", url=url)
        self.wait_for("document.readyState === 'complete' && typeof state !== 'undefined' && state.settings !== null")

    def wait_for(self, condition: str, timeout: float = 120) -> None:
        """Wait until a JavaScript expression is true in the page."""
        t0 = time.time()
        while not self.js(f"Boolean({condition})"):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"waited {timeout} s for: {condition}")
            time.sleep(0.1)

    def click(self, selector: str) -> None:
        """Click the center of an element with a real (trusted) mouse click."""
        x, y = self.center(selector)
        self.mouse("mouseMoved", x, y)
        self.mouse("mousePressed", x, y, buttons=1)
        self.mouse("mouseReleased", x, y)

    def center(self, selector: str) -> tuple[float, float]:
        """Center of an element in CSS pixels, after scrolling it into view (as a user would)."""
        rect = self.js(f"(() => {{ const el = document.querySelector({json.dumps(selector)});"
                       f" el.scrollIntoView({{block: 'nearest'}}); const r = el.getBoundingClientRect();"
                       f" return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
        return rect[0], rect[1]

    def mouse(self, kind: str, x: float, y: float, buttons: int = 0, **extra) -> None:
        """One mouse event at CSS pixel (x, y). kind: mouseMoved, mousePressed, mouseReleased or mouseWheel."""
        params = {"type": kind, "x": x, "y": y, "buttons": buttons, **extra}
        if kind in ("mousePressed", "mouseReleased"):
            params.update(button="left", clickCount=1)
        self.send("Input.dispatchMouseEvent", **params)

    def drag(self, points: list[tuple[float, float]]) -> None:
        """Press at the first point, move through the others, release at the last (a brush stroke)."""
        self.mouse("mouseMoved", *points[0])
        self.mouse("mousePressed", *points[0], buttons=1)
        for p in points[1:]:
            self.mouse("mouseMoved", *p, buttons=1)
        self.mouse("mouseReleased", *points[-1])

    def key(self, name: str) -> None:
        code, vk = KEYS[name]
        for kind in ("rawKeyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", type=kind, key=name, code=code, windowsVirtualKeyCode=vk)

    def screenshot(self, path: Path | None = None) -> bytes:
        """PNG of the whole page (device pixels). Saved to path if given."""
        png = base64.b64decode(self.send("Page.captureScreenshot", format="png")["data"])
        if path:
            Path(path).write_bytes(png)
        return png
