"""End-to-end tests of the review page in a real browser (headless Google Chrome), used like a participant would.

Chrome gets trusted mouse, wheel and key input through the DevTools protocol (tests/browser.py), so the page
receives the same pointer events as from a trackpad. The tests cover the requirements from the first user test
of the interface (24 September):
- painting and erasing with pointer events change the SAVED mask exactly where the brush went (round brush,
  the circle has the brush's size), and undo takes a stroke back;
- the Move tool, a two-finger swipe and a pinch move/zoom the image without painting; keys and buttons change slice;
- the mask is drawn on top of the heatmap;
- the toolbar only shows when a scan is open;
- the study flow: guide -> practice -> "time is up" lock -> correct answer -> study scans (without heatmap
  where the balancing says so) -> NASA-TLX -> end.
Skipped if Chrome is not installed.
"""

import io
import json
import time

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from segreview.study import WITH, assignment, study_dir
from tests.browser import Browser, find_chrome
from tests.conftest import manifest

pytestmark = pytest.mark.skipif(find_chrome() is None, reason="Google Chrome not installed")
SLACK = 1.5   # voxels; see test_painting_with_pointer_events_changes_the_saved_mask

# Helper functions installed in the page (window.t). Only the tests use them.
PAGE_HELPERS = """
window.t = {
  // CSS pixel position of voxel (x, y) of the current slice (inverse of the viewer's own mapping).
  screenOf(x, y) {
    const v = state.viewer, nv = v.nv, s = nv.screenSlices[0], w = s.leftTopWidthHeight, dpr = nv.uiData.dpr;
    const mm = nv.frac2mm(nv.vox2frac([x, y, v.slice]));
    const r = nv.canvas.getBoundingClientRect();
    return [r.left + (w[0] + ((mm[0] - s.leftTopMM[0]) / s.fovMM[0]) * w[2]) / dpr,
            r.top + (w[1] + (1 - (mm[1] - s.leftTopMM[1]) / s.fovMM[1]) * w[3]) / dpr];
  },
  maskSum() { return state.viewer.nv.drawBitmap.reduce((a, b) => a + (b > 0), 0); },
  // A mask voxel with only mask around it (radius m), preferring a high heatmap value if there is a heatmap.
  interiorVoxel(m) {
    const v = state.viewer, nv = v.nv, [nx, ny, nz] = v.dims, bmp = nv.drawBitmap, heat = nv.volumes[1];
    let best = null;
    for (let z = 0; z < nz; z++) for (let y = m; y < ny - m; y++) for (let x = m; x < nx - m; x++) {
      const i = x + y * nx + z * nx * ny;
      if (!bmp[i]) continue;
      const h = heat ? heat.img[i] * (heat.hdr.scl_slope || 1) : 0;
      if (best && h <= best.h) continue;
      let inside = true;
      for (let dy = -m; dy <= m && inside; dy++) for (let dx = -m; dx <= m; dx++) if (!bmp[i + dx + dy * nx]) { inside = false; break; }
      if (inside) best = { x, y, z, h };
    }
    return best;
  },
  frame() { return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); },
};
true;
"""


SCALE = 2   # device pixels per CSS pixel, as on the Retina screens of the study MacBooks


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    b = Browser(tmp_path_factory.mktemp("chrome"), scale=SCALE)
    yield b
    b.close()


def open_queue_scan(b, base, index=0):
    """Start page -> demo queue -> the index-th scan, like a user clicking. Returns its case id."""
    b.goto(base + "/")
    assert b.js("document.querySelector('.toolbar').getClientRects().length") == 0   # no toolbar without a scan
    b.click("#start-queue")
    b.wait_for("document.querySelectorAll('#queue-list button').length > 0")
    case_id = b.js(f"document.querySelectorAll('#queue-list button')[{index}].textContent.split(' ')[0]")
    b.click(f"#queue-list li:nth-child({index + 1}) button")
    b.wait_for("document.getElementById('viewer').dataset.mode === 'edit'")
    b.js(PAGE_HELPERS)
    return case_id


def pixel(png: bytes, x: float, y: float) -> np.ndarray:
    """RGB of the screenshot at CSS pixel (x, y)."""
    return np.array(Image.open(io.BytesIO(png)).convert("RGB").getpixel((round(x * SCALE), round(y * SCALE))), int)


def near_segment(shape, a, b, radius):
    """Boolean (x, y) map of the voxels whose centre is within radius of the segment a-b (voxel units)."""
    xs, ys = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
    ab = b - a
    t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / (ab @ ab), 0, 1) if ab.any() else 0
    return (xs - a[0] - t * ab[0]) ** 2 + (ys - a[1] - t * ab[1]) ** 2 <= radius ** 2


def saved_and_original(study, case_id):
    root = study_dir(study)
    original = np.asanyarray(nib.load(root / "queue" / case_id / "mask.nii.gz").dataobj) > 0
    saved_img = nib.load(root / "queue_edits" / f"{case_id}_mask.nii.gz")
    ct = nib.load(root / "queue" / case_id / "ct.nii.gz")
    assert saved_img.shape == ct.shape and np.allclose(saved_img.affine, ct.affine)
    return np.asanyarray(saved_img.dataobj) > 0, original


def finish_queue_scan(b, study, case_id):
    """Press Done and wait until the mask and log are saved (the page goes back to the queue)."""
    b.click("#done")
    b.wait_for("!document.getElementById('queue').hidden")
    return json.loads((study_dir(study) / "queue_edits" / f"{case_id}_log.json").read_text())


def test_painting_with_pointer_events_changes_the_saved_mask(server, browser):
    base, study = server
    b = browser
    case_id = open_queue_scan(b, base)
    pen = b.js("state.viewer.penSize")
    r = pen / 2
    # A single click (a round stamp) and a stroke, both near the image centre, on the start slice.
    cx, cy = b.js("(() => { const r = state.viewer.canvas.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()")
    click = (cx - 60, cy + 40)
    stroke = [(cx - 20 + 6 * i, cy - 30 + 2 * i) for i in range(15)]
    v_click = b.js(f"state.viewer.voxelAt({click[0]}, {click[1]})")
    v_a, v_b = b.js(f"state.viewer.voxelAt({stroke[0][0]}, {stroke[0][1]})"), b.js(f"state.viewer.voxelAt({stroke[-1][0]}, {stroke[-1][1]})")
    z = b.js("state.viewer.slice")
    assert v_click[2] == v_a[2] == z

    # The brush circle shows the brush's real size on screen.
    b.mouse("mouseMoved", *click)
    size = b.js("parseFloat(document.getElementById('brush').style.width)")
    expected = b.js("state.viewer.penSize * state.viewer.nv.back.pixDimsRAS[1] * state.viewer.cssPxPerMm()")
    assert b.js("!document.getElementById('brush').hidden") and size == pytest.approx(expected, rel=1e-3)

    before = b.screenshot()
    b.drag([click])
    b.drag(stroke)
    b.js("t.frame()")
    after = b.screenshot()
    # The paint shows where the pointer was: the pixel under the stroke turned red.
    mid = stroke[7]
    p0, p1 = pixel(before, *mid), pixel(after, *mid)
    assert p1[0] - max(p1[1], p1[2]) > 60 and abs(p0[0] - p0[1]) < 30, (p0, p1)

    log = finish_queue_scan(b, study, case_id)
    saved, original = saved_and_original(study, case_id)
    assert (saved | original).sum() == saved.sum()                  # adding never removes anything
    added = saved & ~original
    assert not added[:, :, np.arange(added.shape[2]) != z].any()    # only the slice that was shown
    shape = added.shape[:2]
    disc = near_segment(shape, v_click, v_click, r)                 # the click: exactly the round brush
    region_click = near_segment(shape, v_click, v_click, r + 1)
    assert np.array_equal(added[:, :, z] & region_click, disc & ~original[:, :, z])
    # The stroke: everything near the path is painted, nothing further away. The tolerance (SLACK) is there
    # because each pointer position is rounded to a voxel (up to 0.71 voxel off the line) and so is each
    # brush stamp between two positions (another 0.71).
    painted_stroke = added[:, :, z] & ~region_click
    assert not (painted_stroke & ~near_segment(shape, v_a, v_b, r + SLACK)).any()
    assert not (near_segment(shape, v_a, v_b, r - SLACK) & ~saved[:, :, z]).any()
    assert log["reason"] == "done" and log["strokes"] == 2 and log["time_limit_s"] is None
    assert [e["type"] for e in log["events"]].count("stroke") == 2


def test_erase_and_undo(server, browser):
    base, study = server
    b = browser
    case_id = open_queue_scan(b, base, index=1)
    target = b.js("t.interiorVoxel(6)")
    b.js(f"state.viewer.setSlice({target['z']})")
    total = b.js("t.maskSum()")
    b.click("#tool-erase")
    a, c = b.js(f"t.screenOf({target['x'] - 4}, {target['y']})"), b.js(f"t.screenOf({target['x'] + 4}, {target['y']})")
    path = [(a[0] + (c[0] - a[0]) * i / 8, a[1] + (c[1] - a[1]) * i / 8) for i in range(9)]
    b.drag(path)
    assert b.js("t.maskSum()") < total
    b.click("#undo")
    assert b.js("t.maskSum()") == total                               # undo takes the whole stroke back
    b.drag(path)
    v_a, v_b = b.js(f"state.viewer.voxelAt({path[0][0]}, {path[0][1]})"), b.js(f"state.viewer.voxelAt({path[-1][0]}, {path[-1][1]})")
    r = b.js("state.viewer.penSize") / 2
    log = finish_queue_scan(b, study, case_id)
    saved, original = saved_and_original(study, case_id)
    removed = original & ~saved
    z = target["z"]
    assert not (saved & ~original).any() and removed.any()
    assert not removed[:, :, np.arange(removed.shape[2]) != z].any()
    assert not (removed[:, :, z] & ~near_segment(removed.shape[:2], v_a, v_b, r + SLACK)).any()
    assert not (near_segment(removed.shape[:2], v_a, v_b, r - SLACK) & saved[:, :, z]).any()
    assert log["undo_count"] == 1 and log["strokes"] == 2


def test_move_zoom_and_slices_do_not_paint(server, browser):
    base, _ = server
    b = browser
    open_queue_scan(b, base)
    total = b.js("t.maskSum()")
    cx, cy = b.js("(() => { const r = state.viewer.canvas.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()")
    mm_at = lambda x, y: np.array(b.js(f"state.viewer.mmAt({x}, {y})"))

    # Move tool: the point under the pointer follows the pointer.
    b.click("#tool-move")
    start = mm_at(cx, cy)
    b.drag([(cx + 12 * i, cy + 5 * i) for i in range(11)])
    assert np.allclose(mm_at(cx + 120, cy + 50), start, atol=0.2)
    # Two-finger swipe (a wheel event): the image moves with the fingers.
    before = mm_at(cx, cy)
    b.mouse("mouseWheel", cx, cy, deltaX=0, deltaY=40)
    assert np.allclose(mm_at(cx, cy - 40), before, atol=0.2)
    # Pinch (a wheel event with Ctrl): zooms in around the pointer, which stays on the same point.
    before = mm_at(cx - 100, cy)
    b.mouse("mouseWheel", cx - 100, cy, deltaX=0, deltaY=-60, modifiers=2)
    assert b.js("state.viewer.zoom") > 1.5
    assert np.allclose(mm_at(cx - 100, cy), before, atol=0.2)
    b.click("#reset-view")
    assert b.js("state.viewer.zoom") == 1

    # Slices: arrow keys and the buttons; the slice number follows.
    z = b.js("state.viewer.slice")
    b.key("ArrowUp")
    b.key("ArrowUp")
    b.click("#slice-down")
    assert b.js("state.viewer.slice") == z + 1
    assert b.js("document.getElementById('slice-number').textContent") == f"Slice {z + 2} / {b.js('state.viewer.nSlices')}"
    b.click("#slice-track")     # the middle of the slider
    assert abs(b.js("state.viewer.slice") - b.js("state.viewer.nSlices") / 2) <= 1
    assert b.js("t.maskSum()") == total                               # none of this painted anything
    # With the heatmap on, the high-uncertainty slices are marked on the slider, and blue is explained.
    assert b.js("state.marks.length") > 0 and b.js("state.heatmapOn")
    visible = "(id) => document.getElementById(id).getClientRects().length > 0"
    assert b.js(f"({visible})('key') && ({visible})('key-blue')")
    b.click("#heatmap-toggle")
    assert b.js(f"({visible})('key') && !({visible})('key-blue')")     # heatmap off: no blue in the key
    # The toolbar is hidden while a scan loads (checked through the CSS rule).
    b.js("document.getElementById('viewer').dataset.mode = 'loading'")
    assert b.js("getComputedStyle(document.getElementById('done')).visibility") == "hidden"


def test_mask_is_drawn_on_top_of_the_heatmap(server, browser):
    base, _ = server
    b = browser
    open_queue_scan(b, base)
    target = b.js("t.interiorVoxel(3)")
    assert target["h"] > 0.3, "need a mask voxel where the heatmap is clearly drawn"
    b.js(f"state.viewer.setSlice({target['z']})")
    x, y = b.js(f"t.screenOf({target['x']}, {target['y']})")
    b.js(f"state.viewer.zoomAt(4, {x}, {y})")
    x, y = b.js(f"t.screenOf({target['x']}, {target['y']})")
    # Make both layers fully opaque: then the pixel shows whichever layer is drawn last.
    b.js("state.viewer.nv.setOpacity(1, 1); state.viewer.nv.setDrawOpacity(1); state.viewer.nv.drawScene(); t.frame()")
    on_top = pixel(b.screenshot(), x, y)
    b.js("state.viewer.nv.setDrawOpacity(0); state.viewer.nv.drawScene(); t.frame()")
    heat_only = pixel(b.screenshot(), x, y)
    mask_rgb = np.array(b.js("state.settings.viewer.mask_color"))
    assert np.abs(on_top - mask_rgb).max() <= 12, (on_top, heat_only)
    assert heat_only[2] > heat_only[0] + 50, heat_only                # without the mask the (blue) heatmap is there


def test_study_flow_guide_practice_timeout_answer(quick_server, browser):
    base, study = quick_server
    b = browser
    m = manifest(study)
    root = study_dir(study)
    b.goto(base + "/")
    b.js("document.getElementById('participant').value = 'P01'")
    b.click("#start-study")
    # Guide with the example image, before the practice scan.
    b.wait_for("!document.getElementById('guide').hidden")
    b.wait_for("document.getElementById('guide-image').naturalWidth > 0", timeout=20)
    assert b.js("document.querySelector('.toolbar').getClientRects().length") == 0
    b.click("#guide-go")
    b.wait_for("document.getElementById('intro-title').textContent === 'Practice scan'")
    b.click("#intro-go")
    b.wait_for("document.getElementById('viewer').dataset.mode === 'edit'")
    assert b.js("state.heatmapOn")                                       # the practice has the heatmap
    # Let the 4-second time limit run out: the scan is locked, saved, and a message says so.
    b.wait_for("!document.getElementById('lock').hidden", timeout=30)
    assert b.js("document.getElementById('lock-title').textContent") == "Time is up"
    assert not b.js("state.viewer.paintEnabled")
    b.wait_for("document.getElementById('lock-text').textContent.includes('have been saved')", timeout=30)
    practice = m["practice_scans"][0]
    log = json.loads((root / "sessions" / "P01" / "practice" / f"{practice}_log.json").read_text())
    assert log["reason"] == "timeout" and log["duration_s"] >= 4
    # Continue -> the correct answer for the practice scan (comparison in the drawing layer).
    b.click("#lock-button")
    b.wait_for("document.getElementById('viewer').dataset.mode === 'answer'")
    assert b.js("!document.getElementById('legend').hidden")
    assert b.js("state.viewer.nv.drawBitmap.some((v) => v >= 2)")       # missed or extra voxels are shown
    # Continue -> first study scan. P01 has scans 1-3 WITHOUT the heatmap.
    b.click("#answer-continue")
    b.wait_for("document.getElementById('intro-title').textContent === 'Scan 1 of 6'")
    first = assignment("P01", m["study_scans"], "P")[0]
    assert first["condition"] != WITH
    b.click("#intro-go")
    b.wait_for("document.getElementById('viewer').dataset.mode === 'edit'")
    assert b.js("getComputedStyle(document.getElementById('heatmap-toggle')).display") == "none"
    assert b.js("document.getElementById('key').getClientRects().length > 0")          # task + red explained
    assert b.js("document.getElementById('key-blue').getClientRects().length") == 0    # no blue to explain
    assert b.js("state.viewer.nv.volumes.length") == 1 and b.js("state.marks.length") == 0
    assert b.js("state.viewer.truthUrl") is None                        # no ground truth for study scans
    b.click("#done")
    b.wait_for("document.getElementById('intro-title').textContent === 'Scan 2 of 6'")
    log = json.loads((root / "sessions" / "P01" / f"{first['case_id']}_log.json").read_text())
    assert log["reason"] == "done" and not log["heatmap_available"] and log["time_limit_s"] == pytest.approx(4)
    assert (root / "sessions" / "P01" / f"{first['case_id']}_mask.nii.gz").exists()

    # Scans 2-6 (pressing Done right away), then the NASA-TLX questionnaire as the last step.
    for k in range(2, 7):
        b.click("#intro-go")
        b.wait_for("document.getElementById('viewer').dataset.mode === 'edit'")
        b.click("#done")
        b.wait_for(f"document.getElementById('intro-title').textContent === 'Scan {k + 1} of 6'"
                   if k < 6 else "!document.getElementById('tlx').hidden")
    assert b.js("document.getElementById('tlx-submit').disabled")          # nothing answered yet
    for i in range(6):                                                       # click each scale somewhere
        x0, y0 = b.center(f".tlx-scale:nth-child({i + 1}) input")
        width = b.js(f"document.querySelectorAll('.tlx-range')[{i}].getBoundingClientRect().width")
        b.mouse("mousePressed", x0 - width / 2 + width * (0.1 + 0.15 * i), y0, buttons=1)
        b.mouse("mouseReleased", x0 - width / 2 + width * (0.1 + 0.15 * i), y0)
    answers = b.js("Object.fromEntries([...document.querySelectorAll('.tlx-range')].map((r) => [r.dataset.key, r.valueAsNumber]))")
    assert len(set(answers.values())) == 6                                   # each click set its own value
    b.click("#tlx-submit")
    b.wait_for("!document.getElementById('end').hidden")
    tlx = json.loads((root / "sessions" / "P01" / "tlx_session.json").read_text())
    assert tlx["scales"] == answers and tlx["raw_tlx"] == pytest.approx(sum(answers.values()) / 6)
