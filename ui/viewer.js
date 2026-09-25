// Slice viewer for one scan (G2, pipeline step 5), built on NiiVue. Used by app.js.
//
// NiiVue draws three layers: the CT, the uncertainty heatmap (an "overlay" volume) and the editable mask
// (NiiVue's "drawing" layer). This file adds what the review study needs on top of NiiVue:
// - Layer order. NiiVue's own slice shader puts overlays on top of the drawing, so the heatmap covered the
//   mask. Our shader (sliceShader below) draws CT -> heatmap -> mask, so the mask is always on top, with a
//   solid outline along its edge.
// - Our own mouse/trackpad handling with pointer events: a round brush with a circle that shows its size,
//   a Move tool, pinch to zoom, and a two-finger swipe that moves the image. NiiVue's own mouse, wheel and
//   key handling is blocked, so no other (hidden) actions exist, such as right-drag or keyboard shortcuts.
// - Slice navigation, and the list of the most uncertain slices for the slider (with-heatmap condition).
// - A comparison with the ground truth (the answer after the practice scan, and in the demo queue).
//
// Coordinates: the prepared scan files are in canonical RAS orientation (x = towards the patient's right,
// y = towards the front, z = towards the head). NiiVue then keeps the voxels in the file's order, so
// voxel (x, y, z) is at index x + y * nx + z * nx * ny in the drawing and in every loaded volume.

const DRAW_TEXTURE_UNIT = 7;   // NiiVue keeps the drawing in texture unit 7 (TEXTURE7)
const MASK = 1, MISSED = 2, EXTRA = 3;   // values in the drawing layer (MISSED/EXTRA only in the comparison)

// Fragment shader for the 2D slices. NiiVue compiles it with its own vertex shader and fills in the
// uniforms (texture units: volume 0, colormap 1, overlay 2, drawing 7). NiiVue finds the uniform names
// with a simple text search that only sees one name per line, so each uniform has its own line.
function sliceShader(outlineDevicePx) {
  const px = outlineDevicePx.toFixed(1);
  return `#version 300 es
precision highp int;
precision highp float;
uniform highp sampler3D volume;
uniform highp sampler3D overlay;
uniform highp sampler3D drawing;
uniform highp sampler2D colormap;
uniform float overlays;
uniform float overlayAlphaShader;
uniform float opacity;
uniform float drawOpacity;
uniform bool isAlphaClipDark;
uniform int backgroundMasksOverlays;
in vec3 texPos;
out vec4 color;

// Color of a drawing value, from NiiVue's drawing color table (the last row of the colormap texture).
vec4 drawColor(float value) {
  float rows = float(textureSize(colormap, 0).y);
  return texture(colormap, vec2((value * 255.0) / 256.0 + 0.5 / 256.0, (rows - 0.5) / rows));
}

void main() {
  // How far a few screen pixels are in texture coordinates (for the outline). Computed before any "if":
  // screen-space derivatives are only defined when neighboring pixels run the same code.
  vec3 dx = dFdx(texPos) * ${px};
  vec3 dy = dFdy(texPos) * ${px};

  // 1. The CT (gray values after the contrast window).
  vec4 background = texture(volume, texPos);
  color = vec4(background.rgb, opacity);
  if (isAlphaClipDark && background.a == 0.0) color.a = 0.0;

  // 2. The heatmap, blended over the CT.
  if (overlays > 0.0 && !(backgroundMasksOverlays > 0 && background.a == 0.0)) {
    vec4 ocolor = texture(overlay, texPos);
    ocolor.a *= overlayAlphaShader;
    float a = color.a + ocolor.a * (1.0 - color.a);
    if (a > 0.0) {
      color.rgb = mix(color.rgb, ocolor.rgb, ocolor.a / a);
      color.a = a;
    }
  }

  // 3. The mask LAST, so nothing covers it: a see-through fill, and a solid line where a pixel a few
  //    screen pixels away has another value (the edge of the mask).
  float v = texture(drawing, texPos).r;
  if (v > 0.0) {
    vec4 dcolor = drawColor(v);
    bool edge = texture(drawing, texPos + dx).r != v || texture(drawing, texPos - dx).r != v
             || texture(drawing, texPos + dy).r != v || texture(drawing, texPos - dy).r != v;
    float a = (edge ? 1.0 : drawOpacity) * dcolor.a;
    color.rgb = mix(color.rgb, dcolor.rgb, a);
    color.a = max(color.a, a);
  }
}`;
}

class ReviewViewer {
  /**
   * canvas: the <canvas> NiiVue draws in; brush: the <div> drawn as the brush circle.
   * settings: the "viewer" part of configs/study.yaml.
   * hooks: callbacks {onSlice(z), onStroke(info), onView()} that app.js uses for the slider and the log.
   */
  constructor(canvas, brush, settings, hooks = {}) {
    this.canvas = canvas;
    this.brush = brush;
    this.s = settings;
    this.hooks = hooks;
    this.nv = null;
    this.tool = "add";          // add | erase | move
    this.penSize = settings.default_pen_size;
    this.paintEnabled = false;  // false while loading, when locked, and in the comparison
    this.drag = null;           // the stroke or pan in progress
    this.truth = null;          // ground truth (Uint8Array), loaded on demand
    this.userDrawing = null;    // the participant's mask, kept while the comparison is shown
    this.scrollRest = 0;        // leftover swipe distance (two_finger_scroll: slice)
  }

  // ---------- setup ----------
  async init() {
    const nv = new niivue.Niivue({
      backColor: [0.07, 0.07, 0.07, 1],
      isColorbar: false,
      show3Dcrosshair: false,
      crosshairWidth: 0,
      isOrientationTextVisible: false,   // the page shows its own R / L labels
      isRadiologicalConvention: true,    // patient's right on the left of the screen, as radiologists view CT
    });
    await nv.attachTo(this.canvas.id);
    nv.setSliceType(nv.sliceTypeAxial);
    nv.setCustomSliceShader(sliceShader(this.s.mask_outline_px * nv.uiData.dpr));
    const c = this.s.compare_colors, m = this.s.mask_color;
    nv.setDrawColormap({ R: [0, m[0], c.missed[0], c.extra[0]], G: [0, m[1], c.missed[1], c.extra[1]],
                         B: [0, m[2], c.missed[2], c.extra[2]], A: [0, 255, 255, 255],
                         labels: ["", "mask", "missed", "extra"] });
    this.canvas.removeAttribute("tabindex");   // the canvas never takes the keyboard focus (NiiVue's shortcuts)
    this.nv = nv;
    this.bindInput();
  }

  bindInput() {
    const wrap = this.canvas.parentElement;
    // Block NiiVue's own handlers. They are registered on the canvas; an event stopped here, while it is on
    // its way down through the canvas's parent (capture phase), never reaches the canvas.
    for (const type of ["mousedown", "mouseup", "mousemove", "mouseleave", "mouseenter", "dblclick", "contextmenu",
                        "touchstart", "touchmove", "touchend", "keydown", "keyup"]) {
      wrap.addEventListener(type, (e) => {
        e.stopPropagation();
        if (type === "contextmenu") e.preventDefault();
      }, true);
    }
    wrap.addEventListener("wheel", (e) => this.onWheel(e), { capture: true, passive: false });
    this.canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    this.canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    this.canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
    this.canvas.addEventListener("pointerleave", () => { if (!this.drag) this.brush.hidden = true; });
  }

  /** Load one scan. urls: {ct, mask, heatmap (optional), truth (optional, loaded only when shown)}. */
  async load(urls) {
    const nv = this.nv;
    this.paintEnabled = false;
    this.drag = null;
    this.truth = null;
    this.truthUrl = urls.truth || null;
    this.userDrawing = null;
    const volumes = [{ url: urls.ct, colormap: "gray" }];
    if (urls.heatmap) {
      volumes.push({ url: urls.heatmap, colormap: this.s.heatmap_colormap, cal_min: this.s.heatmap_min, cal_max: 1,
                     opacity: this.s.heatmap_opacity });
    }
    nv.closeDrawing();
    await nv.loadVolumes(volumes);
    for (const v of nv.volumes) {
      const p = Array.from(v.permRAS);
      if (p.join() !== "1,2,3") throw new Error(`scan is not in RAS orientation (permRAS ${p})`);
    }
    if (!(await nv.loadDrawingFromUrl(urls.mask, true))) throw new Error("the mask could not be loaded");
    nv.setDrawingEnabled(true);
    nv.setDrawOpacity(this.s.mask_opacity);
    nv.scene.crosshairPos = nv.vox2frac([Math.floor(this.dims[0] / 2), Math.floor(this.dims[1] / 2),
                                         Math.floor(this.dims[2] / 2)]);
    this.resetView();
    this.hooks.onSlice?.(this.slice);
  }

  get dims() { return Array.from(this.nv.back.dims.slice(1, 4)); }        // [nx, ny, nz]
  get nSlices() { return this.dims[2]; }
  get slice() { return Math.round(this.nv.frac2vox(this.nv.scene.crosshairPos)[2]); }
  get zoom() { return this.nv.scene.pan2Dxyzmm[3]; }

  // ---------- tools and display ----------
  setTool(tool) {
    this.tool = tool;
    this.canvas.dataset.tool = tool;      // CSS picks the mouse cursor
  }

  setPenSize(size) { this.penSize = size; }

  setWindow(preset) {
    const ct = this.nv.volumes[0];
    ct.cal_min = preset.level - preset.width / 2;
    ct.cal_max = preset.level + preset.width / 2;
    this.nv.updateGLVolume();
  }

  setHeatmapVisible(on) {
    if (this.nv.volumes.length > 1) this.nv.setOpacity(1, on ? this.s.heatmap_opacity : 0);
  }

  undo() {
    this.nv.drawUndo();
  }

  resetView() {
    this.nv.scene.pan2Dxyzmm = [0, 0, 0, 1];
    this.nv.drawScene();
    this.hooks.onView?.();
  }

  // ---------- slices ----------
  setSlice(z) {
    z = Math.max(0, Math.min(this.nSlices - 1, Math.round(z)));
    if (z === this.slice) return;
    this.nv.moveCrosshairInVox(0, 0, z - this.slice);
    this.hooks.onSlice?.(this.slice);
  }

  moveSlice(step) { this.setSlice(this.slice + step); }

  /**
   * Slices to mark on the slider: those whose uncertainty is at least slice_marks_fraction of the most
   * uncertain slice. A slice's uncertainty = the sum of the heatmap values that are drawn (>= heatmap_min).
   * Returns [{z, level}] with level = the slice's uncertainty / the most uncertain slice's (0-1), or []
   * if no heatmap is loaded (the without-heatmap condition).
   */
  uncertainSlices() {
    if (this.nv.volumes.length < 2) return [];
    const heat = this.nv.volumes[1], [nx, ny, nz] = this.dims;
    const slope = heat.hdr.scl_slope || 1, inter = heat.hdr.scl_inter || 0;
    const sums = new Float64Array(nz);
    for (let z = 0; z < nz; z++) {
      let sum = 0;
      for (let i = z * nx * ny, end = i + nx * ny; i < end; i++) {
        const v = heat.img[i] * slope + inter;
        if (v >= this.s.heatmap_min) sum += v;
      }
      sums[z] = sum;
    }
    const max = Math.max(...sums);
    if (max <= 0) return [];
    return [...sums.keys()].filter((z) => sums[z] >= this.s.slice_marks_fraction * max)
      .map((z) => ({ z, level: sums[z] / max }));
  }

  // ---------- screen <-> scan coordinates ----------
  // NiiVue keeps, for the slice on screen, its rectangle on the canvas (leftTopWidthHeight, device pixels)
  // and which mm coordinates that rectangle covers (leftTopMM, fovMM). This works outside the scan too.
  mmAt(clientX, clientY) {
    const s = this.nv.screenSlices[0], t = s.leftTopWidthHeight, dpr = this.nv.uiData.dpr;
    const r = this.canvas.getBoundingClientRect();
    const x = (clientX - r.left) * dpr, y = (clientY - r.top) * dpr;
    return [s.leftTopMM[0] + ((x - t[0]) / t[2]) * s.fovMM[0], s.leftTopMM[1] + (1 - (y - t[1]) / t[3]) * s.fovMM[1]];
  }

  /** Screen (CSS) pixels per mm at the current zoom. */
  cssPxPerMm() {
    const s = this.nv.screenSlices[0];
    return Math.abs(s.leftTopWidthHeight[2] / s.fovMM[0]) / this.nv.uiData.dpr;
  }

  /** The voxel [x, y, z] under a screen point on the current slice, or null outside the scan. */
  voxelAt(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect(), dpr = this.nv.uiData.dpr;
    const f = this.nv.canvasPos2frac([(clientX - r.left) * dpr, (clientY - r.top) * dpr]);
    if (f[0] < 0) return null;
    const v = this.nv.frac2vox(f);
    return [Math.round(v[0]), Math.round(v[1]), this.slice];
  }

  // ---------- pointer input ----------
  onPointerDown(e) {
    if (e.button !== 0 || !this.nv?.back) return;
    e.preventDefault();
    this.canvas.setPointerCapture(e.pointerId);   // keep receiving moves if the pointer leaves the canvas
    if (this.tool === "move" || !this.paintEnabled) {
      this.drag = { kind: "pan", x: e.clientX, y: e.clientY };
      this.canvas.classList.add("dragging");
      return;
    }
    this.drag = { kind: "paint", value: this.tool === "add" ? MASK : 0, last: null, points: 0, slice: this.slice };
    this.paintAlong([e]);
  }

  onPointerMove(e) {
    this.showBrush(e);
    if (!this.drag) return;
    if (this.drag.kind === "pan") {
      this.panBy(e.clientX - this.drag.x, e.clientY - this.drag.y);
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
    } else {
      // Browsers merge fast moves into one event; the merged points make the stroke follow the finger.
      const points = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
      this.paintAlong(points.length ? points : [e]);
    }
  }

  onPointerUp(e) {
    const d = this.drag;
    this.drag = null;
    this.canvas.classList.remove("dragging");
    if (this.canvas.hasPointerCapture?.(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    if (d?.kind === "paint" && d.points > 0) {
      this.nv.drawAddUndoBitmap();      // one undo step per stroke
      this.hooks.onStroke?.({ tool: this.tool, slice: d.slice });
    }
  }

  /**
   * Paint (or erase) along pointer positions, on the current slice: from the previous point of the stroke
   * to each new point, as a line of round brush stamps (one per voxel step, so fast strokes leave no gaps).
   * Then the changed slice is sent to the graphics card and the image is drawn once.
   */
  paintAlong(points) {
    if (!this.paintEnabled) return;
    let changed = false;
    for (const p of points) {
      const v = this.voxelAt(p.clientX, p.clientY);
      if (!v) { this.drag.last = null; continue; }      // outside the scan: the next point starts anew
      const a = this.drag.last || v;
      const steps = Math.max(Math.abs(v[0] - a[0]), Math.abs(v[1] - a[1]), 1);
      for (let i = 0; i <= steps; i++) {
        this.stamp(Math.round(a[0] + ((v[0] - a[0]) * i) / steps), Math.round(a[1] + ((v[1] - a[1]) * i) / steps),
                   v[2], this.drag.value);
      }
      this.drag.last = v;
      this.drag.points += 1;
      changed = true;
    }
    if (!changed) return;
    this.uploadSlice(this.drag.slice);
    this.nv.drawScene();
  }

  /**
   * Round brush: set every voxel whose center lies within penSize / 2 voxels of the center voxel.
   * (NiiVue's own brush is a square; a round one matches the circle that shows the brush on screen.)
   */
  stamp(cx, cy, z, value) {
    const [nx, ny] = this.dims, r = this.penSize / 2, ri = Math.floor(r), bmp = this.nv.drawBitmap;
    for (let dy = -ri; dy <= ri; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= ny) continue;
      for (let dx = -ri; dx <= ri; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= nx || dx * dx + dy * dy > r * r) continue;
        bmp[x + y * nx + z * nx * ny] = value;
      }
    }
  }

  /**
   * Send only the changed slice of the drawing to the graphics card, so painting shows at once.
   * (NiiVue's refreshDrawing uploads the whole volume, which is slow for large scans.)
   */
  uploadSlice(z) {
    const gl = this.nv.gl, [nx, ny] = this.dims;
    gl.activeTexture(gl.TEXTURE0 + DRAW_TEXTURE_UNIT);
    gl.bindTexture(gl.TEXTURE_3D, this.nv.drawTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, z, nx, ny, 1, gl.RED, gl.UNSIGNED_BYTE,
                     this.nv.drawBitmap.subarray(z * nx * ny, (z + 1) * nx * ny));
  }

  /** Draw the brush circle at the pointer, with the brush's real size on screen. */
  showBrush(e) {
    const visible = this.paintEnabled && this.tool !== "move" && this.nv?.back && this.voxelAt(e.clientX, e.clientY);
    this.brush.hidden = !visible;
    if (!visible) return;
    const voxelMm = this.nv.back.pixDimsRAS[1];
    const size = Math.max(6, this.penSize * voxelMm * this.cssPxPerMm());
    const r = this.canvas.getBoundingClientRect();
    Object.assign(this.brush.style, { width: `${size}px`, height: `${size}px`,
                                      left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px` });
    this.brush.classList.toggle("erase", this.tool === "erase");
  }

  // ---------- pan and zoom ----------
  // NiiVue shows the point p (mm) at display position p * zoom + pan (pan = scene.pan2Dxyzmm[0..2]).
  // Moving the image by a screen distance: the point under the pointer should follow the pointer, so
  // pan changes by (mm under the new position - mm under the old position) * zoom.
  panBy(dxCss, dyCss) {
    const s = this.nv.screenSlices[0], t = s.leftTopWidthHeight, dpr = this.nv.uiData.dpr, p = this.nv.scene.pan2Dxyzmm;
    p[0] += ((dxCss * dpr) / t[2]) * s.fovMM[0] * p[3];
    p[1] -= ((dyCss * dpr) / t[3]) * s.fovMM[1] * p[3];
    this.nv.drawScene();
    this.hooks.onView?.();
  }

  /** Zoom by a factor, keeping the point under the pointer in place: pan += c * (old zoom - new zoom). */
  zoomAt(factor, clientX, clientY) {
    const p = this.nv.scene.pan2Dxyzmm, c = this.mmAt(clientX, clientY);
    const z1 = Math.min(this.s.zoom_max, Math.max(this.s.zoom_min, p[3] * factor));
    p[0] += c[0] * (p[3] - z1);
    p[1] += c[1] * (p[3] - z1);
    p[3] = z1;
    this.nv.drawScene();
    this.hooks.onView?.();
  }

  // Trackpad: a pinch arrives as a wheel event with ctrlKey set; a two-finger swipe as a plain wheel event.
  onWheel(e) {
    e.preventDefault();
    e.stopPropagation();
    if (!this.nv?.back || this.drag) return;
    const scale = e.deltaMode === 1 ? 16 : 1;           // lines -> pixels (some mouse wheels)
    const dx = e.deltaX * scale, dy = e.deltaY * scale;
    const action = e.ctrlKey ? "zoom" : this.s.two_finger_scroll;
    if (action === "zoom") this.zoomAt(Math.exp(-dy * this.s.pinch_zoom_speed), e.clientX, e.clientY);
    else if (action === "pan") this.panBy(-dx, -dy);
    else if (action === "slice") {
      this.scrollRest += dy;
      const steps = Math.trunc(this.scrollRest / this.s.scroll_px_per_slice);
      this.scrollRest -= steps * this.s.scroll_px_per_slice;
      if (steps) this.moveSlice(-steps);
    }
    this.showBrush(e);
  }

  // ---------- ground truth comparison ----------
  /**
   * Show the participant's mask against the ground truth in the drawing layer: red = marked and kidney,
   * yellow = kidney that is not marked (missed), blue = marked but not kidney (extra). Painting is off
   * meanwhile; hideComparison() puts the participant's own mask back unchanged.
   */
  async showComparison() {
    if (!this.truthUrl) throw new Error("no ground truth for this scan");
    if (!this.truth) {
      const img = await niivue.NVImage.loadFromUrl({ url: this.truthUrl });
      // In RAS orientation (permRAS 1,2,3) the file's dimensions are the viewer's dimensions.
      if (Array.from(img.permRAS).join() !== "1,2,3" || Array.from(img.hdr.dims.slice(1, 4)).join() !== this.dims.join()) {
        throw new Error("the ground truth does not match the scan");
      }
      this.truth = img.img;
    }
    if (!this.userDrawing) this.userDrawing = this.nv.drawBitmap;
    const user = this.userDrawing, truth = this.truth, cmp = new Uint8Array(user.length);
    for (let i = 0; i < cmp.length; i++) {
      const u = user[i] > 0, g = truth[i] > 0;
      cmp[i] = u && g ? MASK : g ? MISSED : u ? EXTRA : 0;
    }
    this.nv.drawBitmap = cmp;
    this.nv.refreshDrawing(true);
  }

  hideComparison() {
    if (!this.userDrawing) return;
    this.nv.drawBitmap = this.userDrawing;
    this.userDrawing = null;
    this.nv.refreshDrawing(true);
  }

  /** The participant's mask as NIfTI bytes (with the CT's header), for saving. */
  async maskBytes() {
    this.hideComparison();
    return this.nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  }
}
