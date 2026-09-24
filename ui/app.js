// Review interface (G2, pipeline step 5): the participant corrects the model's kidney mask with a brush,
// with or without the uncertainty heatmap. Talks to the local server in src/segreview/ui_server.py.
//
// Flow in study mode: participant id -> practice scan(s) -> the 6 study scans in the balanced order
// the server returns -> thank-you screen. Each scan: intro screen (condition) -> viewer with timer ->
// "Færdig" or time limit -> the corrected mask and a log are sent to the server -> next scan.
// Scans that already have a saved log are skipped, so a session can be resumed after a crash.
//
// Timing: the clock starts when the scan is shown and ready to edit, and stops at "Færdig" or when the
// time limit is reached (then the mask is saved as it is). The log records both the reason and the time.

const $ = (id) => document.getElementById(id);
const state = {
  settings: null,
  participant: null,
  items: [],          // scans still to do: {case_id, condition, position, mode}
  current: null,
  nv: null,
  t0: 0,              // performance.now() when the scan became editable
  startedIso: null,
  timerId: null,
  finishing: false,
  events: [],
  heatmapLoaded: false,
  heatmapOn: false,
  heatmapSince: 0,    // when the heatmap was last switched on (ms)
  heatmapMs: 0,       // total time the heatmap was visible (ms)
  undoCount: 0,
  drawActions: 0,
};

// ---------- small helpers ----------
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function show(screenId) {
  for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== screenId;
}

function seconds() {
  return (performance.now() - state.t0) / 1000;
}

function logEvent(type, extra = {}) {
  state.events.push({ t: Math.round(seconds() * 100) / 100, type, ...extra });
}

function fileUrl(item, name) {
  // The participant id is part of the path so the server can refuse the heatmap in the "without" condition.
  const who = item.mode === "queue" ? "demo" : state.participant;
  const set = item.mode === "queue" ? "queue" : "scans";
  return `/files/${who}/${set}/${item.case_id}/${name}.nii.gz`;
}

// ---------- start screen ----------
async function init() {
  state.settings = await api("/api/settings");
  $("intro-limit").textContent = state.settings.time_limit_min;
  $("start-study").onclick = startStudy;
  $("participant").onkeydown = (e) => { if (e.key === "Enter") startStudy(); };
  $("start-queue").onclick = openQueue;
  $("queue-back").onclick = () => show("start");
  $("end-home").onclick = () => show("start");
  $("intro-go").onclick = () => openScan(state.current);
  $("done").onclick = () => finish("done");
  buildToolbar();
  const selftest = new URLSearchParams(location.search).get("selftest");
  if (selftest === "view") openScan({ case_id: (await api("/api/queue"))[0].case_id, condition: "with_heatmap",
                                      position: 1, mode: "queue" });
  else if (selftest) selfTest();
}

async function startStudy() {
  const pid = $("participant").value.trim().toUpperCase();
  $("start-error").hidden = true;
  try {
    const plan = await api(`/api/study/${encodeURIComponent(pid)}`);
    state.participant = pid;
    state.items = [
      ...plan.practice.filter((s) => !s.done).map((s) => ({ ...s, mode: "practice" })),
      ...plan.scans.filter((s) => !s.done).map((s) => ({ ...s, mode: "study", total: plan.scans.length })),
    ];
    nextItem();
  } catch (err) {
    $("start-error").textContent = `Kunne ikke starte: ${err.message}`;
    $("start-error").hidden = false;
  }
}

function nextItem() {
  state.current = state.items.shift() || null;
  if (!state.current) return show("end");
  const it = state.current;
  const withHeatmap = it.condition === "with_heatmap";
  $("intro-title").textContent = it.mode === "practice" ? "Øvelse" : `Scanning ${it.position} af ${it.total}`;
  $("intro-text").textContent = it.mode === "practice"
    ? "Prøv værktøjerne af. Denne scanning tæller ikke med."
    : withHeatmap
      ? "Denne gang kan du se, hvor AI'en er usikker (farvet lag). Brug det som hjælp til at finde fejl."
      : "Denne gang er der intet usikkerhedslag. Find fejlene selv.";
  show("intro");
}

// ---------- queue (demo) ----------
async function openQueue() {
  state.participant = "demo";
  const list = await api("/api/queue");
  const ol = $("queue-list");
  ol.innerHTML = "";
  for (const q of list) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.textContent = `${q.case_id}   (usikkerhed ${q.score.toFixed(3)})`;
    b.onclick = () => openScan({ case_id: q.case_id, condition: "with_heatmap", position: q.rank, mode: "queue" });
    li.appendChild(b);
    ol.appendChild(li);
  }
  show("queue");
}

// ---------- toolbar ----------
function buildToolbar() {
  const v = state.settings.viewer;
  const sizes = $("pen-sizes");
  for (const size of v.pen_sizes) {
    const b = document.createElement("button");
    b.textContent = size;
    b.className = "tool size";
    b.setAttribute("aria-label", `Penselstørrelse ${size}`);
    b.onclick = () => setPenSize(size);
    sizes.appendChild(b);
  }
  const presets = $("window-presets");
  v.window_presets.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = p.name;
    b.className = "tool window";
    b.onclick = () => setWindow(i);
    presets.appendChild(b);
  });
  $("tool-add").onclick = () => setTool("add");
  $("tool-erase").onclick = () => setTool("erase");
  $("undo").onclick = undo;
  $("slice-up").onclick = () => moveSlice(1);
  $("slice-down").onclick = () => moveSlice(-1);
  $("heatmap-toggle").onclick = () => setHeatmap(!state.heatmapOn);
  document.addEventListener("keydown", (e) => {
    if ($("viewer").hidden) return;
    if (e.key === "PageUp" || e.key === "ArrowUp") { moveSlice(1); e.preventDefault(); }
    if (e.key === "PageDown" || e.key === "ArrowDown") { moveSlice(-1); e.preventDefault(); }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undo(); e.preventDefault(); }
  });
}

function setTool(tool) {
  // Pen value 1 paints kidney, 0 erases (NiiVue's drawing layer).
  state.nv.setPenValue(tool === "add" ? 1 : 0, false);
  for (const [id, t] of [["tool-add", "add"], ["tool-erase", "erase"]]) {
    $(id).classList.toggle("active", t === tool);
    $(id).setAttribute("aria-pressed", String(t === tool));
  }
  logEvent("tool", { tool });
}

function setPenSize(size) {
  state.nv.opts.penSize = size;
  for (const b of document.querySelectorAll("#pen-sizes button")) b.classList.toggle("active", Number(b.textContent) === size);
  logEvent("pen_size", { size });
}

function setWindow(i) {
  const p = state.settings.viewer.window_presets[i];
  const ct = state.nv.volumes[0];
  ct.cal_min = p.level - p.width / 2;
  ct.cal_max = p.level + p.width / 2;
  state.nv.updateGLVolume();
  for (const [j, b] of [...document.querySelectorAll("#window-presets button")].entries()) b.classList.toggle("active", j === i);
  logEvent("window", { preset: p.name });
}

function undo() {
  state.nv.drawUndo();
  state.undoCount += 1;
  logEvent("undo");
}

function moveSlice(step) {
  state.nv.moveCrosshairInVox(0, 0, step);
}

function setHeatmap(on) {
  if (!state.heatmapLoaded) return;
  const now = performance.now();
  if (state.heatmapOn && !on) state.heatmapMs += now - state.heatmapSince;
  if (!state.heatmapOn && on) state.heatmapSince = now;
  state.heatmapOn = on;
  state.nv.setOpacity(1, on ? state.settings.viewer.heatmap_opacity : 0);
  $("heatmap-toggle").textContent = on ? "Usikkerhed: til" : "Usikkerhed: fra";
  $("heatmap-toggle").setAttribute("aria-pressed", String(on));
  logEvent("heatmap", { on });
}

// ---------- viewer ----------
async function ensureViewer() {
  if (state.nv) return state.nv;
  const nv = new niivue.Niivue({
    backColor: [0.07, 0.07, 0.07, 1],
    isColorbar: false,
    show3Dcrosshair: false,
    crosshairWidth: 0,
    isRadiologicalConvention: true,      // patient's right on the left of the screen, as radiologists view CT
    dragMode: niivue.DRAG_MODE.pan,      // right-drag moves the image instead of changing the contrast
  });
  await nv.attachTo("gl");
  nv.setSliceType(nv.sliceTypeAxial);
  nv.onDrawingChanged = () => { state.drawActions += 1; };
  state.nv = nv;
  return nv;
}

async function openScan(item) {
  state.current = item;
  show("viewer");
  $("status").textContent = "Indlæser scanning ...";
  $("done").disabled = true;
  const v = state.settings.viewer;
  const nv = await ensureViewer();
  const withHeatmap = item.condition === "with_heatmap";

  const volumes = [{ url: fileUrl(item, "ct"), colormap: "gray" }];
  if (withHeatmap) {
    volumes.push({ url: fileUrl(item, "heatmap"), colormap: v.heatmap_colormap, cal_min: v.heatmap_min, cal_max: 1,
                   opacity: v.heatmap_opacity });
  }
  nv.closeDrawing();
  await nv.loadVolumes(volumes);
  await nv.loadDrawingFromUrl(fileUrl(item, "mask"), true);
  nv.setDrawOpacity(v.mask_opacity);
  nv.setDrawingEnabled(true);
  nv.drawClearAllUndoBitmaps?.();

  // Reset per-scan state and start the clock only now that everything is ready.
  Object.assign(state, { events: [], heatmapLoaded: withHeatmap, heatmapOn: false, heatmapMs: 0, undoCount: 0,
                         drawActions: 0, finishing: false, t0: performance.now(), startedIso: new Date().toISOString() });
  setTool("add");
  setPenSize(v.default_pen_size);
  setWindow(0);
  $("heatmap-toggle").hidden = !withHeatmap;
  if (withHeatmap) setHeatmap(true);
  $("scan-label").textContent = item.mode === "queue" ? `${item.case_id} (nr. ${item.position} i køen)`
    : item.mode === "practice" ? "Øvelse" : `Scanning ${item.position} af ${item.total}`;
  $("status").textContent = "Mal med venstre museknap. Musehjul: skift snit. Højreklik og træk: flyt billedet.";
  $("done").disabled = false;
  logEvent("start");
  startTimer();
}

function startTimer() {
  clearInterval(state.timerId);
  const limit = state.settings.time_limit_min * 60;
  const tick = () => {
    const left = Math.max(0, limit - seconds());
    const m = Math.floor(left / 60), s = Math.floor(left % 60);
    $("timer").textContent = `${m}:${String(s).padStart(2, "0")}`;
    $("timer").classList.toggle("low", left < 60);
    if (left <= 0 && state.current.mode !== "queue") finish("timeout");
  };
  tick();
  state.timerId = setInterval(tick, 250);
}

async function finish(reason) {
  if (state.finishing) return;
  state.finishing = true;
  clearInterval(state.timerId);
  const duration = seconds();
  if (state.heatmapOn) state.heatmapMs += performance.now() - state.heatmapSince;
  logEvent("end", { reason });
  state.nv.setDrawingEnabled(false);
  $("done").disabled = true;
  $("status").textContent = "Gemmer ...";
  const it = state.current;
  try {
    // With an empty file name NiiVue returns the drawing as NIfTI bytes (with the CT's header) instead of downloading.
    const bytes = await state.nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
    await api(`/api/save/${it.mode}/${state.participant}/${it.case_id}`, { method: "POST", body: bytes });
    const log = {
      condition: it.condition, position: it.position, reason,
      started_iso: state.startedIso, ended_iso: new Date().toISOString(),
      duration_s: Math.round(duration * 100) / 100, time_limit_s: state.settings.time_limit_min * 60,
      heatmap_available: state.heatmapLoaded, heatmap_visible_s: Math.round(state.heatmapMs / 10) / 100,
      undo_count: state.undoCount, draw_actions: state.drawActions, events: state.events,
      user_agent: navigator.userAgent, screen: [window.innerWidth, window.innerHeight],
    };
    await api(`/api/log/${it.mode}/${state.participant}/${it.case_id}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(log) });
    $("status").textContent = reason === "timeout" ? "Tiden er gået. Scanningen er gemt." : "Gemt.";
  } catch (err) {
    // Keep the scan open so the experimenter can retry; nothing is lost from the drawing.
    $("status").textContent = `FEJL ved gemning: ${err.message}. Tilkald forsøgslederen.`;
    state.finishing = false;
    $("done").disabled = false;
    state.nv.setDrawingEnabled(true);
    return;
  }
  if (it.mode === "queue") return openQueue();
  nextItem();
}

// ---------- self-test (used by tests/test_ui_browser.py in a headless browser) ----------
// ?selftest=1 runs the paint-and-save test below; ?selftest=view only opens the first queue scan (for a screenshot).
// Opens the first queue scan, paints a 10 x 10 square in the middle slice directly into the drawing
// bitmap, and saves through the normal path. The test then checks that the saved mask is the original
// mask plus exactly that square, which proves that loading, orientation and saving line up.
async function selfTest() {
  const out = document.createElement("pre");
  out.id = "selftest";
  document.body.appendChild(out);
  try {
    state.participant = "demo";
    const list = await api("/api/queue");
    const item = { case_id: list[0].case_id, condition: "with_heatmap", position: 1, mode: "queue" };
    await openScan(item);
    const nv = state.nv;
    const [, nx, ny, nz] = nv.back.dims;
    const z = Math.floor(nz / 2), x0 = Math.floor(nx / 2) - 5, y0 = Math.floor(ny / 2) - 5;
    for (let y = y0; y < y0 + 10; y++) for (let x = x0; x < x0 + 10; x++) nv.drawBitmap[x + y * nx + z * nx * ny] = 1;
    nv.refreshDrawing(true);
    await finish("selftest");
    out.textContent = JSON.stringify({ ok: true, case_id: item.case_id, square: { x0, y0, z, size: 10 }, dims: [nx, ny, nz] });
  } catch (err) {
    out.textContent = JSON.stringify({ ok: false, error: String(err) });
  }
  document.title = "SELFTEST DONE";
}

init();
