// Review interface (G2, pipeline step 5): the participant corrects the model's kidney mask with a brush,
// with or without the uncertainty heatmap. Talks to the local server in src/segreview/ui_server.py; the
// slice viewer itself (layers, brush, pan/zoom, ground-truth comparison) is in viewer.js.
//
// Study mode: participant id -> guide (what kidneys and tumors look like, how to use the trackpad) ->
// practice scan -> the correct answer for the practice scan -> the 6 study scans in the balanced order the
// server returns -> NASA-TLX questionnaire (after each condition block, or once at the end; study.tlx in the
// config) -> thank-you screen. Each scan: intro screen (condition) -> viewer with timer -> "Done" or
// the time limit -> the corrected mask and a log are sent to the server. When the time is up the scan is
// locked and a message says so. Scans that already have a saved log are skipped, so a session can be resumed.
//
// Demo mode (ranked queue): no time limit, and the ground truth can be shown.
//
// Timing: the clock starts when the scan is shown and ready to edit, and stops at "Done" or when the time
// limit is reached (then the mask is saved as it is). The log records both the reason and the time.

const $ = (id) => document.getElementById(id);

// NASA-TLX scales (official wording). The keys must match TLX_SCALES in src/segreview/study.py.
const TLX = [
  ["mental", "Mental demand", "How mentally demanding was the task?", "Very low", "Very high"],
  ["physical", "Physical demand", "How physically demanding was the task?", "Very low", "Very high"],
  ["temporal", "Temporal demand", "How hurried or rushed was the pace of the task?", "Very low", "Very high"],
  ["performance", "Performance", "How successful were you in accomplishing what you were asked to do?", "Perfect", "Failure"],
  ["effort", "Effort", "How hard did you have to work to accomplish your level of performance?", "Very low", "Very high"],
  ["frustration", "Frustration", "How insecure, discouraged, irritated, stressed, and annoyed were you?", "Very low", "Very high"],
];
const state = {
  settings: null,
  participant: null,
  items: [],          // steps still to do: scans {case_id, condition, position, mode} and questionnaires {mode: "tlx", which}
  current: null,
  viewer: null,       // ReviewViewer (viewer.js)
  t0: 0,              // performance.now() when the scan became editable
  startedIso: null,
  timerId: null,
  finishing: false,
  events: [],
  heatmapLoaded: false,
  heatmapOn: false,
  heatmapSince: 0,    // when the heatmap was last switched on (ms)
  heatmapMs: 0,       // total time the heatmap was visible (ms)
  marks: [],          // high-uncertainty slices, marked on the slider when the heatmap is on
  comparing: false,   // the ground truth comparison is shown
  undoCount: 0,
  strokes: 0,
  slicesSeen: new Set(),
  tool: "add",
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

function clock(s) {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
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

function viewerData(key, value) {
  $("viewer").dataset[key] = value;
}

/** How to move the image, in words (depends on what a two-finger swipe does in the config). */
function moveHelp() {
  const swipe = { pan: "or swipe with two fingers", slice: "(a two-finger swipe changes the slice)",
                  zoom: "(a two-finger swipe zooms)" }[state.settings.viewer.two_finger_scroll];
  return `<b>Move the image:</b> choose <b>Move</b> and drag ${swipe}. <b>Zoom:</b> pinch with two fingers. ` +
         `<b>Reset view</b> fits the image again.`;
}

// ---------- start screen ----------
async function init() {
  state.settings = await api("/api/settings");
  const v = state.settings.viewer;
  for (const el of document.querySelectorAll(".limit")) el.textContent = state.settings.time_limit_min;
  for (const el of document.querySelectorAll(".move-help")) el.innerHTML = moveHelp();
  for (const [name, rgb] of Object.entries(v.compare_colors)) {
    document.documentElement.style.setProperty(`--${name}`, `rgb(${rgb.join(",")})`);
  }
  if (state.settings.guide_image) {
    $("guide-image").src = "/files/guide/example.png";
    $("guide-figure").hidden = false;
  }
  $("start-study").onclick = startStudy;
  $("participant").onkeydown = (e) => { if (e.key === "Enter") startStudy(); };
  $("start-queue").onclick = openQueue;
  $("queue-back").onclick = () => show("start");
  $("end-home").onclick = () => show("start");
  $("guide-go").onclick = nextItem;
  $("intro-go").onclick = () => openScan(state.current);
  $("done").onclick = () => finish("done");
  $("answer-continue").onclick = nextItem;
  $("tlx-submit").onclick = submitTlx;
  buildToolbar();
  buildTlx();
  buildSliceSlider();
}

async function startStudy() {
  const pid = $("participant").value.trim().toUpperCase();
  $("start-error").hidden = true;
  try {
    const plan = await api(`/api/study/${encodeURIComponent(pid)}`);
    state.participant = pid;
    // Each study scan (if not done yet), followed by the questionnaire(s) due after its position.
    const tlxAfter = {};
    for (const t of plan.tlx.filter((t) => !t.done)) (tlxAfter[t.after_position] ||= []).push({ mode: "tlx", which: t.which });
    state.items = [
      ...plan.practice.filter((s) => !s.done).map((s) => ({ ...s, mode: "practice" })),
      ...plan.scans.flatMap((s) => [...(s.done ? [] : [{ ...s, mode: "study", total: plan.scans.length }]),
                                    ...(tlxAfter[s.position] || [])]),
    ];
    // The guide comes before the practice scan (and is skipped when a session is resumed after it).
    if (state.items[0]?.mode === "practice") show("guide");
    else nextItem();
  } catch (err) {
    $("start-error").textContent = `Could not start: ${err.message}`;
    $("start-error").hidden = false;
  }
}

function nextItem() {
  state.current = state.items.shift() || null;
  if (!state.current) return show("end");
  const it = state.current;
  if (it.mode === "tlx") return showTlx(it.which);
  $("intro-title").textContent = it.mode === "practice" ? "Practice scan" : `Scan ${it.position} of ${it.total}`;
  $("intro-text").textContent = it.mode === "practice"
    ? "Try the tools. This scan does not count. The blue colors show where the AI is uncertain. " +
      "Afterwards you will see the correct answer."
    : it.condition === "with_heatmap"
      ? "This time you can see where the AI is uncertain: blue colors in the image, and marks next to the " +
        "slice slider. Use them to find mistakes."
      : "This time there is no uncertainty information. Find the mistakes yourself.";
  show("intro");
}

// ---------- NASA-TLX questionnaire ----------
function buildTlx() {
  for (const [key, title, question, low, high] of TLX) {
    const card = document.createElement("div");
    card.className = "tlx-scale";
    card.innerHTML = `<h2>${title}</h2><p>${question}</p>
      <input type="range" class="tlx-range unset" min="0" max="100" step="5" value="50" data-key="${key}"
             aria-label="${title}: ${question}">
      <div class="tlx-anchors"><span>${low}</span><span>${high}</span></div>`;
    const input = card.querySelector("input");
    // A scale counts as answered once the participant has clicked or moved it; until then it shows no position.
    const answered = () => { input.classList.remove("unset"); updateTlxButton(); };
    input.addEventListener("pointerdown", answered);
    input.addEventListener("input", answered);
    input.addEventListener("keydown", answered);
    $("tlx-scales").appendChild(card);
  }
}

function updateTlxButton() {
  const open = [...document.querySelectorAll(".tlx-range.unset")].length;
  $("tlx-submit").disabled = open > 0;
  $("tlx-missing").textContent = open ? ` ${open} of ${TLX.length} scales still to answer` : "";
}

function showTlx(which) {
  const about = {
    session: "the scans you just corrected (not the practice scan)",
    with_heatmap: "the last scans, where you could see where the AI was unsure (blue)",
    without_heatmap: "the last scans, where there was no uncertainty information",
  }[which];
  $("tlx-lead").textContent = `Think about ${about}. For each scale, click where it fits best. There are no right answers.`;
  for (const input of document.querySelectorAll(".tlx-range")) {
    input.value = 50;
    input.classList.add("unset");
  }
  $("tlx-error").hidden = true;
  state.tlxStarted = new Date().toISOString();
  updateTlxButton();
  show("tlx");
  window.scrollTo(0, 0);
}

async function submitTlx() {
  const scales = {};
  for (const input of document.querySelectorAll(".tlx-range")) scales[input.dataset.key] = input.valueAsNumber;
  $("tlx-submit").disabled = true;
  try {
    await api(`/api/tlx/${state.participant}/${state.current.which}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scales, started_iso: state.tlxStarted, submitted_iso: new Date().toISOString() }) });
  } catch (err) {
    $("tlx-error").textContent = `Could not save: ${err.message}. Please call the experimenter.`;
    $("tlx-error").hidden = false;
    $("tlx-submit").disabled = false;
    return;
  }
  nextItem();
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
    b.textContent = `${q.case_id}   (uncertainty ${q.score.toFixed(3)})`;
    b.onclick = () => openScan({ case_id: q.case_id, condition: "with_heatmap", position: q.rank, mode: "queue" });
    li.appendChild(b);
    ol.appendChild(li);
  }
  show("queue");
}

// ---------- toolbar ----------
function buildToolbar() {
  const v = state.settings.viewer;
  const biggest = Math.max(...v.pen_sizes);
  for (const size of v.pen_sizes) {
    const b = document.createElement("button");
    b.className = "tool size";
    b.dataset.size = size;
    b.title = `Brush ${size} voxels wide`;
    b.setAttribute("aria-label", `Brush size ${size}`);
    const dot = document.createElement("span");
    dot.className = "dot";
    const px = 4 + (14 * (size - 1)) / Math.max(1, biggest - 1);
    dot.style.width = dot.style.height = `${px}px`;
    b.appendChild(dot);
    b.onclick = () => setPenSize(size);
    $("pen-sizes").appendChild(b);
  }
  v.window_presets.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = p.name;
    b.className = "tool window";
    b.onclick = () => setWindow(i);
    $("window-presets").appendChild(b);
  });
  $("tool-add").onclick = () => setTool("add");
  $("tool-erase").onclick = () => setTool("erase");
  $("tool-move").onclick = () => setTool("move");
  $("undo").onclick = undo;
  $("reset-view").onclick = () => { state.viewer.resetView(); logEvent("reset_view"); };
  $("heatmap-toggle").onclick = () => setHeatmap(!state.heatmapOn);
  $("truth-toggle").onclick = () => setComparison(!state.comparing);
  $("slice-up").onclick = () => state.viewer.moveSlice(1);
  $("slice-down").onclick = () => state.viewer.moveSlice(-1);
  $("lock-button").onclick = () => state.lockAction?.();

  // A clicked button keeps the keyboard focus; then Space or Enter would press it again (e.g. "Done").
  document.addEventListener("click", (e) => {
    if (e.target.closest("#viewer button")) e.target.closest("button").blur();
  });
  // Keys in the viewer. Registered on window in the capture phase, so they work wherever the focus is.
  window.addEventListener("keydown", (e) => {
    if ($("viewer").hidden || $("viewer").dataset.mode === "loading" || !$("lock").hidden) return;
    if (e.key === "ArrowUp" || e.key === "PageUp") { state.viewer.moveSlice(1); e.preventDefault(); }
    else if (e.key === "ArrowDown" || e.key === "PageDown") { state.viewer.moveSlice(-1); e.preventDefault(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undo(); e.preventDefault(); }
  }, true);
}

function setTool(tool) {
  state.tool = tool;
  state.viewer.setTool(tool);
  for (const t of ["add", "erase", "move"]) {
    $(`tool-${t}`).classList.toggle("active", t === tool);
    $(`tool-${t}`).setAttribute("aria-pressed", String(t === tool));
  }
  updateHelp();
  logEvent("tool", { tool });
}

function setPenSize(size) {
  state.viewer.setPenSize(size);
  for (const b of document.querySelectorAll("#pen-sizes button")) b.classList.toggle("active", Number(b.dataset.size) === size);
  // Painting with the Move tool selected would be confusing: choosing a brush size goes back to painting.
  if (state.tool === "move") setTool("add");
  logEvent("pen_size", { size });
}

function setWindow(i) {
  const p = state.settings.viewer.window_presets[i];
  state.viewer.setWindow(p);
  for (const [j, b] of [...document.querySelectorAll("#window-presets button")].entries()) b.classList.toggle("active", j === i);
  logEvent("window", { preset: p.name });
}

function undo() {
  if (!state.viewer.paintEnabled) return;
  state.viewer.undo();
  state.undoCount += 1;
  logEvent("undo");
}

function setHeatmap(on) {
  if (!state.heatmapLoaded) return;
  const now = performance.now();
  if (state.heatmapOn && !on) state.heatmapMs += now - state.heatmapSince;
  if (!state.heatmapOn && on) state.heatmapSince = now;
  state.heatmapOn = on;
  state.viewer.setHeatmapVisible(on);
  drawSliceMarks();
  $("key-blue").hidden = !on;          // blue is only explained while it is shown
  $("heatmap-toggle").textContent = on ? "Uncertainty: on" : "Uncertainty: off";
  $("heatmap-toggle").setAttribute("aria-pressed", String(on));
  logEvent("heatmap", { on });
}

/** Demo queue: show or hide the ground truth comparison. Painting is paused while it is shown. */
async function setComparison(on) {
  const viewer = state.viewer;
  if (on) {
    state.heatmapBeforeCompare = state.heatmapOn;
    setHeatmap(false);                     // blue "extra" and the blue heatmap would mix up
    viewer.paintEnabled = false;
    try {
      await viewer.showComparison();
    } catch (err) {
      viewer.paintEnabled = true;
      if (state.heatmapBeforeCompare) setHeatmap(true);
      $("help").textContent = `Could not show the ground truth (${err.message}).`;
      return;
    }
  } else {
    viewer.hideComparison();
    viewer.paintEnabled = true;
    if (state.heatmapBeforeCompare) setHeatmap(true);
  }
  state.comparing = on;
  viewerData("compare", on ? "yes" : "no");
  $("legend").hidden = !on;
  $("truth-toggle").textContent = on ? "Hide ground truth" : "Show ground truth";
  $("truth-toggle").setAttribute("aria-pressed", String(on));
  updateHelp();
  logEvent("ground_truth", { on });
}

function updateHelp() {
  const mode = $("viewer").dataset.mode;
  let text;
  if (mode === "answer" || state.comparing) {
    text = "Red: marked and kidney. Yellow: kidney that is not marked. Blue: marked, but not kidney. " +
           "Look through the slices with ↑ ↓ or the slider. Drag to move the image, pinch to zoom.";
  } else if (state.tool === "move") {
    text = "Drag to move the image. Pinch with two fingers to zoom. Reset view fits the image again.";
  } else {
    const swipe = state.settings.viewer.two_finger_scroll === "pan" ? " or two-finger swipe" : "";
    text = "Paint: press the trackpad down and move your finger while you keep it pressed. " +
           `Slices: ↑ ↓ keys, ▲ ▼ or the slider. Move the image: Move tool${swipe}. Zoom: pinch.`;
  }
  $("help").textContent = text;
}

// ---------- slice slider ----------
// A vertical slider: the last slice (towards the head) at the top, slice 1 at the bottom. Each slice owns an
// equal band of the track; the thumb sits on the current slice's band. In the with-heatmap condition, the
// most uncertain slices get a bar in their band; the longer the bar, the more uncertain the slice.
function buildSliceSlider() {
  const track = $("slice-track");
  const fromPointer = (e) => {
    const r = track.getBoundingClientRect(), n = state.viewer.nSlices;
    state.viewer.setSlice(Math.floor((1 - (e.clientY - r.top) / r.height) * n));
  };
  track.addEventListener("pointerdown", (e) => {
    if (!state.viewer?.nv?.back || e.button !== 0) return;
    track.setPointerCapture(e.pointerId);
    track.dragging = true;
    fromPointer(e);
  });
  track.addEventListener("pointermove", (e) => { if (track.dragging) fromPointer(e); });
  const stop = () => { track.dragging = false; };
  track.addEventListener("pointerup", stop);
  track.addEventListener("pointercancel", stop);
  new ResizeObserver(() => { if (state.viewer?.nv?.back) { drawSliceMarks(); updateSlice(state.viewer.slice); } }).observe(track);
}

function drawSliceMarks() {
  const canvas = $("slice-marks"), track = $("slice-track"), dpr = window.devicePixelRatio || 1;
  const w = track.clientWidth, h = track.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.heatmapOn || !state.viewer?.nv?.back) return;
  const n = state.viewer.nSlices, band = h / n;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--mark").trim();
  for (const { z, level } of state.marks) ctx.fillRect(3, (n - 1 - z) * band, (w - 6) * level, Math.max(2, band));
}

function updateSlice(z) {
  const track = $("slice-track"), thumb = $("slice-thumb"), n = state.viewer.nSlices;
  const h = track.clientHeight, band = h / n, size = Math.max(band, 8);
  thumb.style.height = `${size}px`;
  thumb.style.top = `${(n - 0.5 - z) * band - size / 2}px`;
  $("slice-number").textContent = `Slice ${z + 1} / ${n}`;
  track.setAttribute("aria-valuemin", "1");
  track.setAttribute("aria-valuemax", String(n));
  track.setAttribute("aria-valuenow", String(z + 1));
  state.slicesSeen.add(z);
}

// ---------- viewer ----------
async function ensureViewer() {
  if (state.viewer) return state.viewer;
  const viewer = new ReviewViewer($("gl"), $("brush"), state.settings.viewer, {
    onSlice: (z) => updateSlice(z),
    onStroke: (info) => { state.strokes += 1; logEvent("stroke", info); },
  });
  await viewer.init();
  state.viewer = viewer;
  return viewer;
}

async function openScan(item) {
  state.current = item;
  const withHeatmap = item.condition === "with_heatmap";
  viewerData("mode", "loading");
  viewerData("kind", item.mode);
  viewerData("heatmap", withHeatmap ? "yes" : "no");
  viewerData("compare", "no");
  $("loading").textContent = "Loading scan …";
  $("legend").hidden = true;
  hideLock();
  show("viewer");
  const viewer = await ensureViewer();
  try {
    await viewer.load({ ct: fileUrl(item, "ct"), mask: fileUrl(item, "mask"),
                        heatmap: withHeatmap ? fileUrl(item, "heatmap") : null,
                        truth: item.mode === "study" ? null : fileUrl(item, "truth") });
  } catch (err) {
    $("loading").textContent = `Could not load the scan (${err.message}). Please call the experimenter.`;
    return;
  }

  // Reset per-scan state and start the clock only now that everything is ready.
  Object.assign(state, { events: [], heatmapLoaded: withHeatmap, heatmapOn: false, heatmapMs: 0, undoCount: 0,
                         strokes: 0, comparing: false, finishing: false, slicesSeen: new Set(),
                         marks: viewer.uncertainSlices(),
                         t0: performance.now(), startedIso: new Date().toISOString() });
  viewer.paintEnabled = true;
  setTool("add");
  setPenSize(state.settings.viewer.default_pen_size);
  setWindow(0);
  $("truth-toggle").textContent = "Show ground truth";
  $("scan-label").textContent = item.mode === "queue" ? `${item.case_id} (no. ${item.position} in the queue)`
    : item.mode === "practice" ? "Practice scan" : `Scan ${item.position} of ${item.total}`;
  $("key-blue").hidden = true;          // shown by setHeatmap(true) below, if this scan has the heatmap
  viewerData("mode", "edit");
  updateSlice(viewer.slice);
  if (withHeatmap) setHeatmap(true);
  drawSliceMarks();
  updateHelp();
  logEvent("start");
  startTimer();
}

function startTimer() {
  clearInterval(state.timerId);
  const timed = state.current.mode !== "queue";      // the demo queue has no time limit
  const limit = state.settings.time_limit_min * 60;
  const tick = () => {
    if (!timed) {
      $("timer").textContent = `Time ${clock(seconds())}`;
      return;
    }
    const left = Math.max(0, limit - seconds());
    $("timer").textContent = `Time left ${clock(Math.ceil(left))}`;
    $("timer").classList.toggle("low", left < 60);
    if (left <= 0) finish("timeout");
  };
  $("timer").classList.remove("low");
  tick();
  state.timerId = setInterval(tick, 250);
}

// ---------- finishing a scan ----------
async function finish(reason) {
  if (state.finishing) return;
  state.finishing = true;
  clearInterval(state.timerId);
  const duration = seconds();
  const viewer = state.viewer;
  viewer.paintEnabled = false;          // locked: no more painting from here on
  viewer.drag = null;                   // a stroke in progress ends here (what it painted so far is kept)
  if (state.comparing) await setComparison(false);
  viewer.paintEnabled = false;
  if (state.heatmapOn) state.heatmapMs += performance.now() - state.heatmapSince;
  logEvent("end", { reason });
  if (reason === "timeout") showLock("Time is up", "This scan is now locked. Saving your changes …", null);
  else $("help").textContent = "Saving …";
  await save(reason, duration);
}

/** Send the mask and the log to the server. On failure the message offers to try again. */
async function save(reason, duration) {
  const it = state.current;
  try {
    // With an empty file name NiiVue returns the drawing as NIfTI bytes (with the CT's header) instead of downloading.
    const bytes = await state.viewer.maskBytes();
    await api(`/api/save/${it.mode}/${state.participant}/${it.case_id}`, { method: "POST", body: bytes });
    const log = {
      condition: it.condition, position: it.position, reason,
      started_iso: state.startedIso, ended_iso: new Date().toISOString(),
      duration_s: Math.round(duration * 100) / 100,
      time_limit_s: it.mode === "queue" ? null : state.settings.time_limit_min * 60,
      heatmap_available: state.heatmapLoaded, heatmap_visible_s: Math.round(state.heatmapMs / 10) / 100,
      undo_count: state.undoCount, strokes: state.strokes,
      slices_visited: state.slicesSeen.size, n_slices: state.viewer.nSlices, events: state.events,
      user_agent: navigator.userAgent, screen: [window.innerWidth, window.innerHeight],
      device_pixel_ratio: window.devicePixelRatio,
    };
    await api(`/api/log/${it.mode}/${state.participant}/${it.case_id}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(log) });
  } catch (err) {
    // The drawing stays in the viewer, so nothing is lost; the experimenter can try again.
    showLock("Saving failed", `Could not save: ${err.message}. Please call the experimenter.`,
             ["Try again", () => { hideLock(); save(reason, duration); }]);
    return;
  }
  if (reason === "timeout") {
    showLock("Time is up", "This scan is now locked. Your changes have been saved.", ["Continue", afterScan]);
  } else {
    afterScan();
  }
}

function afterScan() {
  hideLock();
  const it = state.current;
  if (it.mode === "queue") return openQueue();
  if (it.mode === "practice") return showAnswer();
  nextItem();
}

/** After the practice scan: show the participant's correction against the correct answer. */
async function showAnswer() {
  const viewer = state.viewer;
  viewer.setHeatmapVisible(false);
  state.heatmapOn = false;
  drawSliceMarks();
  try {
    await viewer.showComparison();
  } catch (err) {
    $("help").textContent = `Could not show the answer (${err.message}).`;
  }
  viewer.setTool("move");               // dragging moves the image; painting is off
  $("scan-label").textContent = "Practice scan: the correct answer";
  $("answer-continue").textContent = state.items.length ? "Continue to the study" : "Finish";
  $("legend").hidden = false;
  viewerData("mode", "answer");
  updateHelp();
}

/** Message over the image. button: [label, action] or null (no button, e.g. while saving). */
function showLock(title, text, button) {
  $("lock-title").textContent = title;
  $("lock-text").textContent = text;
  $("lock-button").hidden = !button;
  if (button) {
    $("lock-button").textContent = button[0];
    state.lockAction = button[1];
  }
  $("lock").hidden = false;
  $("brush").hidden = true;
  for (const el of document.querySelectorAll("#viewer .toolbar, #viewer .slice-nav")) el.inert = true;   // locked
  if (button) $("lock-button").focus();
}

function hideLock() {
  $("lock").hidden = true;
  for (const el of document.querySelectorAll("#viewer .toolbar, #viewer .slice-nav")) el.inert = false;
  state.lockAction = null;
}

init();
