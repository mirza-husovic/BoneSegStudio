/* BoneSeg Studio frontend.
 *
 * One Canvas 2D view for everything: overlay / original / mask / skeleton,
 * with pan+zoom, brush/eraser painting, component select+delete and a
 * dirty-rect undo/redo history. No WebGL, no frameworks — the previous
 * Gradio ImageEditor white-screened on tool switches; this one cannot.
 *
 * The mask lives in an offscreen canvas at "edit resolution" (long side
 * capped server-side). Edits are POSTed back as a PNG; the server applies
 * only the CHANGED pixels to the full-resolution mask.
 */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

/* ------------------------------------------------------------------ */
/* State                                                                */
/* ------------------------------------------------------------------ */
const S = {
  image: null,            // {name,width,height,megapixels,georef}
  edit: { width: 0, height: 0, scale: 1 },
  result: null,           // stats from server
  maskVersion: -1,

  photo: null,            // ImageBitmap (edit res)
  skel: null, skelCtx: null,      // EDITABLE centerline canvas (red-on-transparent)
  sbase: null, sbaseCtx: null,    // pristine copy of skel — client-side diff basis
  vectors: null,          // centerline polylines (edit-res coords) — same as exports
  vectorPaths: null,      // Path2D cache for crisp CAD-style rendering
  skelTint: null,         // skeleton recolored black, for the Clean view
  mask: null, maskCtx: null,      // offscreen canvas, white-on-transparent
  tint: null, tintCtx: null, tintDirty: true,
  backup: null, backupCtx: null,  // pre-stroke snapshot for undo rects
  hasMask: false, hasSkel: false,
  dirty: false,           // unapplied local edits

  mode: "overlay",
  tool: "pan",
  brushSize: 3, eraserSize: 24, penSize: 3,
  opacity: 0.55,

  view: { k: 1, tx: 0, ty: 0 },
  spaceDown: false,
  pointer: { x: -1, y: -1, over: false },
  pan: null,              // {x,y} while panning
  stroke: null,           // {lastX,lastY,minX,minY,maxX,maxY} while painting

  history: [], redoStack: [], historyBytes: 0,
  selection: null,        // {x,y,w,h,canvas,count}

  // Direct vector editing (node tool + vector line pen). These edit S.vectors
  // in place; "Apply edits" POSTs them to /api/set_vectors as authoritative
  // geometry. Independent of the raster mask/skeleton history above.
  vecDirty: false,        // unapplied vector edits
  nodeSel: null,          // {li, vi}  (vi = -1 → whole line selected)
  nodeDrag: false, nodeMoved: false,
  nodeMarquee: null,      // {x0,y0,x1,y1} image coords while rubber-banding
  nodeMulti: [],          // [{li,vi},...] marquee-selected vertices
  vecPen: null,           // in-progress new polyline (edit-res points) or null
  vecHistory: [], vecRedo: [],   // JSON snapshots of S.vectors for undo/redo

  jobTimer: null,
  busy: false,
};

const VEC_HISTORY_MAX = 40;
const NODE_HIT_PX = 8;   // screen-px grab radius for vertices / segments
const SNAP_PX = 12;      // screen-px snap radius for the vector line pen

const HISTORY_MAX = 40;
const HISTORY_BYTES_MAX = 320 * 1024 * 1024;
const MASK_COLOR = "#00c8dc";
const SKEL_COLOR = "#ff4040";

/* Which layer a tool touches right now. The pen always edits centerlines;
 * eraser and select follow the view: over Skeleton/Clean they act on the
 * centerlines, everywhere else on the mask. The brush always edits the mask. */
function activeLayer(tool = S.tool) {
  if (tool === "pen") return "skel";
  if ((tool === "eraser" || tool === "select") &&
      (S.mode === "skeleton" || S.mode === "clean")) return "skel";
  return "mask";
}
function layerCtx(layer) { return layer === "skel" ? S.skelCtx : S.maskCtx; }
function markLayerDirty(layer) {
  if (layer === "skel") S.skelTint = null; else S.tintDirty = true;
}

/* Centerlines are DISPLAYED as crisp vector paths (exactly what the export
 * writes) whenever there are no pending local centerline edits; during
 * editing the AA raster edit canvas is shown instead, so eraser strokes
 * remove lines live. */
function hasPendingSkelEdits() {
  return S.stroke?.layer === "skel" ||
         S.history.some((e) => e.layer === "skel") ||
         S.redoStack.some((e) => e.layer === "skel");
}

function strokeVectors(c2d, color) {
  if (!S.vectorPaths) return;
  c2d.save();
  c2d.strokeStyle = color;
  c2d.lineWidth = Math.max(1.5 / S.view.k, 0.15); // constant ~1.5 px on screen
  c2d.lineJoin = "round";
  c2d.lineCap = "round";
  for (const p of S.vectorPaths) c2d.stroke(p);
  c2d.restore();
}

/* ------------------------------------------------------------------ */
/* API helpers                                                          */
/* ------------------------------------------------------------------ */
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function apiForm(url, fd) {
  const r = await fetch(url, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function toast(msg, kind = "", ms = 5000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function setBusy(b) {
  S.busy = b;
  document.body.classList.toggle("busy", b);
}

/* ------------------------------------------------------------------ */
/* Canvas & rendering                                                   */
/* ------------------------------------------------------------------ */
const canvas = $("#view");
const ctx = canvas.getContext("2d");
let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

function canvasSize() {
  const r = canvas.parentElement.getBoundingClientRect();
  return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
}

function ensureCanvasResolution() {
  const dpr = window.devicePixelRatio || 1;
  const { w, h } = canvasSize();
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw; canvas.height = ph;
  }
}

function updateTint() {
  if (!S.tintDirty || !S.mask) return;
  S.tintCtx.globalCompositeOperation = "source-over";
  S.tintCtx.clearRect(0, 0, S.tint.width, S.tint.height);
  S.tintCtx.drawImage(S.mask, 0, 0);
  S.tintCtx.globalCompositeOperation = "source-in";
  S.tintCtx.fillStyle = MASK_COLOR;
  S.tintCtx.fillRect(0, 0, S.tint.width, S.tint.height);
  S.tintCtx.globalCompositeOperation = "source-over";
  S.tintDirty = false;
}

function draw() {
  ensureCanvasResolution();
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0a0e12";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!S.photo) return;

  const { k, tx, ty } = S.view;
  ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);
  ctx.imageSmoothingEnabled = k < 3;

  const ew = S.edit.width, eh = S.edit.height;
  if (S.mode === "original") {
    ctx.drawImage(S.photo, 0, 0);
  } else if (S.mode === "overlay") {
    ctx.drawImage(S.photo, 0, 0);
    if (S.hasMask) {
      updateTint();
      ctx.globalAlpha = S.opacity;
      ctx.drawImage(S.tint, 0, 0);
      ctx.globalAlpha = 1;
    }
    // While the line pen or node tool is in hand, show the centerlines.
    if (S.hasSkel && (S.tool === "pen" || S.tool === "node" || S.stroke?.layer === "skel")) {
      if (hasPendingSkelEdits()) ctx.drawImage(S.skel, 0, 0);
      else strokeVectors(ctx, SKEL_COLOR);
    }
  } else if (S.mode === "mask") {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, ew, eh);
    if (S.hasMask) ctx.drawImage(S.mask, 0, 0);
  } else if (S.mode === "skeleton") {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, ew, eh);
    ctx.globalAlpha = 0.45;
    ctx.drawImage(S.photo, 0, 0);
    ctx.globalAlpha = 1;
    if (S.hasSkel) {
      if (hasPendingSkelEdits()) ctx.drawImage(S.skel, 0, 0);
      else strokeVectors(ctx, SKEL_COLOR);
    }
  } else if (S.mode === "clean") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, ew, eh);
    if (S.hasSkel) {
      if (hasPendingSkelEdits()) { updateSkelTint(); ctx.drawImage(S.skelTint, 0, 0); }
      else strokeVectors(ctx, "#000000");
    }
  }

  if (S.selection) {
    const onSkelView = S.mode === "skeleton" || S.mode === "clean";
    if ((S.selection.layer === "skel") === onSkelView) {
      ctx.drawImage(S.selection.canvas, S.selection.x, S.selection.y);
    }
  }

  // Vector line pen: preview the in-progress polyline (edit-space transform
  // is still active here) with a rubber-band segment to the cursor. The
  // cursor snaps to nearby existing vertices/segments so joins are exact.
  let penSnap = null;
  if (S.tool === "pen" && S.pointer.over) {
    const c = toImage(S.pointer.x, S.pointer.y);
    penSnap = snapPoint(c.x, c.y);
    if (S.vecPen && S.vecPen.length) {
      ctx.save();
      ctx.strokeStyle = SKEL_COLOR;
      ctx.lineWidth = Math.max(1.5 / k, 0.15);
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(S.vecPen[0][0], S.vecPen[0][1]);
      for (let i = 1; i < S.vecPen.length; i++) ctx.lineTo(S.vecPen[i][0], S.vecPen[i][1]);
      if (penSnap) ctx.lineTo(penSnap.x, penSnap.y); else ctx.lineTo(c.x, c.y);
      ctx.stroke();
      ctx.restore();
    }
  } else if (S.tool === "pen" && S.vecPen && S.vecPen.length > 1) {
    ctx.save();
    ctx.strokeStyle = SKEL_COLOR;
    ctx.lineWidth = Math.max(1.5 / k, 0.15);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(S.vecPen[0][0], S.vecPen[0][1]);
    for (let i = 1; i < S.vecPen.length; i++) ctx.lineTo(S.vecPen[i][0], S.vecPen[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  // Marquee rectangle (node tool rubber-band selection).
  if (S.nodeMarquee) {
    const m = S.nodeMarquee;
    ctx.save();
    ctx.strokeStyle = "#ffd400";
    ctx.lineWidth = 1.2 / k;
    ctx.setLineDash([6 / k, 4 / k]);
    ctx.strokeRect(Math.min(m.x0, m.x1), Math.min(m.y0, m.y1),
                   Math.abs(m.x1 - m.x0), Math.abs(m.y1 - m.y0));
    ctx.restore();
  }

  // Node handles for the selected centerline (screen space, constant size).
  if (S.tool === "node" && S.nodeSel && S.vectors && S.vectors[S.nodeSel.li]) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const line = S.vectors[S.nodeSel.li];
    for (let i = 0; i < line.length; i++) {
      const sx = S.view.tx + line[i][0] * k;
      const sy = S.view.ty + line[i][1] * k;
      const on = i === S.nodeSel.vi;
      ctx.beginPath();
      ctx.rect(sx - 3.5, sy - 3.5, 7, 7);
      ctx.fillStyle = on ? "#ffd400" : "#00e5ff";
      ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "#04222a"; ctx.stroke();
    }
  }

  // Marquee-selected vertices (screen space, yellow).
  if (S.tool === "node" && S.nodeMulti.length && S.vectors) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const { li, vi } of S.nodeMulti) {
      const p = S.vectors[li] && S.vectors[li][vi];
      if (!p) continue;
      const sx = S.view.tx + p[0] * k;
      const sy = S.view.ty + p[1] * k;
      ctx.beginPath();
      ctx.rect(sx - 3.5, sy - 3.5, 7, 7);
      ctx.fillStyle = "#ffd400"; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "#04222a"; ctx.stroke();
    }
  }

  // Pen snap indicator: green ring on the point the next click will land on.
  if (penSnap) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const sx = S.view.tx + penSnap.x * k;
    const sy = S.view.ty + penSnap.y * k;
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = "#3dff8a";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Brush cursor (screen space) — mask paint tools only.
  if (S.pointer.over && (S.tool === "brush" || S.tool === "eraser")) {
    const size = S.tool === "brush" ? S.brushSize : S.eraserSize;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.beginPath();
    ctx.arc(S.pointer.x, S.pointer.y, Math.max(1.5, (size / 2) * k), 0, Math.PI * 2);
    ctx.strokeStyle = S.tool === "eraser" ? "#ff6666" : "#00e5ff";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  $("#zoomlabel").textContent = `${Math.round(k * 100)}%`;
}

function updateSkelTint() {
  if (S.skelTint || !S.hasSkel) return;
  const c = document.createElement("canvas");
  c.width = S.edit.width; c.height = S.edit.height;
  const t = c.getContext("2d");
  t.drawImage(S.skel, 0, 0);
  t.globalCompositeOperation = "source-in";
  t.fillStyle = "#000000";
  t.fillRect(0, 0, c.width, c.height);
  S.skelTint = c;
}

function fitView() {
  if (!S.photo) return;
  const { w, h } = canvasSize();
  const k = Math.min(w / S.edit.width, h / S.edit.height) * 0.97;
  S.view.k = k;
  S.view.tx = (w - S.edit.width * k) / 2;
  S.view.ty = (h - S.edit.height * k) / 2;
  requestDraw();
}

function toImage(px, py) {
  return { x: (px - S.view.tx) / S.view.k, y: (py - S.view.ty) / S.view.k };
}

/* ------------------------------------------------------------------ */
/* Undo / redo (dirty-rect history)                                     */
/* ------------------------------------------------------------------ */
function clampRect(x0, y0, x1, y1) {
  const ew = S.edit.width, eh = S.edit.height;
  const x = Math.max(0, Math.floor(x0)), y = Math.max(0, Math.floor(y0));
  const X = Math.min(ew, Math.ceil(x1)), Y = Math.min(eh, Math.ceil(y1));
  if (X - x < 1 || Y - y < 1) return null;
  return { x, y, w: X - x, h: Y - y };
}

function pushHistory(rect, beforeData, layer) {
  const entry = { ...rect, data: beforeData, layer };
  S.history.push(entry);
  S.historyBytes += beforeData.data.length;
  S.redoStack = [];
  while (S.history.length > HISTORY_MAX || S.historyBytes > HISTORY_BYTES_MAX) {
    const old = S.history.shift();
    if (!old) break;
    S.historyBytes -= old.data.data.length;
  }
  S.dirty = true;
  updateEditButtons();
}

function undo() {
  const e = S.history.pop();
  if (!e) return;
  S.historyBytes -= e.data.data.length;
  const ctx2 = layerCtx(e.layer);
  const cur = ctx2.getImageData(e.x, e.y, e.w, e.h);
  ctx2.putImageData(e.data, e.x, e.y);
  S.redoStack.push({ x: e.x, y: e.y, w: e.w, h: e.h, data: cur, layer: e.layer });
  markLayerDirty(e.layer);
  clearSelection();
  updateEditButtons();
  requestDraw();
}

function redo() {
  const e = S.redoStack.pop();
  if (!e) return;
  const ctx2 = layerCtx(e.layer);
  const cur = ctx2.getImageData(e.x, e.y, e.w, e.h);
  ctx2.putImageData(e.data, e.x, e.y);
  S.history.push({ x: e.x, y: e.y, w: e.w, h: e.h, data: cur, layer: e.layer });
  S.historyBytes += cur.data.length;
  markLayerDirty(e.layer);
  S.dirty = true;
  clearSelection();
  updateEditButtons();
  requestDraw();
}

function updateEditButtons() {
  const vecCtx = S.tool === "node" || S.tool === "pen";
  $("#undobtn").disabled = vecCtx ? S.vecHistory.length === 0 : S.history.length === 0;
  $("#redobtn").disabled = vecCtx ? S.vecRedo.length === 0 : S.redoStack.length === 0;
  $("#applyedits").disabled = !(S.dirty || S.vecDirty) || !S.result;
  $("#delselbtn").hidden = !S.selection && !S.nodeMulti.length;
}

/* ------------------------------------------------------------------ */
/* Painting                                                             */
/* ------------------------------------------------------------------ */
function toolSize() {
  return S.tool === "brush" ? S.brushSize
       : S.tool === "pen" ? S.penSize : S.eraserSize;
}

function strokeBegin(ix, iy) {
  const layer = activeLayer();
  // Snapshot the whole layer cheaply (canvas->canvas copy stays on GPU);
  // the undo entry extracts only the stroke's bounding rect at the end.
  S.backupCtx.clearRect(0, 0, S.backup.width, S.backup.height);
  S.backupCtx.drawImage(layer === "skel" ? S.skel : S.mask, 0, 0);
  S.stroke = { layer, size: toolSize(),
               lastX: ix, lastY: iy, minX: ix, minY: iy, maxX: ix, maxY: iy };
  strokeSegment(ix, iy, ix, iy);
}

function strokeSegment(x0, y0, x1, y1) {
  const st = S.stroke;
  const m = layerCtx(st.layer);
  const color = st.layer === "skel" ? SKEL_COLOR : "#ffffff";
  m.save();
  m.globalCompositeOperation = S.tool === "eraser" ? "destination-out" : "source-over";
  m.strokeStyle = color;
  m.fillStyle = color;
  m.lineWidth = st.size;
  m.lineCap = "round";
  m.lineJoin = "round";
  if (x0 === x1 && y0 === y1) {
    m.beginPath();
    m.arc(x0, y0, st.size / 2, 0, Math.PI * 2);
    m.fill();
  } else {
    m.beginPath();
    m.moveTo(x0, y0);
    m.lineTo(x1, y1);
    m.stroke();
  }
  m.restore();
  st.minX = Math.min(st.minX, x0, x1); st.maxX = Math.max(st.maxX, x0, x1);
  st.minY = Math.min(st.minY, y0, y1); st.maxY = Math.max(st.maxY, y0, y1);
  markLayerDirty(st.layer);
}

function strokeEnd() {
  const st = S.stroke;
  S.stroke = null;
  if (!st) return;
  const pad = st.size / 2 + 2;
  const rect = clampRect(st.minX - pad, st.minY - pad, st.maxX + pad, st.maxY + pad);
  if (!rect) return;
  pushHistory(rect, S.backupCtx.getImageData(rect.x, rect.y, rect.w, rect.h), st.layer);
}

/* ------------------------------------------------------------------ */
/* Component selection (flood fill on the mask alpha)                   */
/* ------------------------------------------------------------------ */
function clearSelection() {
  if (S.selection) { S.selection = null; updateEditButtons(); requestDraw(); }
}

function selectComponentAt(ix, iy) {
  if (!S.hasMask) { toast("Run inference first — there is no mask yet."); return; }
  const layer = activeLayer("select");
  const what = layer === "skel" ? "centerline" : "mask component";
  const ew = S.edit.width, eh = S.edit.height;
  const x0 = Math.floor(ix), y0 = Math.floor(iy);
  if (x0 < 0 || y0 < 0 || x0 >= ew || y0 >= eh) return;

  const img = layerCtx(layer).getImageData(0, 0, ew, eh);
  const a = img.data;
  const at = (x, y) => a[(y * ew + x) * 4 + 3] > 0;

  // 1 px centerlines are hard to hit exactly — snap to the nearest "on"
  // pixel within a small radius before giving up.
  let sx = -1, sy = -1;
  outer:
  for (let rad = 0; rad <= (layer === "skel" ? 6 : 0); rad++) {
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const x = x0 + dx, y = y0 + dy;
        if (x >= 0 && y >= 0 && x < ew && y < eh && at(x, y)) { sx = x; sy = y; break outer; }
      }
    }
  }
  if (sx < 0) { clearSelection(); toast(`No ${what} under that point.`); return; }

  // Flood fill (8-connected — skeleton pixels chain diagonally).
  const seen = new Uint8Array(ew * eh);
  const stack = new Int32Array(ew * eh);
  let sp = 0, count = 0;
  let minX = sx, maxX = sx, minY = sy, maxY = sy;
  stack[sp++] = sy * ew + sx;
  seen[sy * ew + sx] = 1;
  while (sp > 0) {
    const p = stack[--sp];
    const px = p % ew, py = (p / ew) | 0;
    count++;
    if (px < minX) minX = px; if (px > maxX) maxX = px;
    if (py < minY) minY = py; if (py > maxY) maxY = py;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= ew || ny >= eh) continue;
        const np = ny * ew + nx;
        if (!seen[np] && at(nx, ny)) { seen[np] = 1; stack[sp++] = np; }
      }
    }
  }

  const w = maxX - minX + 1, h = maxY - minY + 1;
  const selCanvas = document.createElement("canvas");
  selCanvas.width = w; selCanvas.height = h;
  const sctx = selCanvas.getContext("2d");
  const sel = sctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (seen[(minY + y) * ew + (minX + x)]) {
        const o = (y * w + x) * 4;
        sel.data[o] = 255; sel.data[o + 1] = 220; sel.data[o + 2] = 0; sel.data[o + 3] = 200;
      }
    }
  }
  sctx.putImageData(sel, 0, 0);
  S.selection = { x: minX, y: minY, w, h, canvas: selCanvas, seen, count, layer };
  updateEditButtons();
  requestDraw();
  toast(`Selected ${what} (${count.toLocaleString()} px) — press Delete to remove it.`, "ok", 3500);
}

function deleteSelection() {
  const sel = S.selection;
  if (!sel) return;
  const ctx2 = layerCtx(sel.layer);
  const before = ctx2.getImageData(sel.x, sel.y, sel.w, sel.h);
  const cur = ctx2.getImageData(sel.x, sel.y, sel.w, sel.h);
  const ew = S.edit.width;
  for (let y = 0; y < sel.h; y++) {
    for (let x = 0; x < sel.w; x++) {
      if (sel.seen[(sel.y + y) * ew + (sel.x + x)]) {
        cur.data[(y * sel.w + x) * 4 + 3] = 0;
      }
    }
  }
  ctx2.putImageData(cur, sel.x, sel.y);
  pushHistory({ x: sel.x, y: sel.y, w: sel.w, h: sel.h }, before, sel.layer);
  S.selection = null;
  markLayerDirty(sel.layer);
  updateEditButtons();
  requestDraw();
  toast("Deleted — “Apply edits” rebuilds the skeleton/vectors.", "ok", 3500);
}

/* ------------------------------------------------------------------ */
/* Direct vector editing (node tool + vector line pen)                  */
/*                                                                      */
/* These operate on S.vectors (edit-res polylines) directly, which are  */
/* bit-for-bit the exported geometry — so what you drag is what exports,*/
/* with no skeletonize/spline round-trip. Persisted via /api/set_vectors*/
/* ------------------------------------------------------------------ */
function markVecDirty() {
  S.vecDirty = true;
  updateEditButtons();
  updateStats();
}

function snapshotVectors() {
  S.vecRedo = [];
  S.vecHistory.push(JSON.stringify(S.vectors || []));
  if (S.vecHistory.length > VEC_HISTORY_MAX) S.vecHistory.shift();
}

function vecUndo() {
  if (!S.vecHistory.length) return;
  S.vecRedo.push(JSON.stringify(S.vectors || []));
  S.vectors = JSON.parse(S.vecHistory.pop());
  rebuildVectorPaths();
  S.nodeSel = null; S.nodeMulti = [];   // vertex indices are stale now
  markVecDirty();
  requestDraw();
}

function vecRedo() {
  if (!S.vecRedo.length) return;
  S.vecHistory.push(JSON.stringify(S.vectors || []));
  S.vectors = JSON.parse(S.vecRedo.pop());
  rebuildVectorPaths();
  S.nodeSel = null; S.nodeMulti = [];
  markVecDirty();
  requestDraw();
}

/* Rebuild the Path2D cache used by strokeVectors() from S.vectors. */
function rebuildVectorPaths() {
  S.vectorPaths = (S.vectors || []).map(pathForLine);
}
function pathForLine(line) {
  const p = new Path2D();
  if (line && line.length) {
    p.moveTo(line[0][0], line[0][1]);
    for (let i = 1; i < line.length; i++) p.lineTo(line[i][0], line[i][1]);
  }
  return p;
}
function rebuildOnePath(li) {
  if (!S.vectorPaths) S.vectorPaths = [];
  S.vectorPaths[li] = pathForLine(S.vectors[li]);
}

/* --- geometry hit-testing (all in edit-res image coords) --------------- */
function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return { d: Math.hypot(px - cx, py - cy), cx, cy };
}
function nearestVertex(ix, iy, tol) {
  let best = null;
  const V = S.vectors || [];
  for (let li = 0; li < V.length; li++) {
    const line = V[li];
    for (let vi = 0; vi < line.length; vi++) {
      const d = Math.hypot(ix - line[vi][0], iy - line[vi][1]);
      if (d <= tol && (!best || d < best.d)) best = { li, vi, d };
    }
  }
  return best;
}
function nearestSegment(ix, iy, tol) {
  let best = null;
  const V = S.vectors || [];
  for (let li = 0; li < V.length; li++) {
    const line = V[li];
    for (let vi = 0; vi < line.length - 1; vi++) {
      const r = segDist(ix, iy, line[vi][0], line[vi][1], line[vi + 1][0], line[vi + 1][1]);
      if (r.d <= tol && (!best || r.d < best.d)) best = { li, vi, d: r.d, cx: r.cx, cy: r.cy };
    }
  }
  return best;
}

/* Snap target for the line pen: endpoints of existing lines win (that's how
 * you JOIN a new line to an old one), then any vertex, then the closest
 * point ON a segment (so you can T-join into the middle of a line). */
function snapPoint(ix, iy) {
  const tol = SNAP_PX / S.view.k;
  const V = S.vectors || [];
  let end = null;
  for (let li = 0; li < V.length; li++) {
    const line = V[li];
    for (const vi of [0, line.length - 1]) {
      const d = Math.hypot(ix - line[vi][0], iy - line[vi][1]);
      if (d <= tol && (!end || d < end.d)) end = { x: line[vi][0], y: line[vi][1], d };
    }
  }
  if (end) return { x: end.x, y: end.y, kind: "end" };
  const v = nearestVertex(ix, iy, tol);
  if (v) { const p = V[v.li][v.vi]; return { x: p[0], y: p[1], kind: "vertex" }; }
  const seg = nearestSegment(ix, iy, tol);
  if (seg) return { x: seg.cx, y: seg.cy, kind: "segment" };
  return null;
}

/* --- node tool operations ---------------------------------------------- */
function nodeSelectAt(ix, iy) {
  if (!(S.vectors && S.vectors.length)) {
    toast("No centerlines yet — run inference or draw with the Line pen.");
    return true;                 // nothing to marquee either
  }
  clearSelection();
  S.nodeMulti = [];
  const tol = NODE_HIT_PX / S.view.k;
  const v = nearestVertex(ix, iy, tol);
  if (v) {                       // grab a vertex → drag it
    snapshotVectors();
    S.nodeSel = { li: v.li, vi: v.vi };
    S.nodeDrag = true; S.nodeMoved = false;
    requestDraw();
    return true;
  }
  const seg = nearestSegment(ix, iy, tol);
  if (seg) {                     // click a line body → select whole line
    S.nodeSel = { li: seg.li, vi: -1 };
    requestDraw();
    return true;
  }
  S.nodeSel = null;
  requestDraw();
  return false;                  // empty space → caller starts a marquee
}

/* --- marquee: drag a rectangle over vertices to select them ------------ */
function marqueeFinish() {
  const m = S.nodeMarquee;
  S.nodeMarquee = null;
  if (!m) return;
  const x0 = Math.min(m.x0, m.x1), x1 = Math.max(m.x0, m.x1);
  const y0 = Math.min(m.y0, m.y1), y1 = Math.max(m.y0, m.y1);
  // A sub-pixel rectangle is just a click on empty space — clear selection.
  if (x1 - x0 < 1 / S.view.k && y1 - y0 < 1 / S.view.k) { requestDraw(); return; }
  const hits = [];
  const V = S.vectors || [];
  for (let li = 0; li < V.length; li++) {
    const line = V[li];
    for (let vi = 0; vi < line.length; vi++) {
      const p = line[vi];
      if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1) hits.push({ li, vi });
    }
  }
  S.nodeMulti = hits;
  updateEditButtons();
  requestDraw();
  if (hits.length) {
    toast(`Selected ${hits.length} point(s) — press Delete to remove them.`, "ok", 3500);
  }
}

function deleteMarkedNodes() {
  if (!S.nodeMulti.length) return;
  snapshotVectors();
  // Group per line, remove vertices back-to-front so indices stay valid.
  const byLine = new Map();
  for (const { li, vi } of S.nodeMulti) {
    if (!byLine.has(li)) byLine.set(li, []);
    byLine.get(li).push(vi);
  }
  const n = S.nodeMulti.length;
  const deadLines = [];
  for (const [li, vis] of byLine) {
    const line = S.vectors[li];
    if (!line) continue;
    vis.sort((a, b) => b - a);
    for (const vi of vis) line.splice(vi, 1);
    if (line.length < 2) deadLines.push(li);
  }
  deadLines.sort((a, b) => b - a);
  for (const li of deadLines) S.vectors.splice(li, 1);
  rebuildVectorPaths();
  S.nodeMulti = [];
  S.nodeSel = null;
  markVecDirty();
  requestDraw();
  toast(`Deleted ${n} point(s)` +
        (deadLines.length ? ` and ${deadLines.length} emptied line(s)` : "") +
        " — “Apply edits” saves this.", "ok", 3500);
}

function nodeDragTo(ix, iy) {
  const sel = S.nodeSel;
  if (!sel || sel.vi < 0) return;
  const line = S.vectors[sel.li];
  if (!line) return;
  line[sel.vi] = [ix, iy];
  rebuildOnePath(sel.li);
  if (!S.nodeMoved) { S.nodeMoved = true; markVecDirty(); }
  requestDraw();
}

function nodeDragEnd() {
  if (!S.nodeDrag) return;
  S.nodeDrag = false;
  // A click with no drag changed nothing — drop the snapshot we pushed.
  if (!S.nodeMoved && S.vecHistory.length) S.vecHistory.pop();
  S.nodeMoved = false;
}

function insertNodeAt(ix, iy) {
  const tol = NODE_HIT_PX / S.view.k;
  const seg = nearestSegment(ix, iy, tol);
  if (!seg) return;
  snapshotVectors();
  S.vectors[seg.li].splice(seg.vi + 1, 0, [seg.cx, seg.cy]);
  rebuildOnePath(seg.li);
  S.nodeSel = { li: seg.li, vi: seg.vi + 1 };
  markVecDirty();
  requestDraw();
}

function deleteNode() {
  const sel = S.nodeSel;
  if (!sel || !S.vectors[sel.li]) return;
  snapshotVectors();
  const line = S.vectors[sel.li];
  if (sel.vi < 0 || line.length <= 2) {
    S.vectors.splice(sel.li, 1);   // whole line, or a 2-pt line losing a vertex
  } else {
    line.splice(sel.vi, 1);
  }
  rebuildVectorPaths();
  S.nodeSel = null;
  markVecDirty();
  requestDraw();
}

/* --- vector line pen (click to place points) --------------------------- */
function vecPenAdd(ix, iy) {
  if (!S.result) { toast("Run inference first, then draw centerlines."); return; }
  if (S.mode === "mask" || S.mode === "original") setMode("overlay");
  const snap = snapPoint(ix, iy);
  if (!S.vecPen) S.vecPen = [];
  S.vecPen.push(snap ? [snap.x, snap.y] : [ix, iy]);
  requestDraw();
}

function finishVecPen(commit) {
  const pts = S.vecPen;
  S.vecPen = null;
  if (commit && pts && pts.length >= 2) {
    // A finishing double-click leaves a near-duplicate last point — drop it.
    const a = pts[pts.length - 1], b = pts[pts.length - 2];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.5) pts.pop();
  }
  if (commit && pts && pts.length >= 2) {
    snapshotVectors();
    if (!S.vectors) S.vectors = [];
    S.vectors.push(pts.map((p) => [p[0], p[1]]));
    rebuildOnePath(S.vectors.length - 1);
    S.hasSkel = true;
    markVecDirty();
  }
  requestDraw();
}

async function applyVectorEdits() {
  if (!S.result) { toast("Run inference first."); return; }
  if (S.vecPen) finishVecPen(true);
  setBusy(true);
  try {
    const sum = await apiPost("/api/set_vectors", { polylines: S.vectors || [] });
    applySummary(sum);
    await loadMaskAndSkeleton();
    updateStats();
    toast("Vector edits applied — exports now use your polylines.", "ok");
  } catch (err) {
    toast(`Applying vector edits failed: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ */
/* Pointer / keyboard interaction                                       */
/* ------------------------------------------------------------------ */
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("pointerdown", (e) => {
  if (!S.photo) return;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic/stale pointer */ }
  const wantPan = e.button === 1 || e.button === 2 || S.spaceDown || S.tool === "pan";
  if (wantPan) {
    S.pan = { x: e.offsetX, y: e.offsetY };
    canvas.classList.add("panning");
    return;
  }
  if (e.button !== 0) return;
  const p = toImage(e.offsetX, e.offsetY);
  if (S.tool === "pen") {
    vecPenAdd(p.x, p.y);            // vector line pen: drop a vertex
  } else if (S.tool === "node") {
    // Grab a vertex / select a line; on empty space start a marquee rectangle.
    if (!nodeSelectAt(p.x, p.y)) S.nodeMarquee = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  } else if (S.tool === "brush" || S.tool === "eraser") {
    if (!S.hasMask) { toast("Run inference first, then paint corrections."); return; }
    // The brush edits the mask — painting blind over Skeleton/Clean would be
    // confusing, so hop to Overlay.
    if (S.tool === "brush" && (S.mode === "skeleton" || S.mode === "clean")) setMode("overlay");
    clearSelection();
    strokeBegin(p.x, p.y);
    requestDraw();
  } else if (S.tool === "select") {
    selectComponentAt(p.x, p.y);
  }
});

canvas.addEventListener("pointermove", (e) => {
  S.pointer = { x: e.offsetX, y: e.offsetY, over: true };
  if (S.pan) {
    S.view.tx += e.offsetX - S.pan.x;
    S.view.ty += e.offsetY - S.pan.y;
    S.pan = { x: e.offsetX, y: e.offsetY };
    requestDraw();
    return;
  }
  if (S.stroke) {
    const p = toImage(e.offsetX, e.offsetY);
    strokeSegment(S.stroke.lastX, S.stroke.lastY, p.x, p.y);
    S.stroke.lastX = p.x; S.stroke.lastY = p.y;
  }
  if (S.nodeDrag) {
    const p = toImage(e.offsetX, e.offsetY);
    nodeDragTo(p.x, p.y);
  }
  if (S.nodeMarquee) {
    const p = toImage(e.offsetX, e.offsetY);
    S.nodeMarquee.x1 = p.x; S.nodeMarquee.y1 = p.y;
  }
  requestDraw();
});

function endPointer() {
  if (S.pan) { S.pan = null; canvas.classList.remove("panning"); }
  if (S.stroke) strokeEnd();
  if (S.nodeDrag) nodeDragEnd();
  if (S.nodeMarquee) marqueeFinish();
  requestDraw();
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("pointerleave", () => { S.pointer.over = false; requestDraw(); });

canvas.parentElement.addEventListener("wheel", (e) => {
  if (!S.photo) return;
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const k = Math.min(80, Math.max(0.02, S.view.k * factor));
  const cx = e.offsetX, cy = e.offsetY;
  S.view.tx = cx - (cx - S.view.tx) * (k / S.view.k);
  S.view.ty = cy - (cy - S.view.ty) * (k / S.view.k);
  S.view.k = k;
  requestDraw();
}, { passive: false });

canvas.addEventListener("dblclick", (e) => {
  if (S.tool === "pan") { fitView(); return; }
  if (S.tool === "pen") { finishVecPen(true); return; }   // finish the line
  if (S.tool === "node") {
    const p = toImage(e.offsetX, e.offsetY);
    insertNodeAt(p.x, p.y);                                // add a vertex on a segment
  }
});

window.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") {
    if (e.key !== "Escape") return;
  }
  if (e.code === "Space") { S.spaceDown = true; return; }
  // Node/pen tools have their own vector undo stack; other tools use the
  // raster (paint) history.
  const vecCtx = S.tool === "node" || S.tool === "pen";
  if (e.ctrlKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (e.shiftKey) { vecCtx ? vecRedo() : redo(); } else { vecCtx ? vecUndo() : undo(); }
    return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === "y") { e.preventDefault(); vecCtx ? vecRedo() : redo(); return; }
  if (e.ctrlKey) return;
  switch (e.key) {
    case "b": case "B": setTool("brush"); break;
    case "e": case "E": setTool("eraser"); break;
    case "p": case "P": setTool("pen"); break;
    case "n": case "N": setTool("node"); break;
    case "h": case "H": case "v": case "V": setTool("pan"); break;
    case "s": case "S": setTool("select"); break;
    case "f": case "F": fitView(); break;
    case "1": setMode("overlay"); break;
    case "2": setMode("original"); break;
    case "3": setMode("mask"); break;
    case "4": setMode("skeleton"); break;
    case "5": setMode("clean"); break;
    case "[": bumpSize(-2); break;
    case "]": bumpSize(2); break;
    case "Enter": if (S.vecPen) { e.preventDefault(); finishVecPen(true); } break;
    case "Delete": case "Backspace":
      if (S.tool === "node" && S.nodeMulti.length) { e.preventDefault(); deleteMarkedNodes(); }
      else if (S.tool === "node" && S.nodeSel) { e.preventDefault(); deleteNode(); }
      else if (S.selection) { e.preventDefault(); deleteSelection(); }
      break;
    case "Escape":
      if (S.vecPen) finishVecPen(false);
      else if (S.nodeMulti.length) { S.nodeMulti = []; updateEditButtons(); requestDraw(); }
      else if (S.nodeSel) { S.nodeSel = null; requestDraw(); }
      else clearSelection();
      break;
  }
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") S.spaceDown = false; });

/* ------------------------------------------------------------------ */
/* Tools / modes UI                                                     */
/* ------------------------------------------------------------------ */
const SIZE_KEYS = { brush: "brushSize", eraser: "eraserSize" };

function setTool(tool) {
  if (S.vecPen && tool !== "pen") finishVecPen(true);   // commit a dangling line
  if (tool !== "node") { S.nodeSel = null; S.nodeMulti = []; S.nodeMarquee = null; }
  S.tool = tool;
  $$("#tools .tbtn").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
  canvas.className = `tool-${tool}`;
  const hasSize = tool in SIZE_KEYS;
  $("#sizegroup").hidden = !hasSize;
  if (hasSize) {
    $("#toolsize").value = S[SIZE_KEYS[tool]];
    $("#toolsizeval").textContent = `${S[SIZE_KEYS[tool]]} px`;
  }
  updateEditButtons();
  requestDraw();
}

function bumpSize(d) {
  const key = SIZE_KEYS[S.tool];
  if (!key) return;
  S[key] = Math.min(100, Math.max(1, S[key] + d));
  $("#toolsize").value = S[key];
  $("#toolsizeval").textContent = `${S[key]} px`;
  requestDraw();
}

function setMode(mode) {
  S.mode = mode;
  $$("#modes .tbtn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  requestDraw();
}

$$("#modes .tbtn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$$("#tools .tbtn").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
$("#toolsize").addEventListener("input", (e) => {
  const v = parseInt(e.target.value, 10);
  const key = SIZE_KEYS[S.tool];
  if (key) S[key] = v;
  $("#toolsizeval").textContent = `${v} px`;
  requestDraw();
});
$("#undobtn").addEventListener("click", () => (S.tool === "node" || S.tool === "pen") ? vecUndo() : undo());
$("#redobtn").addEventListener("click", () => (S.tool === "node" || S.tool === "pen") ? vecRedo() : redo());
$("#delselbtn").addEventListener("click", () =>
  S.nodeMulti.length ? deleteMarkedNodes() : deleteSelection());
$("#fitbtn").addEventListener("click", fitView);

/* ------------------------------------------------------------------ */
/* Settings panel helpers                                               */
/* ------------------------------------------------------------------ */
function slider(id, valId, fmt = (v) => v) {
  const el = $(id), val = $(valId);
  el.addEventListener("input", () => { val.textContent = fmt(el.value); });
  return {
    get: () => parseFloat(el.value),
    set: (v) => { el.value = v; val.textContent = fmt(v); },
  };
}
const thresholdSl = slider("#threshold", "#thval");
const minCompSl = slider("#mincomp", "#mcval", (v) => `${v} px`);
const pruneSl = slider("#prune", "#prval", (v) => `${v} px`);
const minSkelSl = slider("#minskel", "#msval", (v) => `${v} px`);
const opacitySl = slider("#opacity", "#opval");
$("#opacity").addEventListener("input", () => { S.opacity = opacitySl.get(); requestDraw(); });

function ppSettings() {
  return {
    threshold: thresholdSl.get(),
    min_component_px: minCompSl.get(),
    prune_branch_px: pruneSl.get(),
    min_skeleton_px: minSkelSl.get(),
  };
}

/* ------------------------------------------------------------------ */
/* Server data loading                                                  */
/* ------------------------------------------------------------------ */
async function fetchBitmap(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  const blob = await r.blob();
  return { bitmap: await createImageBitmap(blob), headers: r.headers };
}

function setupMaskCanvases(w, h) {
  for (const name of ["mask", "tint", "backup", "skel", "sbase"]) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    S[name] = c;
    S[`${name}Ctx`] = c.getContext("2d", { willReadFrequently: name !== "tint" });
  }
  S.tintDirty = true;
  S.skelTint = null;
  S.vectors = null;
  S.vectorPaths = null;
}

function resetEditState() {
  S.history = []; S.redoStack = []; S.historyBytes = 0;
  S.selection = null; S.stroke = null; S.dirty = false;
  S.vecDirty = false; S.nodeSel = null; S.nodeDrag = false; S.nodeMoved = false;
  S.nodeMarquee = null; S.nodeMulti = [];
  S.vecPen = null; S.vecHistory = []; S.vecRedo = [];
  updateEditButtons();
}

async function loadPhoto() {
  const { bitmap } = await fetchBitmap("/api/image/photo.jpg");
  S.photo = bitmap;
  $("#emptystate").style.display = "none";
  setupMaskCanvases(S.edit.width, S.edit.height);
  S.hasMask = false;
  S.hasSkel = false;
  resetEditState();
  fitView();
}

async function loadMaskAndSkeleton() {
  const m = await fetchBitmap("/api/image/mask.png");
  S.maskVersion = parseInt(m.headers.get("X-Mask-Version") || "-1", 10);
  S.maskCtx.clearRect(0, 0, S.mask.width, S.mask.height);
  S.maskCtx.drawImage(m.bitmap, 0, 0);
  m.bitmap.close();
  S.hasMask = true;
  S.tintDirty = true;
  try {
    const vec = await apiGet("/api/vectors");
    S.vectors = vec.polylines;
    S.vectorPaths = S.vectors.map((line) => {
      const p = new Path2D();
      p.moveTo(line[0][0], line[0][1]);
      for (let i = 1; i < line.length; i++) p.lineTo(line[i][0], line[i][1]);
      return p;
    });
    renderSkelCanvas();
    S.hasSkel = true;
  } catch {
    S.vectors = null;
    S.vectorPaths = null;
    S.skelCtx.clearRect(0, 0, S.skel.width, S.skel.height);
    S.sbaseCtx.clearRect(0, 0, S.sbase.width, S.sbase.height);
    S.hasSkel = false;
  }
  S.skelTint = null;
  resetEditState();
  requestDraw();
}

/* Rasterize the vector centerlines into the EDIT canvas (the substrate the
 * eraser/pen/select tools operate on) and snapshot it as the diff basis. */
function renderSkelCanvas() {
  const c2d = S.skelCtx;
  c2d.clearRect(0, 0, S.skel.width, S.skel.height);
  c2d.save();
  c2d.strokeStyle = SKEL_COLOR;
  c2d.lineWidth = 1.6;
  c2d.lineJoin = "round";
  c2d.lineCap = "round";
  for (const p of S.vectorPaths) c2d.stroke(p);
  c2d.restore();
  S.sbaseCtx.clearRect(0, 0, S.sbase.width, S.sbase.height);
  S.sbaseCtx.drawImage(S.skel, 0, 0);
}

function applySummary(sum) {
  if (sum.image) S.image = sum.image;
  if (sum.edit) S.edit = sum.edit;
  S.result = sum.result || null;
  updateStats();
  updateEditButtons();
}

function updateStats() {
  const el = $("#stats");
  if (!S.image) { el.textContent = "Ready."; return; }
  let t = `<b>${S.image.name}</b> · ${S.image.width}×${S.image.height} (${S.image.megapixels} MP)`;
  if (S.image.georef) t += ` · ${S.image.georef}`;
  const r = S.result;
  if (r) {
    t += ` · inference ${r.inference_seconds}s · <b>${r.n_components}</b> components · ` +
         `<b>${r.n_centerlines}</b> centerlines · fg ${(r.fg_fraction * 100).toFixed(2)}%`;
    if (r.edited) t += " · ✏️ edited";
  } else {
    t += " · not processed yet";
  }
  if (S.dirty || S.vecDirty) t += " · <b>unapplied edits</b>";
  el.innerHTML = t;
  $("#imginfo").textContent = `${S.image.name} — ${S.image.width}×${S.image.height}`;
}

/* ------------------------------------------------------------------ */
/* Actions: open / infer / postprocess / apply edits                    */
/* ------------------------------------------------------------------ */
function confirmDiscardEdits() {
  return !(S.dirty || S.vecDirty) ||
    confirm("You have unapplied edits — they will be lost. Continue?");
}

async function openImage(fileOrPath) {
  if (!confirmDiscardEdits()) return;
  const fd = new FormData();
  if (fileOrPath instanceof File) fd.append("file", fileOrPath);
  else fd.append("path", fileOrPath);
  setBusy(true);
  try {
    const sum = await apiForm("/api/open", fd);
    applySummary(sum);
    await loadPhoto();
    updateStats();
    if (S.image.megapixels > 40) {
      toast(`${S.image.megapixels} MP image — inference will take a while ` +
            `(TTA would make it ~4× slower; leave it off unless needed).`, "", 9000);
    }
  } catch (err) {
    toast(`Could not open image: ${err.message}`, "error", 8000);
  } finally {
    setBusy(false);
  }
}

$("#openbtn").addEventListener("click", () => $("#fileinput").click());
$("#fileinput").addEventListener("change", (e) => {
  if (e.target.files[0]) openImage(e.target.files[0]);
  e.target.value = "";
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  // With the batch panel open every drop is a batch drop, even if the
  // user misses the dashed zone.
  if (batchPanelOpen()) { stageBatchDrop(e.dataTransfer); return; }
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) openImage(f);
});

$("#runbtn").addEventListener("click", async () => {
  if (!S.image) { toast("Open an image first."); return; }
  if (!confirmDiscardEdits()) return;
  try {
    await apiPost("/api/infer", {
      model_key: $("#modelsel").value,
      use_tta: $("#ttacb").checked,
      ...ppSettings(),
    });
    watchJob(async () => {
      const sum = await apiGet("/api/status");
      applySummary(sum);
      await loadMaskAndSkeleton();
      updateStats();
      toast(`Inference done in ${sum.result.inference_seconds}s — ` +
            `${sum.result.n_components} components.`, "ok");
    });
  } catch (err) {
    toast(err.message, "error", 8000);
  }
});

$("#applysettings").addEventListener("click", async () => {
  if (!S.result) { toast("Run inference first."); return; }
  if (!confirmDiscardEdits()) return;
  setBusy(true);
  try {
    const sum = await apiPost("/api/postprocess", ppSettings());
    applySummary(sum);
    await loadMaskAndSkeleton();
    updateStats();
  } catch (err) {
    toast(err.message, "error", 8000);
  } finally {
    setBusy(false);
  }
});

$("#applyedits").addEventListener("click", async () => {
  if (S.vecDirty) { await applyVectorEdits(); return; }
  if (!S.dirty || !S.result) return;
  setBusy(true);
  try {
    const blob = await new Promise((res) => S.mask.toBlob(res, "image/png"));
    const fd = new FormData();
    fd.append("file", blob, "mask.png");
    if (S.hasSkel && hasPendingSkelEdits()) {
      // Client-side diff: which centerline pixels were added / removed
      // relative to the pristine render of the server's vectors.
      const ew = S.edit.width, eh = S.edit.height;
      const cur = S.skelCtx.getImageData(0, 0, ew, eh).data;
      const bas = S.sbaseCtx.getImageData(0, 0, ew, eh).data;
      const addImg = new ImageData(ew, eh), remImg = new ImageData(ew, eh);
      let nAdd = 0, nRem = 0;
      for (let i = 3; i < cur.length; i += 4) {
        const c = cur[i] > 10, b = bas[i] > 10;
        if (c && !b) { addImg.data[i] = 255; nAdd++; }
        else if (!c && b) { remImg.data[i] = 255; nRem++; }
      }
      const toBlobPng = (imgData) => {
        const c = document.createElement("canvas");
        c.width = ew; c.height = eh;
        c.getContext("2d").putImageData(imgData, 0, 0);
        return new Promise((res) => c.toBlob(res, "image/png"));
      };
      if (nAdd) fd.append("skel_add", await toBlobPng(addImg), "skel_add.png");
      if (nRem) fd.append("skel_rem", await toBlobPng(remImg), "skel_rem.png");
    }
    fd.append("mask_version", String(S.maskVersion));
    fd.append("prune_branch_px", String(pruneSl.get()));
    fd.append("min_skeleton_px", String(minSkelSl.get()));
    const sum = await apiForm("/api/apply_mask", fd);
    applySummary(sum);
    await loadMaskAndSkeleton();
    updateStats();
    toast("Edits applied — skeleton and vectors rebuilt.", "ok");
  } catch (err) {
    toast(`Applying edits failed: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
});

/* ------------------------------------------------------------------ */
/* Job polling                                                          */
/* ------------------------------------------------------------------ */
function watchJob(onDone) {
  setBusy(true);
  $("#jobarea").hidden = false;
  clearInterval(S.jobTimer);
  S.jobTimer = setInterval(async () => {
    let job;
    try { job = await apiGet("/api/job"); }
    catch { return; } // transient network hiccup — keep polling
    $("#jobfill").style.width = `${Math.round(job.progress * 100)}%`;
    $("#jobmsg").textContent = job.message || "";
    if (job.status === "running") return;
    clearInterval(S.jobTimer);
    $("#jobarea").hidden = true;
    setBusy(false);
    if (job.status === "done") {
      try { await onDone(job); } catch (err) { toast(err.message, "error", 8000); }
    } else if (job.status === "error") {
      toast(`Job failed: ${job.error}`, "error", 10000);
    } else if (job.status === "cancelled") {
      toast("Job cancelled.");
    }
  }, 400);
}

$("#cancelbtn").addEventListener("click", () => apiPost("/api/cancel").catch(() => {}));

/* ------------------------------------------------------------------ */
/* Export                                                               */
/* ------------------------------------------------------------------ */
$("#exportbtn").addEventListener("click", async () => {
  if (!S.result) { toast("Nothing to export yet — run inference first."); return; }
  if (S.dirty) { toast("You have unapplied edits — click “Apply edits” first so the export includes them.", "error", 7000); return; }
  const choices = $$("#exportchoices input:checked").map((c) => c.value);
  setBusy(true);
  try {
    const res = await apiPost("/api/export", {
      choices,
      out_dir: $("#outdir").value,
      append_master: $("#mastercb").checked,
      opacity: S.opacity,
    });
    const st = $("#exportstatus");
    st.innerHTML = `Exported <b>${res.files.length}</b> file(s) to <code>${res.target}</code> ` +
      `<a href="#" id="revealout">open folder</a>` +
      (res.master ? `<br>Master updated: <code>${res.master}</code>` : "");
    $("#revealout").addEventListener("click", (e) => {
      e.preventDefault();
      apiPost("/api/reveal", { path: res.target }).catch((err) => toast(err.message, "error"));
    });
    toast(`Exported ${res.files.length} files.`, "ok");
  } catch (err) {
    toast(`Export failed: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
});

/* ------------------------------------------------------------------ */
/* Save to training set (staging folder, FINAL4 layout)                 */
/* ------------------------------------------------------------------ */
$("#trainbtn").addEventListener("click", async () => {
  if (!S.result) { toast("Run inference first — there is nothing to save."); return; }
  if (S.dirty) { toast("You have unapplied edits — click “Apply edits” first so the saved mask includes them.", "error", 7000); return; }
  setBusy(true);
  try {
    const res = await apiPost("/api/save_training", {
      out_dir: $("#traindir").value,
      opacity: S.opacity,
    });
    const st = $("#trainstatus");
    st.innerHTML = `Saved <b>${res.stem}</b> (image + mask + overlay)` +
      (res.edited ? " <b>with your edits</b>" : "") +
      ` — staging now holds <b>${res.n_pairs}</b> pair(s). ` +
      `<a href="#" id="revealtrain">open folder</a>`;
    $("#revealtrain").addEventListener("click", (e) => {
      e.preventDefault();
      apiPost("/api/reveal", { path: res.target }).catch((err) => toast(err.message, "error"));
    });
    toast(`Training pair saved: ${res.stem}`, "ok");
  } catch (err) {
    toast(`Saving training pair failed: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
});

/* ------------------------------------------------------------------ */
/* Batch                                                                */
/* ------------------------------------------------------------------ */
$("#batchtoggle").addEventListener("click", () => {
  const p = $("#batchpanel");
  const showBatch = p.hidden;
  p.hidden = !showBatch;
  $("#canvaswrap").style.display = showBatch ? "none" : "";
  $("#batchtoggle").classList.toggle("active", showBatch);
  if (!showBatch) requestDraw();
});

$("#batchrun").addEventListener("click", async () => {
  const choices = $$("#exportchoices input:checked").map((c) => c.value);
  try {
    await apiPost("/api/batch", {
      in_dir: $("#batchin").value,
      out_dir: $("#batchout").value,
      model_key: $("#modelsel").value,
      use_tta: $("#ttacb").checked,
      ...ppSettings(),
      choices,
      append_master: $("#batchmaster").checked,
      opacity: S.opacity,
    });
    watchJob(async (job) => {
      $("#batchsummary").textContent = job.message;
      renderBatchTable(job.rows || []);
    });
  } catch (err) {
    toast(err.message, "error", 8000);
  }
});

/* Drag & drop a folder (or a pile of images) onto the batch panel: the
   files are staged server-side and the input-folder field fills itself. */
const IMG_EXTS = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff"]);

function batchPanelOpen() { return !$("#batchpanel").hidden; }

async function stageBatchDrop(dt) {
  // webkitGetAsEntry() only works synchronously inside the drop event,
  // so collect the entries before the first await.
  const entries = [];
  for (const it of dt.items || []) {
    const e = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (e) entries.push(e);
  }
  let files = [];
  if (entries.length) {
    async function walk(entry) {
      if (entry.isFile) {
        files.push(await new Promise((res, rej) => entry.file(res, rej)));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        for (;;) { // readEntries hands out ≤100 per call — drain it
          const chunk = await new Promise((res, rej) => reader.readEntries(res, rej));
          if (!chunk.length) break;
          for (const e of chunk) await walk(e);
        }
      }
    }
    for (const e of entries) await walk(e);
  } else {
    files = [...dt.files];
  }
  const dot = (n) => n.includes(".") ? "." + n.split(".").pop().toLowerCase() : "";
  const wanted = files.filter((f) => IMG_EXTS.has(dot(f.name)));
  if (!wanted.length) {
    toast("No supported images in the drop (.jpg .jpeg .png .tif .tiff).", "error", 7000);
    return;
  }
  const fd = new FormData();
  for (const f of wanted) fd.append("files", f);
  setBusy(true);
  $("#batchsummary").textContent = `Uploading ${wanted.length} image(s)…`;
  try {
    const res = await apiForm("/api/batch_upload", fd);
    $("#batchin").value = res.dir;
    $("#batchsummary").textContent =
      `${res.count} image(s) staged — press ▶ Process folder.` +
      (res.skipped.length ? ` ${res.skipped.length} unsupported file(s) skipped.` : "");
    renderBatchTable([]);
    toast(`${res.count} image(s) ready for batch.`, "ok");
  } catch (err) {
    $("#batchsummary").textContent = "";
    toast(`Batch staging failed: ${err.message}`, "error", 8000);
  } finally {
    setBusy(false);
  }
}

{
  const panel = $("#batchpanel"), zone = $("#batchdrop");
  panel.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  panel.addEventListener("dragleave", (e) => {
    if (!panel.contains(e.relatedTarget)) zone.classList.remove("drag");
  });
  panel.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation(); // keep the single-image handler out of it
    zone.classList.remove("drag");
    stageBatchDrop(e.dataTransfer);
  });
}

let batchRows = [];

function renderBatchTable(rows) {
  batchRows = rows;
  if (!rows.length) { $("#batchtablewrap").innerHTML = ""; return; }
  const cells = rows.map((r, i) => {
    const name = (r.status === "ok" && r.path)
      ? `<a href="#" data-i="${i}" title="Open in the editor with its batch result">${r.file}</a>`
      : r.file;
    return `<tr class="${r.status === "ok" ? "" : "failed"}">` +
      `<td>${name}</td><td>${r.status}</td><td>${r.width}×${r.height}</td>` +
      `<td>${r.components}</td><td>${r.centerlines}</td><td>${r.seconds}</td></tr>`;
  }).join("");
  $("#batchtablewrap").innerHTML =
    `<table><tr><th>File</th><th>Status</th><th>Size</th><th>Components</th>` +
    `<th>Centerlines</th><th>Seconds</th></tr>${cells}</table>` +
    `<p class="hint">Click a file name to open it in the editor with its batch result (no re-inference).</p>`;
}

$("#batchtablewrap").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-i]");
  if (!a) return;
  e.preventDefault();
  const row = batchRows[+a.dataset.i];
  if (row) openBatchItem(row);
});

async function openBatchItem(row) {
  if (!confirmDiscardEdits()) return;
  setBusy(true);
  try {
    const sum = await apiPost("/api/batch_open",
                              { path: row.path, out_dir: row.out_dir, ...ppSettings() });
    applySummary(sum);
    await loadPhoto();
    if (sum.has_result) await loadMaskAndSkeleton();
    updateStats();
    // Re-exports should replace this item's batch artifacts in place.
    if (sum.mask_loaded) $("#outdir").value = row.out_dir;
    if (batchPanelOpen()) $("#batchtoggle").click(); // back to the editor
    if (sum.mask_loaded) {
      toast(`${row.file} loaded with its batch result — edit away.`, "ok");
    } else {
      toast(`${row.file} loaded, but no exported mask was found ` +
            `(enable "Binary mask" in Export for editable batches) — run inference.`, "", 9000);
    }
  } catch (err) {
    toast(`Could not open batch item: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ */
/* Init                                                                 */
/* ------------------------------------------------------------------ */
new ResizeObserver(() => requestDraw()).observe(canvas.parentElement);

async function init() {
  let st;
  try {
    st = await apiGet("/api/status");
  } catch (err) {
    toast(`Cannot reach the BoneSeg server: ${err.message}`, "error", 15000);
    return;
  }
  $("#version").textContent = `v${st.version}`;
  $("#device").textContent = st.device;
  const sel = $("#modelsel");
  sel.innerHTML = st.models.map((m) =>
    `<option value="${m.key}" title="${m.description}">${m.name}</option>`).join("");
  sel.value = st.model_key;
  $("#ttacb").checked = st.defaults.use_tta;
  thresholdSl.set(st.defaults.threshold);
  minCompSl.set(st.defaults.min_component_px);
  pruneSl.set(st.defaults.prune_branch_px);
  minSkelSl.set(st.defaults.min_skeleton_px);
  opacitySl.set(st.defaults.opacity);
  S.opacity = st.defaults.opacity;
  $("#outdir").value = st.defaults.out_dir;
  $("#batchout").value = st.defaults.out_dir + "\\batch";
  $("#traindir").value = st.defaults.train_dir;

  // Resume state if the server already has an image/result (page reload).
  if (st.has_image) {
    applySummary(st);
    await loadPhoto();
    if (st.has_result) await loadMaskAndSkeleton();
    updateStats();
  }
  const job = await apiGet("/api/job").catch(() => null);
  if (job && job.status === "running") {
    watchJob(async () => {
      const sum = await apiGet("/api/status");
      applySummary(sum);
      await loadPhoto();
      await loadMaskAndSkeleton();
      updateStats();
    });
  }
  setTool("pan");
  requestDraw();
}

init();
