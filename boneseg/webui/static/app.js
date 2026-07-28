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

  // SAM2 promptable segmentation (any feature, not just bones) — a prompt
  // session accumulates points/one box, previews a candidate mask from the
  // server, and "Accept" draws it into S.mask exactly like a brush stroke
  // (same undo history + Apply-edits path as everything else).
  samPoints: [],          // [{x,y,positive}] edit-res coords
  samBox: null,           // {x0,y0,x1,y1} edit-res coords, or null
  samBoxLive: null,       // in-progress drag rectangle, or null
  samDragStart: null,     // {x,y,shift,sx,sy} while a pointer is down
  samPreview: null,       // ImageBitmap of the last predicted mask, or null
  samScore: null,
  samBusy: false,

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
/* Tools that edit the centerline VECTORS (S.vectors) rather than a raster
 * layer — they share the vector undo/redo stack. The eraser is vector-based
 * only over Skeleton/Clean (where it trims centerlines); over the mask it
 * paints raster like the brush. */
function vectorToolActive() {
  return S.tool === "pen" || S.tool === "node" ||
         (S.tool === "eraser" && activeLayer() === "skel");
}
function markLayerDirty(layer) {
  if (layer === "skel") S.skelTint = null; else S.tintDirty = true;
}

/* Whether any raster centerline edit is pending vs the server's vectors —
 * used to decide the "Apply edits" skeleton diff (skel_add / skel_rem). */
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
    // While the line pen or node tool is in hand, show the centerlines
    // (always crisp vectors — never a rasterized skeleton).
    if (S.hasSkel && (S.tool === "pen" || S.tool === "node")) {
      strokeVectors(ctx, SKEL_COLOR);
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
    if (S.hasSkel) strokeVectors(ctx, SKEL_COLOR);
  } else if (S.mode === "clean") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, ew, eh);
    if (S.hasSkel) strokeVectors(ctx, "#000000");
  }

  if (S.selection) {
    const onSkelView = S.mode === "skeleton" || S.mode === "clean";
    if (S.selection.vectorLi != null) {
      // Highlight the selected centerline vector (crisp, screen-constant width).
      const path = S.vectorPaths && S.vectorPaths[S.selection.vectorLi];
      if (path && (S.tool === "select" || onSkelView)) {
        ctx.save();
        ctx.strokeStyle = "#ffd400";
        ctx.lineWidth = Math.max(2.5 / S.view.k, 0.2);
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        ctx.stroke(path);
        ctx.restore();
      }
    } else if ((S.selection.layer === "skel") === onSkelView) {
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

  // GCP markers (screen space): orange crosshair + index while georeferencing.
  if (S.tool === "gcp" && gcps.length) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    gcps.forEach((g, i) => {
      const sx = S.view.tx + g.px * k;
      const sy = S.view.ty + g.py * k;
      ctx.strokeStyle = "#ff9d00";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(sx - 9, sy); ctx.lineTo(sx + 9, sy);
      ctx.moveTo(sx, sy - 9); ctx.lineTo(sx, sy + 9);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.font = "bold 11px sans-serif";
      ctx.fillStyle = "#ff9d00";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 3;
      const lbl = `${i + 1}${g.id ? " " + g.id : ""}`;
      ctx.strokeText(lbl, sx + 8, sy - 8);
      ctx.fillText(lbl, sx + 8, sy - 8);
    });
  }

  // SAM prompt session: candidate mask preview (image space), then
  // points/box markers (screen space, constant size).
  if (S.tool === "sam" && S.samPreview) {
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);
    ctx.globalAlpha = 0.6;
    ctx.drawImage(S.samPreview, 0, 0);
    ctx.globalAlpha = 1;
  }
  if (S.tool === "sam") {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const box = S.samBoxLive || S.samBox;
    if (box) {
      ctx.strokeStyle = "#ffa500";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 4]);
      const sx0 = S.view.tx + box.x0 * k, sy0 = S.view.ty + box.y0 * k;
      const sx1 = S.view.tx + box.x1 * k, sy1 = S.view.ty + box.y1 * k;
      ctx.strokeRect(Math.min(sx0, sx1), Math.min(sy0, sy1), Math.abs(sx1 - sx0), Math.abs(sy1 - sy0));
      ctx.setLineDash([]);
    }
    for (const pt of S.samPoints) {
      const sx = S.view.tx + pt.x * k, sy = S.view.ty + pt.y * k;
      ctx.strokeStyle = pt.positive ? "#3dff8a" : "#ff4040";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx - 3, sy); ctx.lineTo(sx + 3, sy);
      if (!pt.positive) { /* minus sign only */ } else { ctx.moveTo(sx, sy - 3); ctx.lineTo(sx, sy + 3); }
      ctx.stroke();
    }
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
  const vecCtx = vectorToolActive();
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
  // On the skeleton the "component" is a VECTOR centerline: pick the whole
  // polyline under the cursor so Delete removes it as a vector (crisp, no
  // raster round-trip).
  if (layer === "skel") {
    const tol = NODE_HIT_PX / S.view.k;
    const seg = nearestSegment(ix, iy, Math.max(tol, 6 / S.view.k));
    if (!seg) { clearSelection(); toast("No centerline under that point."); return; }
    S.selection = { layer: "skel", vectorLi: seg.li };
    updateEditButtons();
    requestDraw();
    toast("Selected centerline — press Delete to remove it.", "ok", 3500);
    return;
  }
  const what = "mask component";
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
  if (sel.vectorLi != null) {           // vector centerline selection
    if (S.vectors && S.vectors[sel.vectorLi]) {
      snapshotVectors();
      S.vectors.splice(sel.vectorLi, 1);
      rebuildVectorPaths();
      markVecDirty();
    }
    S.selection = null;
    updateEditButtons();
    requestDraw();
    toast("Centerline deleted.", "ok", 2500);
    return;
  }
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

/* --- vector eraser: trim centerline polylines in place ------------------- */
/* Resample a polyline so no gap between consecutive points exceeds `step`,
 * so the circular eraser can cut mid-segment (not only at existing vertices). */
function densifyLine(line, step) {
  if (line.length < 2) return line.slice();
  const out = [];
  for (let i = 0; i < line.length - 1; i++) {
    const [x0, y0] = line[i], [x1, y1] = line[i + 1];
    const d = Math.hypot(x1 - x0, y1 - y0);
    const n = Math.max(1, Math.ceil(d / step));
    for (let k = 0; k < n; k++) out.push([x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n]);
  }
  out.push(line[line.length - 1].slice());
  return out;
}

/* Remove every polyline point within `r` of any point on the eraser stroke
 * segment (ex0,ey0)->(ex1,ey1); surviving runs of >=2 points become lines,
 * so erasing the middle of a line SPLITS it. Returns true if anything changed.
 * Operates directly on S.vectors — the centerlines stay crisp vectors, no
 * raster round-trip. */
function eraseVectorsAlong(ex0, ey0, ex1, ey1, r) {
  if (!S.vectors || !S.vectors.length) return false;
  const r2 = r * r;
  const step = Math.max(1, r / 2);
  // Pre-sample the eraser stroke itself so a fast drag still cuts continuously.
  const segLen = Math.hypot(ex1 - ex0, ey1 - ey0);
  const en = Math.max(1, Math.ceil(segLen / step));
  const epts = [];
  for (let k = 0; k <= en; k++) epts.push([ex0 + (ex1 - ex0) * k / en, ey0 + (ey1 - ey0) * k / en]);
  const minEx = Math.min(ex0, ex1) - r, maxEx = Math.max(ex0, ex1) + r;
  const minEy = Math.min(ey0, ey1) - r, maxEy = Math.max(ey0, ey1) + r;
  const erased = (x, y) => {
    for (const [px, py] of epts) {
      const dx = x - px, dy = y - py;
      if (dx * dx + dy * dy <= r2) return true;
    }
    return false;
  };
  let changed = false;
  const result = [];
  for (const line of S.vectors) {
    // bbox cull: skip lines the eraser can't possibly touch
    let miX = Infinity, miY = Infinity, maX = -Infinity, maY = -Infinity;
    for (const [x, y] of line) {
      if (x < miX) miX = x; if (x > maX) maX = x;
      if (y < miY) miY = y; if (y > maY) maY = y;
    }
    if (maX < minEx || miX > maxEx || maY < minEy || miY > maxEy) { result.push(line); continue; }
    const dl = densifyLine(line, step);
    let cur = [], hit = false;
    for (const p of dl) {
      if (erased(p[0], p[1])) { hit = true; if (cur.length >= 2) result.push(cur); cur = []; }
      else cur.push(p);
    }
    if (!hit) { result.push(line); continue; }   // untouched — keep original
    if (cur.length >= 2) result.push(cur);
    changed = true;
  }
  if (changed) { S.vectors = result; rebuildVectorPaths(); }
  return changed;
}

/* Snap target for the line pen: endpoints of existing lines win (that's how
 * you JOIN a new line to an old one), then any vertex, then the closest
 * point ON a segment (so you can T-join into the middle of a line).
 *
 * Snaps to ALL present centerlines — the user's own drawn lines AND the ones
 * the model produced. Only snaps to points that are actually VISIBLE: when a
 * raster skeleton edit is pending, a line the user just erased still lingers
 * in S.vectors until "Apply edits", so we reject any candidate whose pixel is
 * gone from the rendered skeleton raster — that was the source of the pen
 * snapping to "points that aren't there". */
function snapVisible(x, y) {
  // Only gate on visibility while the raster skeleton view is authoritative
  // (pending edits); otherwise S.vectors == what's drawn, so accept.
  if (!hasPendingSkelEdits() || !S.skelCtx) return true;
  const px = Math.round(x), py = Math.round(y);
  if (px < 0 || py < 0 || px >= S.skel.width || py >= S.skel.height) return true;
  // small neighborhood: the rendered stroke is a few px wide
  const r = 3;
  const x0 = Math.max(0, px - r), y0 = Math.max(0, py - r);
  const w = Math.min(S.skel.width - x0, r * 2 + 1), h = Math.min(S.skel.height - y0, r * 2 + 1);
  const d = S.skelCtx.getImageData(x0, y0, w, h).data;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 10) return true;
  return false;
}
function snapPoint(ix, iy) {
  const tol = SNAP_PX / S.view.k;
  const V = S.vectors || [];
  let end = null;
  for (let li = 0; li < V.length; li++) {
    const line = V[li];
    for (const vi of [0, line.length - 1]) {
      const d = Math.hypot(ix - line[vi][0], iy - line[vi][1]);
      if (d <= tol && (!end || d < end.d) && snapVisible(line[vi][0], line[vi][1]))
        end = { x: line[vi][0], y: line[vi][1], d };
    }
  }
  if (end) return { x: end.x, y: end.y, kind: "end" };
  const v = nearestVertex(ix, iy, tol);
  if (v) {
    const p = V[v.li][v.vi];
    if (snapVisible(p[0], p[1])) return { x: p[0], y: p[1], kind: "vertex" };
  }
  const seg = nearestSegment(ix, iy, tol);
  if (seg && snapVisible(seg.cx, seg.cy)) return { x: seg.cx, y: seg.cy, kind: "segment" };
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

/* --- SAM2 promptable segmentation (any feature, not just bones) -------- */
function updateSamStatus() {
  const el = $("#samstatus");
  if (!el) return;
  const n = S.samPoints.length;
  const parts = [];
  if (n) parts.push(`${n} point${n === 1 ? "" : "s"}`);
  if (S.samBox) parts.push("1 box");
  if (S.samBusy) {
    el.textContent = parts.length ? `${parts.join(" + ")} — predicting…` : "Predicting…";
  } else if (S.samPreview) {
    const pct = S.samScore != null ? ` (confidence ${Math.round(S.samScore * 100)}%)` : "";
    el.textContent = `${parts.join(" + ")} — preview ready${pct}. Accept, add more, or Esc to clear.`;
  } else if (parts.length) {
    el.textContent = `${parts.join(" + ")}.`;
  } else {
    el.textContent = "Click = include, Shift+click = exclude, drag = box.";
  }
  $("#samaccept").disabled = !S.samPreview;
  $("#samclear").disabled = !n && !S.samBox;
}

let samReqId = 0;
async function samPredict() {
  if (!S.samPoints.length && !S.samBox) { S.samPreview = null; updateSamStatus(); requestDraw(); return; }
  const myReq = ++samReqId;
  S.samBusy = true;
  updateSamStatus();
  try {
    const r = await fetch("/api/sam/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        points: S.samPoints.map((p) => ({ x: p.x, y: p.y, positive: p.positive })),
        box: S.samBox,
      }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    if (myReq !== samReqId) return;   // superseded by a newer prompt
    S.samScore = parseFloat(r.headers.get("X-Sam-Score") || "0");
    const blob = await r.blob();
    if (S.samPreview) S.samPreview.close();
    S.samPreview = await createImageBitmap(blob);
  } catch (err) {
    toast(`SAM prediction failed: ${err.message}`, "error", 8000);
  } finally {
    if (myReq === samReqId) S.samBusy = false;
    updateSamStatus();
    requestDraw();
  }
}

function samFinishDrag(e) {
  const start = S.samDragStart;
  S.samDragStart = null;
  const live = S.samBoxLive;
  S.samBoxLive = null;
  if (!start) return;
  // No pointermove fired at all (fast click) or near-zero movement — both
  // are a point click, not a box.
  const distScreen = live ? Math.hypot(e.offsetX - start.sx, e.offsetY - start.sy) : 0;
  if (!live || distScreen < 6) {
    S.samPoints.push({ x: start.x, y: start.y, positive: !start.shift });
  } else {
    S.samBox = {
      x0: Math.min(live.x0, live.x1), y0: Math.min(live.y0, live.y1),
      x1: Math.max(live.x0, live.x1), y1: Math.max(live.y0, live.y1),
    };
  }
  samPredict();
}

function samClear() {
  S.samPoints = []; S.samBox = null; S.samBoxLive = null; S.samDragStart = null;
  if (S.samPreview) { S.samPreview.close(); S.samPreview = null; }
  S.samScore = null; S.samBusy = false;
  samReqId++;   // orphan any in-flight predict so it can't land after clear
  updateSamStatus();
  requestDraw();
}

function samAccept() {
  if (!S.samPreview) { toast("Nothing to accept yet — click a point or drag a box first.", "error", 5000); return; }
  S.backupCtx.clearRect(0, 0, S.backup.width, S.backup.height);
  S.backupCtx.drawImage(S.mask, 0, 0);
  S.maskCtx.save();
  S.maskCtx.globalCompositeOperation = "source-over";
  S.maskCtx.drawImage(S.samPreview, 0, 0);
  S.maskCtx.restore();
  markLayerDirty("mask");
  const rect = { x: 0, y: 0, w: S.mask.width, h: S.mask.height };
  pushHistory(rect, S.backupCtx.getImageData(0, 0, rect.w, rect.h), "mask");
  toast("Region added to the mask — Apply edits to save, or keep annotating.", "ok", 6000);
  samClear();
}

$("#samaccept").addEventListener("click", samAccept);
$("#samclear").addEventListener("click", samClear);
$("#samblank").addEventListener("click", async () => {
  setBusy(true);
  try {
    const sum = await apiPost("/api/blank_canvas", {});
    applySummary(sum);
    await loadMaskAndSkeleton();
    updateStats();
    toast("Blank canvas ready — click points or drag a box to segment a feature.", "ok");
  } catch (err) {
    toast(`Could not start a blank canvas: ${err.message}`, "error", 8000);
  } finally {
    setBusy(false);
  }
});

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
    const line = pts.map((p) => [p[0], p[1]]);
    S.vectors.push(line);
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
    // Middle click otherwise triggers the browser's native autoscroll mode,
    // which hijacks the drag instead of panning the canvas.
    e.preventDefault();
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
  } else if (S.tool === "eraser" && activeLayer() === "skel") {
    // Vector eraser: over Skeleton/Clean the eraser trims the centerline
    // polylines directly, so lines stay crisp vectors while erasing (no raster).
    if (!S.vectors || !S.vectors.length) { toast("No centerlines to erase yet."); return; }
    clearSelection();
    snapshotVectors();
    S.vecErase = { lastX: p.x, lastY: p.y, changed: false };
    if (eraseVectorsAlong(p.x, p.y, p.x, p.y, S.eraserSize / 2)) S.vecErase.changed = true;
    requestDraw();
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
  } else if (S.tool === "gcp") {
    gcpAddPoint(p.x, p.y);
  } else if (S.tool === "sam") {
    if (!S.result) {
      toast("Run inference first, or click “Start blank canvas” to annotate without it.", "error", 8000);
      return;
    }
    S.samDragStart = { x: p.x, y: p.y, shift: e.shiftKey, sx: e.offsetX, sy: e.offsetY };
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
  if (S.vecErase) {
    const p = toImage(e.offsetX, e.offsetY);
    if (eraseVectorsAlong(S.vecErase.lastX, S.vecErase.lastY, p.x, p.y, S.eraserSize / 2))
      S.vecErase.changed = true;
    S.vecErase.lastX = p.x; S.vecErase.lastY = p.y;
  }
  if (S.nodeDrag) {
    const p = toImage(e.offsetX, e.offsetY);
    nodeDragTo(p.x, p.y);
  }
  if (S.nodeMarquee) {
    const p = toImage(e.offsetX, e.offsetY);
    S.nodeMarquee.x1 = p.x; S.nodeMarquee.y1 = p.y;
  }
  if (S.samDragStart) {
    const p = toImage(e.offsetX, e.offsetY);
    S.samBoxLive = { x0: S.samDragStart.x, y0: S.samDragStart.y, x1: p.x, y1: p.y };
  }
  requestDraw();
});

function endPointer(e) {
  if (S.pan) { S.pan = null; canvas.classList.remove("panning"); }
  if (S.stroke) strokeEnd();
  if (S.vecErase) {
    const changed = S.vecErase.changed;
    S.vecErase = null;
    if (changed) markVecDirty();
    else S.vecHistory.pop();   // nothing erased — discard the snapshot we took
  }
  if (S.nodeDrag) nodeDragEnd();
  if (S.nodeMarquee) marqueeFinish();
  if (S.samDragStart) samFinishDrag(e);
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
  // Vector tools (pen / node / skeleton eraser) have their own vector undo
  // stack; other tools use the raster (paint) history.
  const vecCtx = vectorToolActive();
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
    case "g": case "G": setTool("gcp"); break;
    case "m": case "M": setTool("sam"); break;
    case "f": case "F": fitView(); break;
    case "1": setMode("overlay"); break;
    case "2": setMode("original"); break;
    case "3": setMode("mask"); break;
    case "4": setMode("skeleton"); break;
    case "5": setMode("clean"); break;
    case "[": bumpSize(-2); break;
    case "]": bumpSize(2); break;
    case "Enter":
      if (S.vecPen) { e.preventDefault(); finishVecPen(true); }
      else if (S.tool === "sam" && S.samPreview) { e.preventDefault(); samAccept(); }
      break;
    case "Delete": case "Backspace":
      if (S.tool === "node" && S.nodeMulti.length) { e.preventDefault(); deleteMarkedNodes(); }
      else if (S.tool === "node" && S.nodeSel) { e.preventDefault(); deleteNode(); }
      else if (S.selection) { e.preventDefault(); deleteSelection(); }
      break;
    case "Escape":
      if (S.vecPen) finishVecPen(false);
      else if (S.tool === "sam" && (S.samPoints.length || S.samBox || S.samPreview)) samClear();
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
  if (tool !== "sam" && S.tool === "sam") samClear();   // drop a dangling prompt session
  S.tool = tool;
  $$("#tools .tbtn").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
  canvas.className = `tool-${tool}`;
  $("#gcppanel").hidden = tool !== "gcp";
  $("#sampanel").hidden = tool !== "sam";
  if (tool === "sam") updateSamStatus();
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
$("#undobtn").addEventListener("click", () => vectorToolActive() ? vecUndo() : undo());
$("#redobtn").addEventListener("click", () => vectorToolActive() ? vecRedo() : redo());
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
  gcpReset(false);   // GCPs belong to the previous photo
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
function isDirDrop(dt) {
  for (const it of dt.items || []) {
    const en = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (en && en.isDirectory) return true;
  }
  return false;
}

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  // With the batch panel open every drop is a batch drop, even if the
  // user misses the dashed zone.
  if (batchPanelOpen()) { stageDrop(e.dataTransfer, "batch"); return; }
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  // A coordinate file dropped while georeferencing loads GCP points, even
  // if the user misses the panel's dashed zone.
  if (S.tool === "gcp" && GCP_FILE_RE.test(f.name)) { gcpLoadFile(f); return; }
  // A folder (or multiple images) dropped while georeferencing is a
  // batch-GCP queue, even outside the dashed #gcpbatchdrop zone.
  if (S.tool === "gcp" && (e.dataTransfer.files.length > 1 || isDirDrop(e.dataTransfer))) {
    stageDrop(e.dataTransfer, "gcp");
    return;
  }
  openImage(f);
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

$("#sitename").addEventListener("input",
  () => localStorage.setItem("boneseg.site", $("#sitename").value));

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
      plate_site: $("#sitename").value,
    });
    const st = $("#exportstatus");
    st.innerHTML = `Exported <b>${res.files.length}</b> file(s) to <code>${res.target}</code> ` +
      `<a href="#" id="revealout">open folder</a>` +
      (res.master ? `<br>Master updated: <code>${res.master}</code>` : "") +
      (res.note ? `<br>⚠ ${res.note}` : "");
    if (res.note) toast(res.note, "error", 9000);
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
/* GCP georeferencing                                                   */
/* ------------------------------------------------------------------ */
let gcps = [];       // {px, py, e, n, id} — px/py in edit-image coords
let gcpQueue = [];   // parsed-but-unclicked points {id, e, n}
let gcpPoints = [];  // ALL parsed file/paste points — auto-match uses these,
                     // so click order (and extra unclicked points) don't matter
let gcpBatchFiles = [];   // [{name, path, georeferenced}] — batch-GCP queue
let gcpBatchIndex = -1;   // current index into gcpBatchFiles; -1 = no batch loaded

/* A multi-grave master file (one txt with every grave's points, grouped by
   a trailing code column) is filtered down to one grave by #gcpcodefilter
   — server-side auto-match caps at 40 points, and this is also just less
   error-prone than matching clicks against every grave on the site. */
function currentGcpPool() {
  const code = ($("#gcpcodefilter").value || "").trim().toLowerCase();
  if (!code) return gcpPoints;
  return gcpPoints.filter((p) => (p.code || "").toLowerCase().includes(code));
}

function updateCodeList() {
  const codes = [...new Set(gcpPoints.map((p) => p.code).filter(Boolean))].sort();
  const dl = $("#gcpcodelist");
  dl.textContent = "";
  for (const c of codes) {
    const opt = document.createElement("option");
    opt.value = c;
    dl.appendChild(opt);
  }
  $("#gcpcodefilterrow").hidden = codes.length === 0;
}

/* Point parsing lives on the SERVER (boneseg/data/gcp_parse.py) so txt,
   CSV and Excel all go through one parser; the UI just uploads. */
async function gcpLoadPoints(formData) {
  const r = await fetch("/api/gcp_points", { method: "POST", body: formData });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  const res = await r.json();
  gcpPoints = res.points;
  gcpQueue = res.points.slice();
  $("#gcpcodefilter").value = "";
  updateCodeList();
  gcpQueueInfo();
  if (res.lonlat_warning) {
    toast("These look like lat/lon DEGREES — georeferencing needs projected " +
          "metric coordinates (e.g. HTRS96/TM). Check the file and the EPSG.",
          "error", 12000);
  }
  applyCrsGuess(res.crs_guess);
  toast(`${res.n} point(s) loaded — click them on the photo in ANY order.`, "ok", 7000);
}

/* Pre-fills EPSG + swap from the server's guess_crs() heuristic — always a
   suggestion, never locked: both fields stay freely editable afterwards. */
function applyCrsGuess(guess) {
  const hint = $("#gcpcrsguess");
  hint.className = "hint";
  if (!guess || !guess.epsg) {
    hint.textContent = "";
    return;
  }
  $("#gcpepsg").value = String(guess.epsg);
  $("#gcpswap").checked = !!guess.swap;
  if (guess.confidence === "high") {
    hint.textContent = `🌍 Auto-detected: EPSG:${guess.epsg} (${guess.name})` +
      (guess.swap ? ", E/N swapped" : "") + " — change above if wrong.";
    hint.className = "hint ok";
  } else {
    const alts = guess.candidates
      .filter((c) => !(c.epsg === guess.epsg && c.swap === guess.swap))
      .map((c) => `EPSG:${c.epsg} (${c.name}${c.swap ? ", swapped" : ""})`)
      .join(" or ");
    hint.textContent = `⚠ Ambiguous — guessed EPSG:${guess.epsg} (${guess.name})` +
      (guess.swap ? ", swapped" : "") +
      (alts ? `, but could be ${alts}` : "") +
      ". Verify against a known point/GPS photo.";
    hint.className = "hint bad";
  }
}

async function gcpLoadFile(file) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  try { await gcpLoadPoints(fd); } catch (err) {
    toast(`Could not read point file: ${err.message}`, "error", 9000);
  }
}

const GCP_FILE_RE = /\.(txt|csv|tsv|dat|asc|xlsx|xlsm)$/i;
$("#gcpbrowse").addEventListener("click", (e) => { e.preventDefault(); $("#gcpfile").click(); });
$("#gcpdrop").addEventListener("click", () => $("#gcpfile").click());
$("#gcpfile").addEventListener("change", (e) => {
  if (e.target.files[0]) gcpLoadFile(e.target.files[0]);
  e.target.value = "";
});
$("#gcpdrop").addEventListener("dragover", (e) => {
  e.preventDefault(); e.stopPropagation(); $("#gcpdrop").classList.add("drag");
});
$("#gcpdrop").addEventListener("dragleave", () => $("#gcpdrop").classList.remove("drag"));
$("#gcpdrop").addEventListener("drop", (e) => {
  e.preventDefault(); e.stopPropagation();
  $("#gcpdrop").classList.remove("drag");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) gcpLoadFile(f);
});

function gcpQueueInfo() {
  const pool = currentGcpPool();
  const auto = $("#gcpauto").checked && pool.length >= 3;
  $("#gcpqueue").textContent = gcpQueue.length
    ? (auto
        ? `${pool.length} point(s) in pool — click each on the photo, any order (auto-match on Apply).`
        : `${gcpQueue.length} point(s) left — next click places “${gcpQueue[0].id || gcpQueue[0].e}”.`)
    : "";
}
$("#gcpauto").addEventListener("change", gcpQueueInfo);
$("#gcpcodefilter").addEventListener("input", () => {
  gcpQueue = currentGcpPool().slice();
  gcpQueueInfo();
});

function gcpAddPoint(x, y) {
  const q = gcpQueue.shift();
  gcps.push({ px: x, py: y, e: q ? q.e : "", n: q ? q.n : "", id: q ? q.id : "" });
  gcpQueueInfo();
  renderGcpTable();
  requestDraw();
}

function renderGcpTable() {
  if (!gcps.length) { $("#gcptablewrap").innerHTML = ""; return; }
  const rows = gcps.map((g, i) =>
    `<tr><td>${i + 1}</td>` +
    `<td class="gcpid" title="${g.id || ""}">${g.id || ""}</td>` +
    `<td><input data-i="${i}" data-f="e" value="${g.e}" spellcheck="false"></td>` +
    `<td><input data-i="${i}" data-f="n" value="${g.n}" spellcheck="false"></td>` +
    `<td><button class="gcpdel" data-i="${i}" title="Remove">✕</button></td></tr>`).join("");
  $("#gcptablewrap").innerHTML =
    `<table><tr><th>#</th><th>ID</th><th>E (easting)</th><th>N (northing)</th><th></th></tr>${rows}</table>`;
}

$("#gcptablewrap").addEventListener("input", (e) => {
  const t = e.target;
  if (t.dataset.i !== undefined && t.dataset.f) gcps[+t.dataset.i][t.dataset.f] = t.value;
});
$("#gcptablewrap").addEventListener("click", (e) => {
  const b = e.target.closest("button.gcpdel");
  if (!b) return;
  gcps.splice(+b.dataset.i, 1);
  renderGcpTable();
  requestDraw();
});

$("#gcpparse").addEventListener("click", async () => {
  const text = $("#gcppaste").value;
  if (!text.trim()) { toast("Paste the points first (ID E N per line).", "error", 6000); return; }
  const fd = new FormData();
  fd.append("text", text);
  try { await gcpLoadPoints(fd); } catch (err) {
    toast(`No points recognized: ${err.message}`, "error", 8000);
  }
});

function gcpReset(alsoQueue = true) {
  gcps = [];
  $("#gcpcodefilter").value = "";  // each new photo/grave starts unfiltered
  if (alsoQueue) {
    gcpQueue = []; gcpPoints = []; $("#gcppaste").value = "";
    $("#gcpcrsguess").textContent = ""; $("#gcpcrsguess").className = "hint";
    updateCodeList();
  } else { gcpQueue = currentGcpPool().slice(); }  // same master file, next photo
  gcpQueueInfo();
  renderGcpTable();
  $("#gcpstatus").textContent = "";
  $("#gcpstatus").className = "hint";
  requestDraw();
}
$("#gcpreset").addEventListener("click", () => gcpReset(true));

$("#gcpapply").addEventListener("click", async () => {
  if (gcps.length < 3) { toast("Place at least 3 points on the photo first.", "error", 6000); return; }
  // Auto-match: with a loaded point list the pairing is found geometrically
  // on the server, so neither the click order nor the file order matters.
  // The grave-code filter (multi-grave master files) narrows this pool —
  // auto-matching caps at 40 points server-side.
  const pool = currentGcpPool();
  const auto = $("#gcpauto").checked && pool.length >= 3;
  let payload;
  if (auto) {
    if (pool.length < gcps.length) {
      toast(`You clicked ${gcps.length} spots but the point pool has only ` +
            `${pool.length} — remove extra clicks (✕) or narrow the grave-code filter.`, "error", 8000);
      return;
    }
    payload = {
      auto: true,
      clicks: gcps.map((g) => ({ px: g.px, py: g.py })),
      points: pool.map((p) => ({ id: p.id, e: p.e, n: p.n })),
    };
  } else {
    for (const g of gcps) {
      if (String(g.e).trim() === "" || String(g.n).trim() === "" ||
          Number.isNaN(parseFloat(g.e)) || Number.isNaN(parseFloat(g.n))) {
        toast("Every point needs numeric E and N coordinates (or load a point file and enable auto-match).", "error", 7000);
        return;
      }
    }
    payload = {
      gcps: gcps.map((g) => ({ px: g.px, py: g.py, e: parseFloat(g.e), n: parseFloat(g.n) })),
    };
  }
  payload.epsg = $("#gcpepsg").value.trim();
  payload.swap = $("#gcpswap").checked;
  setBusy(true);
  try {
    const res = await apiPost("/api/georef", payload);
    if (res.matched) {
      // Show which surveyed point each click received.
      res.matched.forEach((m, i) => {
        if (gcps[i]) { gcps[i].id = m.id; gcps[i].e = m.e; gcps[i].n = m.n; }
      });
      renderGcpTable();
    }
    applySummary(res);
    updateStats();
    const st = $("#gcpstatus");
    const worst = Math.max(...res.residuals_m);
    st.className = (worst > 0.15 || res.ambiguous || res.region_warning || res.region_corrected) ? "hint bad" : "hint ok";
    st.innerHTML =
      (res.region_warning ? `⚠⚠ ${res.region_warning}<br>` : "") +
      (res.region_corrected ? "<br>⚠ The auto-match's click-orientation guess conflicted with the " +
                              "surveyed CRS (near-symmetric point layout) — E/N were flipped back to " +
                              "match the CRS. Verify in QGIS." : "") +
      `Applied (<b>${res.mode === "homography" ? "perspective" : "affine"}</b>) — ` +
      `RMS <b>${(res.rms_m * 100).toFixed(1)} cm</b>, per point: ` +
      res.residuals_m.map((r, i) => `#${i + 1} ${(r * 100).toFixed(1)}`).join(", ") + " cm." +
      (res.matched ? "<br>Points auto-matched — table shows the pairing." : "") +
      (res.auto_swapped ? "<br>⚠ E/N were SWAPPED in your points — corrected automatically (geodetic Y/X)." : "") +
      (res.ambiguous ? "<br>⚠ Symmetric point layout — pairing resolved by least distortion + click order. " +
                       "To be safe: click the points in file order, or click one extra (5th) point. Verify in QGIS." : "") +
      (res.mode === "homography"
        ? "<br>Oblique photo detected — perspective corrected. Vectors are exact; " +
          "for the photo in GIS/CAD use the <b>GeoTIFF photo</b> / DXF export (rectified, no stretching). " +
          (gcps.length === 4 ? "⚠ 4 points = no redundancy, residuals read 0 — a 5th point gives a real check. " : "")
        : "") +
      (res.mode !== "homography" && worst > 0.15
        ? "<br>⚠ Large residuals — oblique photo (add a 4th/5th point for perspective correction), or a mistyped digit?"
        : "") +
      (res.world_files.length
        ? `<br>World file written (${res.world_files.join(", ")}) — the photo now opens georeferenced in QGIS.`
        : "");
    if (res.region_warning) {
      toast(`⚠ ${res.region_warning}`, "error", 15000);
    } else if (res.region_corrected) {
      toast("⚠ Auto-match's orientation guess conflicted with the CRS — E/N flipped back automatically. Verify in QGIS.", "error", 12000);
    } else {
      toast(`Georeferenced — RMS ${(res.rms_m * 100).toFixed(1)} cm. Exports are now in real coordinates.`, "ok", 8000);
    }
    // Batch-GCP queue: a clean apply (no region_warning — a bad fit needs
    // the user's eyes before moving on) advances to the next photo.
    if (gcpBatchIndex >= 0 && !res.region_warning) {
      gcpBatchFiles[gcpBatchIndex].georeferenced = true;
      if (gcpBatchIndex < gcpBatchFiles.length - 1) {
        gcpBatchIndex++;
        await gcpBatchOpenCurrent();
      } else {
        toast("Batch georeferencing done — that was the last image in the folder.", "ok", 9000);
        gcpBatchStatus();
      }
    }
  } catch (err) {
    toast(`Georeferencing failed: ${err.message}`, "error", 9000);
  } finally {
    setBusy(false);
  }
});

/* Batch-GCP: click-through-all georeferencing of a whole folder. A master
   points file (all graves, filtered per-photo via #gcpcodefilter) stays
   loaded across the queue; each Apply above advances automatically. */
function gcpBatchStatus() {
  const el = $("#gcpbatchqueue");
  if (!gcpBatchFiles.length || gcpBatchIndex < 0) { el.textContent = ""; return; }
  const item = gcpBatchFiles[gcpBatchIndex];
  const done = gcpBatchFiles.filter((f) => f.georeferenced).length;
  el.textContent = `${gcpBatchIndex + 1}/${gcpBatchFiles.length} — ${item.name}` +
    (item.georeferenced ? " ✓ already georeferenced" : "") +
    ` (${done}/${gcpBatchFiles.length} done)`;
}

async function gcpBatchOpenCurrent() {
  if (gcpBatchIndex < 0 || gcpBatchIndex >= gcpBatchFiles.length) return;
  await openImage(gcpBatchFiles[gcpBatchIndex].path);
  gcpReset(false);  // keeps the loaded master points, clears clicks + code filter
  gcpBatchStatus();
}

async function gcpBatchLoadFolder() {
  const dir = $("#gcpbatchdir").value.trim();
  if (!dir) { toast("Enter a folder path first.", "error", 5000); return; }
  try {
    const res = await apiPost("/api/list_images", { dir });
    if (!res.n) { toast("No supported images found in that folder.", "error", 7000); return; }
    gcpBatchFiles = res.files;
    gcpBatchIndex = gcpBatchFiles.findIndex((f) => !f.georeferenced);
    if (gcpBatchIndex === -1) gcpBatchIndex = 0;
    $("#gcpbatchnav").hidden = false;
    await gcpBatchOpenCurrent();
    const done = gcpBatchFiles.filter((f) => f.georeferenced).length;
    toast(`${res.n} image(s) queued — ${done} already georeferenced, starting at #${gcpBatchIndex + 1}.`, "ok", 7000);
  } catch (err) {
    toast(`Could not list folder: ${err.message}`, "error", 8000);
  }
}
$("#gcpbatchload").addEventListener("click", gcpBatchLoadFolder);

/* Jump straight from a finished batch-inference run to georeferencing the
   SAME folder — no retyping the path. Shown once a batch completes with at
   least one successful image (see renderBatchTable). */
$("#gcpbatchlink").addEventListener("click", async () => {
  const dir = $("#batchin").value.trim();
  setTool("gcp");
  if (batchPanelOpen()) $("#batchtoggle").click();
  $("#gcpbatchdir").value = dir;
  await gcpBatchLoadFolder();
});
$("#gcpbatchprev").addEventListener("click", () => {
  if (gcpBatchIndex > 0) { gcpBatchIndex--; gcpBatchOpenCurrent(); }
});
$("#gcpbatchskip").addEventListener("click", () => {
  if (gcpBatchIndex < gcpBatchFiles.length - 1) { gcpBatchIndex++; gcpBatchOpenCurrent(); }
  else toast("That's the last image in the folder.", "", 5000);
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
      plate_site: $("#sitename").value,
    });
    watchJob(async (job) => {
      $("#batchsummary").textContent = job.message;
      renderBatchTable(job.rows || []);
      $("#gcpbatchlink").hidden = !(job.rows || []).some((r) => r.status === "ok");
    });
  } catch (err) {
    toast(err.message, "error", 8000);
  }
});

/* Drag & drop a folder (or a pile of images) onto the batch panel: the
   files are staged server-side and the input-folder field fills itself. */
const IMG_EXTS = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff"]);

function batchPanelOpen() { return !$("#batchpanel").hidden; }

/* Shared by both drop zones: the batch-inference panel's #batchdrop, and
   the batch-GCP queue's #gcpbatchdrop. Browsers never expose a dropped
   folder's real filesystem path (security), so either way the files are
   uploaded and staged server-side — /api/list_images / /api/batch then
   just point at that staging folder like any other. */
async function stageDrop(dt, target = "batch") {
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
  if (target === "batch") $("#batchsummary").textContent = `Uploading ${wanted.length} image(s)…`;
  try {
    const res = await apiForm("/api/batch_upload", fd);
    if (target === "gcp") {
      $("#gcpbatchdir").value = res.dir;
      toast(`${res.count} image(s) staged — loading queue…`, "ok");
      await gcpBatchLoadFolder();
    } else {
      $("#batchin").value = res.dir;
      $("#gcpbatchlink").hidden = true;  // stale link from a previous run
      $("#batchsummary").textContent =
        `${res.count} image(s) staged — press ▶ Process folder.` +
        (res.skipped.length ? ` ${res.skipped.length} unsupported file(s) skipped.` : "");
      renderBatchTable([]);
      toast(`${res.count} image(s) ready for batch.`, "ok");
    }
  } catch (err) {
    if (target === "batch") $("#batchsummary").textContent = "";
    toast(`Staging failed: ${err.message}`, "error", 8000);
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
    stageDrop(e.dataTransfer, "batch");
  });
}

{
  const zone = $("#gcpbatchdrop");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation();
    zone.classList.remove("drag");
    stageDrop(e.dataTransfer, "gcp");
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
    // Switch back to the editor BEFORE loadPhoto() — it calls fitView(),
    // which measures the canvas element's rendered size; while the batch
    // panel is showing, #canvaswrap is display:none and fitView() computes
    // a 0% zoom against that zero-size box (blank canvas until a manual
    // Fit/F — looked like the click did nothing).
    if (batchPanelOpen()) $("#batchtoggle").click();
    applySummary(sum);
    await loadPhoto();
    if (sum.has_result) await loadMaskAndSkeleton();
    updateStats();
    // Re-exports should replace this item's batch artifacts in place.
    if (sum.mask_loaded) $("#outdir").value = row.out_dir;
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
  $("#sitename").value = localStorage.getItem("boneseg.site") || "";

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
