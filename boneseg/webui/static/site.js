/* Site view — every georeferenced grave of one site on one map.
 *
 * The user picks the graves (photos, export folders/files, whole folders —
 * through the normal Windows dialogs, served by /api/dialog) and can add the
 * grave open in the editor to check a fresh georeference on the map. The
 * server keeps no site state: /api/site/scan returns everything the map
 * needs, photo tiles come per grave from /api/site/photo, and the export
 * (DXF / GeoPackage / PDF) gets the grave list back.
 */
(() => {
  const LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/";
  const COLORS = ["#f97316", "#22d3ee", "#a3e635", "#f472b6", "#facc15", "#60a5fa",
                  "#34d399", "#c084fc", "#fb7185", "#2dd4bf", "#fbbf24", "#818cf8"];
  const BASEMAPS = {
    none: null,
    esri: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      opt: { maxNativeZoom: 18, attribution: "Imagery © Esri, Maxar, Earthstar Geographics, GIS User Community" },
    },
    osm: {
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      opt: { maxNativeZoom: 19, attribution: "© OpenStreetMap contributors" },
    },
  };

  const store = {
    get(k, d) { try { const v = localStorage.getItem("site." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("site." + k, v); } catch { /* private mode */ } },
  };

  let leafletReady = null;
  let map = null, mapMode = null, baseLayer = null;
  let site = null;           // last scan result
  let items = [];            // per grave: {g, color, visible, photoOn, layers...}
  let sources = [];          // picked files / folders
  try { sources = JSON.parse(store.get("sources", "[]")) || []; } catch { sources = []; }
  let scanSeq = 0;

  const panel = $("#sitepanel");
  const isOpen = () => !panel.hidden;
  const editorGeoref = () => !!(S.image && S.image.georef);

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function loadLeaflet() {
    if (window.L) return Promise.resolve();
    if (leafletReady) return leafletReady;
    leafletReady = new Promise((resolve, reject) => {
      const css = document.createElement("link");
      css.rel = "stylesheet"; css.href = LEAFLET + "leaflet.css";
      document.head.appendChild(css);
      const js = document.createElement("script");
      js.src = LEAFLET + "leaflet.js";
      js.onload = () => resolve();
      js.onerror = () => { leafletReady = null; reject(new Error("map library could not be loaded (no internet?)")); };
      document.head.appendChild(js);
    });
    return leafletReady;
  }

  /* ---------------- open / close ---------------- */
  function showPanel() {
    if (batchPanelOpen()) $("#batchtoggle").click();
    panel.hidden = false;
    $("#canvaswrap").style.display = "none";
    $("#sidebar").hidden = true;  // segmentation controls are idle here — give the map the room
    $("#sitetoggle").classList.add("active");
    if (!$("#sitevec").value) $("#sitevec").value = $("#outdir").value;
    if (!$("#sitetitle").value) $("#sitetitle").value = $("#sitename").value;
  }
  function openSite() {
    showPanel();
    loadLeaflet().then(() => {
      if (map) map.invalidateSize();
      // Refresh: the editor grave may have new GCPs or new edits since.
      if (sources.length || ($("#sitecurrent").checked && editorGeoref())) loadSite();
      else $("#sitesummary").textContent = "Add graves: photos, export folders or exported files.";
    }).catch((e) => toast(`Site view: ${e.message}`, "error", 9000));
  }
  function closeSite() {
    panel.hidden = true;
    $("#sidebar").hidden = false;
    $("#sitetoggle").classList.remove("active");
    if ($("#batchpanel").hidden) { $("#canvaswrap").style.display = ""; requestDraw(); }
  }
  $("#sitetoggle").addEventListener("click", () => (isOpen() ? closeSite() : openSite()));
  // Opening Batch while the site view is up: hand the main area over.
  $("#batchtoggle").addEventListener("click", () => {
    if (isOpen() && !$("#batchpanel").hidden) {
      panel.hidden = true;
      $("#sidebar").hidden = false;
      $("#sitetoggle").classList.remove("active");
    }
  });

  /* "Check on map" from the GCP panel: open the site view on the editor grave. */
  window.siteShowCurrent = async () => {
    if (!editorGeoref()) { toast("Georeference the photo first (Apply georeference).", "error"); return; }
    $("#sitecurrent").checked = true;
    showPanel();
    try { await loadLeaflet(); } catch (e) { toast(`Site view: ${e.message}`, "error", 9000); return; }
    if (map) map.invalidateSize();
    await loadSite({ focusCurrent: true });
  };

  /* ---------------- dialogs + sources ---------------- */
  async function dialog(kind, title, initial, filetypes) {
    try {
      const r = await apiPost("/api/dialog", { kind, title, initial: initial || "", filetypes });
      return r.paths || [];
    } catch (err) {
      toast(`Dialog: ${err.message}`, "error");
      return [];
    }
  }
  function lastDir() {
    const s = sources[sources.length - 1];
    return s ? s.replace(/[\\/][^\\/]*$/, "") : ($("#sitevec").value || "");
  }
  function addSources(paths) {
    let n = 0;
    for (const p of paths) if (p && !sources.includes(p)) { sources.push(p); n++; }
    if (n) { saveSources(); loadSite({ fit: true }); }
  }
  function saveSources() { store.set("sources", JSON.stringify(sources)); renderSources(); }
  function renderSources() {
    $("#sitesources").innerHTML = sources.map((s, i) => {
      const short = s.split(/[\\/]/).slice(-2).join("\\");
      return `<div class="sitesrc" title="${escapeHtml(s)}"><span>${escapeHtml(short)}</span>` +
             `<button class="tbtn mini" data-rm="${i}" title="Remove">✕</button></div>`;
    }).join("");
    $("#siteclear").hidden = !sources.length;
  }
  $("#sitesources").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-rm]");
    if (!b) return;
    sources.splice(+b.dataset.rm, 1);
    saveSources();
    loadSite();
  });
  $("#siteclear").addEventListener("click", () => { sources = []; saveSources(); loadSite(); });
  $("#siteaddfiles").addEventListener("click", async () => {
    addSources(await dialog("files", "Pick the graves: photos and/or exported files (Ctrl / Shift for many)", lastDir(), "graves"));
  });
  $("#siteadddir").addEventListener("click", async () => {
    addSources(await dialog("folder", "Folder with grave photos or exports" + FOLDER_HINT, lastDir()));
  });
  $("#siteaddpath").addEventListener("click", () => {
    const v = $("#sitepath").value.trim().replace(/^"|"$/g, "");
    if (v) { $("#sitepath").value = ""; addSources([v]); }
  });
  $("#sitepath").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#siteaddpath").click(); });
  // Windows folder pickers list folders only — an export folder full of
  // files looks "empty". Say so, or it reads as broken.
  const FOLDER_HINT = " — files are not listed here: open the folder and click Select Folder";
  $("#sitevecbrowse").addEventListener("click", async () => {
    const p = await dialog("folder", "Export folder (the one with the per-grave folders)" + FOLDER_HINT,
                           $("#sitevec").value);
    if (p[0]) { $("#sitevec").value = p[0]; loadSite(); }
  });
  $("#sitevec").addEventListener("change", () => loadSite());
  $("#sitecurrent").addEventListener("change", () => loadSite());
  $("#siterecursive").addEventListener("change", () => loadSite());

  /* ---------------- map ---------------- */
  function ensureMap(mode) {
    if (map && mapMode === mode) return false;
    if (map) { map.remove(); map = null; baseLayer = null; }
    mapMode = mode;
    const opts = { preferCanvas: true, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90 };
    map = mode === "map"
      ? L.map("sitemap", { ...opts, maxZoom: 26 })
      : L.map("sitemap", { ...opts, crs: L.CRS.Simple, minZoom: -6, maxZoom: 14 });
    L.control.scale({ imperial: false, maxWidth: 160 }).addTo(map);
    setBasemap($("#sitebase").value);
    return true;
  }

  function setBasemap(key) {
    store.set("base", key);
    if (!map) return;
    if (baseLayer) { map.removeLayer(baseLayer); baseLayer = null; }
    const b = BASEMAPS[key];
    $("#sitebasehint").hidden = !(b && mapMode !== "map");
    if (!b || mapMode !== "map") return;
    baseLayer = L.tileLayer(b.url, { ...b.opt, maxZoom: 26 }).addTo(map);
    baseLayer.bringToBack();
  }

  function graveSpec(g) {
    return { name: g.name, stem: g.stem, photo: g.photo, photo_kind: g.photo_kind,
             photo_affine: g.photo_affine, vec: g.vec, crs: g.crs, current: g.current,
             edited_lines: g.edited_lines };
  }

  async function loadPhoto(it) {
    if (it.photo || it.photoLoading) return;
    it.photoLoading = true;
    try {
      const r = await fetch("/api/site/photo", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grave: graveSpec(it.g), mode: site.mode, origin: site.origin }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      const blob = await r.blob();
      if (it.dead) return;
      it.photoUrl = URL.createObjectURL(blob);
      it.photo = L.imageOverlay(it.photoUrl, it.g.bounds,
                                { opacity: +$("#siteopacity").value, interactive: false });
      refreshItem(it);
    } catch (err) {
      it.g.notes.push(`photo: ${err.message}`);
    } finally {
      it.photoLoading = false;
    }
  }

  function buildLayers(it) {
    const g = it.g;
    if (!g.georeferenced) return;
    const c = it.color;
    if ((g.rings || []).length) it.rings = L.polyline(g.rings, { color: c, weight: 1.6, opacity: 0.95, interactive: false });
    if ((g.lines || []).length) it.lines = L.polyline(g.lines, { color: c, weight: 1.1, opacity: 0.9, interactive: false });
    const shape = g.footprint || hullOf(g);
    if (shape) {
      it.frame = L.polygon(shape, { color: c, weight: g.current ? 2.5 : 1, opacity: 0.85,
                                    dashArray: g.current ? null : "4 4", fill: true, fillOpacity: 0 });
      it.frame.bindPopup(() => popupHtml(it), { maxWidth: 320 });
      it.label = L.tooltip({ permanent: true, direction: "center",
                             className: "sitelabel" + (g.current ? " current" : "") })
        .setLatLng(L.polygon(shape).getBounds().getCenter()).setContent(escapeHtml(g.name));
    }
  }

  function dropLayers(it) {
    it.dead = true;
    for (const l of [it.photo, it.rings, it.lines, it.frame, it.label]) if (l && map) map.removeLayer(l);
    if (it.photoUrl) URL.revokeObjectURL(it.photoUrl);
  }

  function hullOf(g) {
    const pts = [...(g.rings || []), ...(g.lines || [])].flat();
    if (!pts.length) return null;
    let a = Infinity, b = Infinity, cc = -Infinity, d = -Infinity;
    for (const [la, ln] of pts) { a = Math.min(a, la); b = Math.min(b, ln); cc = Math.max(cc, la); d = Math.max(d, ln); }
    return [[a, b], [a, d], [cc, d], [cc, b]];
  }

  function refreshItem(it) {
    if (!map) return;
    const on = it.visible && it.g.georeferenced;
    const toggle = (layer, want) => {
      if (!layer) return;
      if (want && !map.hasLayer(layer)) layer.addTo(map);
      else if (!want && map.hasLayer(layer)) map.removeLayer(layer);
    };
    const wantPhoto = on && $("#sitephotos").checked && it.photoOn && !!it.g.bounds;
    if (wantPhoto && !it.photo) loadPhoto(it);
    toggle(it.photo, wantPhoto);
    toggle(it.frame, on);
    toggle(it.label, on && $("#sitelabels").checked);
    toggle(it.rings, on && $("#siteoutlines").checked);
    toggle(it.lines, on && ($("#sitecenter").checked || !!it.g.edited_lines));
    if (it.photo && map.hasLayer(it.photo)) it.photo.bringToBack();
    if (baseLayer) baseLayer.bringToBack();
  }
  const refreshAll = () => items.forEach(refreshItem);

  function zoomTo(list) {
    let b = null;
    for (const it of list) {
      if (!it.g.georeferenced || !it.frame) continue;
      const bb = it.frame.getBounds();
      b = b ? b.extend(bb) : L.latLngBounds(bb.getSouthWest(), bb.getNorthEast());
    }
    if (b) map.fitBounds(b.pad(0.08), { maxZoom: mapMode === "map" ? 24 : 9 });
  }

  /* ---------------- grave list ---------------- */
  const canOpen = (g) => g.photo && g.photo_kind === "file" && !g.current;

  function popupHtml(it) {
    const g = it.g;
    const notes = g.notes.length ? `<ul>${g.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : "";
    return `<b>${escapeHtml(g.name)}</b><br>
      <span class="hint">${escapeHtml(g.crs || "no CRS")} · ${g.n_lines} bone lines` +
      `${g.dist_m != null ? ` · ${g.dist_m} m from site centre` : ""}</span>${notes}
      ${canOpen(g) ? `<div class="btnrow"><button class="btn sm primary" data-open="${items.indexOf(it)}">✎ Open in editor</button></div>` : ""}
      ${g.current ? `<div class="btnrow"><button class="btn sm primary" data-back="1">✎ Back to the editor</button></div>` : ""}`;
  }

  function renderList() {
    $("#sitelist").innerHTML = items.map((it, i) => {
      const g = it.g;
      const warn = g.notes.length ? `<span class="sitewarn" title="${escapeHtml(g.notes.join("\n"))}">⚠</span>` : "";
      const cls = (g.georeferenced ? "" : " off") + (g.current ? " cur" : "");
      return `<div class="siterow${cls}" data-i="${i}">
        <input type="checkbox" data-act="vis" ${it.visible ? "checked" : ""} ${g.georeferenced ? "" : "disabled"}>
        <span class="swatch" style="background:${it.color}"></span>
        <a href="#" data-act="zoom" title="${g.georeferenced ? "Zoom to grave" : escapeHtml(g.notes[0] || "not georeferenced")}">${escapeHtml(g.name)}</a>
        ${warn}
        <span class="siteicons">
          ${g.bounds ? `<button class="tbtn mini${it.photoOn ? " active" : ""}" data-act="photo" title="Show / hide this photo">📷</button>` : ""}
          ${canOpen(g) ? `<button class="tbtn mini" data-act="open" title="Open in editor">✎</button>` : ""}
        </span>
      </div>`;
    }).join("");
  }

  $("#sitelist").addEventListener("click", (e) => {
    const row = e.target.closest(".siterow");
    const act = e.target.dataset.act;
    if (!row || !act) return;
    const it = items[+row.dataset.i];
    if (act === "zoom") {
      e.preventDefault();
      if (it.g.georeferenced) { zoomTo([it]); if (it.frame) it.frame.openPopup(); }
      else toast(`${it.g.name}: ${it.g.notes[0] || "not georeferenced"}`, "", 7000);
    } else if (act === "photo") {
      it.photoOn = !it.photoOn;
      e.target.classList.toggle("active", it.photoOn);
      refreshItem(it);
    } else if (act === "open") {
      openInEditor(it);
    }
  });
  $("#sitelist").addEventListener("change", (e) => {
    const row = e.target.closest(".siterow");
    if (!row || e.target.dataset.act !== "vis") return;
    const it = items[+row.dataset.i];
    it.visible = e.target.checked;
    refreshItem(it);
  });
  $("#sitemap").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-open]");
    if (b) openInEditor(items[+b.dataset.open]);
    if (e.target.closest("button[data-back]")) closeSite();
  });

  async function openInEditor(it) {
    const g = it.g;
    if (!g.photo) return;
    closeSite();
    await openBatchItem({ path: g.photo, out_dir: g.vec ? g.vec.dir : "", file: g.name });
  }

  /* ---------------- load ---------------- */
  async function loadSite(opt = {}) {
    const wantCurrent = $("#sitecurrent").checked && editorGeoref();
    store.set("vecdir", $("#sitevec").value.trim());
    if (!sources.length && !wantCurrent) {
      for (const it of items) dropLayers(it);
      items = []; site = null;
      renderList();
      $("#sitesummary").textContent = "Add graves: photos, export folders or exported files.";
      return;
    }
    const seq = ++scanSeq;
    $("#sitesummary").textContent = "Reading the graves…";
    try {
      await loadLeaflet();
      const res = await apiPost("/api/site/scan", {
        sources, vec_dir: $("#sitevec").value.trim(), recursive: $("#siterecursive").checked,
        current: wantCurrent,
      });
      if (seq !== scanSeq) return;  // a newer scan superseded this one
      const fresh = ensureMap(res.mode);
      // Keep per-grave choices (visible / photo) across rescans.
      const prev = new Map(items.map((it) => [it.g.name, it]));
      for (const it of items) dropLayers(it);
      site = res;
      items = res.graves.map((g, i) => {
        const old = prev.get(g.name);
        return { g, color: g.current ? "#ff3b3b" : COLORS[i % COLORS.length],
                 visible: g.georeferenced && (old ? old.visible : true),
                 photoOn: old ? old.photoOn : true };
      });
      items.forEach(buildLayers);
      refreshAll();
      renderList();
      const placed = items.filter((it) => it.g.georeferenced);
      const warn = items.filter((it) => it.g.notes.length).length;
      $("#sitesummary").innerHTML =
        `<b>${placed.length}</b> of ${items.length} graves placed` +
        (res.crs ? ` · ${escapeHtml(res.crs)}` : " · local coordinates (no EPSG — no basemap)") +
        (warn ? ` · <span class="sitewarn">⚠ ${warn} to check</span>` : "") +
        (res.problems.length ? `<br><span class="sitewarn">${res.problems.map(escapeHtml).join("<br>")}</span>` : "");
      if (!$("#siteout").value) {
        const base = sources.length ? lastDir() : $("#sitevec").value;
        if (base) $("#siteout").value = base.replace(/[\\/]+$/, "") + "\\site_plan";
      }
      const cur = items.find((it) => it.g.current);
      if (opt.focusCurrent && cur) { zoomTo([cur]); if (cur.frame) cur.frame.openPopup(); }
      else if ((opt.fit || fresh) && placed.length) {
        // Strays (km away: wrong points / EPSG) would zoom out to a continent.
        const near = placed.filter((it) => it.g.dist_m == null || it.g.dist_m <= 2000);
        zoomTo(near.length ? near : placed);
      }
      if (opt.focusCurrent && cur && placed.length === 1) {
        toast("Only the editor grave is on the map — add the other graves of the site to compare.", "", 7000);
      }
    } catch (err) {
      $("#sitesummary").textContent = "";
      toast(`Site view: ${err.message}`, "error", 9000);
    }
  }

  /* ---------------- display controls ---------------- */
  $("#sitebase").addEventListener("change", (e) => setBasemap(e.target.value));
  for (const id of ["#sitephotos", "#siteoutlines", "#sitecenter", "#sitelabels"]) {
    $(id).addEventListener("change", () => {
      const key = id.slice(1) + (id === "#siteoutlines" || id === "#sitecenter" ? "_v2" : "");
      store.set(key, $(id).checked ? "1" : "0");
      refreshAll();
    });
  }
  $("#siteopacity").addEventListener("input", (e) => {
    for (const it of items) if (it.photo) it.photo.setOpacity(+e.target.value);
  });
  $("#sitefit").addEventListener("click", () => {
    const vis = items.filter((it) => it.visible);
    const near = vis.filter((it) => it.g.dist_m == null || it.g.dist_m <= 2000);
    zoomTo(near.length ? near : vis);
  });
  $("#siteall").addEventListener("click", () => {
    const anyOff = items.some((it) => it.g.georeferenced && !it.visible);
    items.forEach((it) => { if (it.g.georeferenced) it.visible = anyOff; });
    refreshAll(); renderList();
  });

  /* ---------------- export ---------------- */
  $("#siteoutbrowse").addEventListener("click", async () => {
    const p = await dialog("save", "Save the site plan as (the extension is added per format)",
                           $("#siteout").value || lastDir(), "site");
    if (p[0]) $("#siteout").value = p[0].replace(/\.(dxf|gpkg|pdf)$/i, "");
  });
  $("#siteexport").addEventListener("click", async () => {
    const chosen = items.filter((it) => it.visible && it.g.georeferenced).map((it) => graveSpec(it.g));
    if (!chosen.length) { toast("No visible georeferenced graves to export.", "error"); return; }
    const formats = [...document.querySelectorAll("#siteformats input:checked")].map((c) => c.value);
    if (!formats.length) { toast("Tick at least one format (CAD / GIS / PDF).", "error"); return; }
    const out = $("#siteout").value.trim();
    if (!out) { toast("Choose where to save the site plan.", "error"); return; }
    $("#siteexportstatus").textContent = `Writing ${chosen.length} graves…`;
    $("#siteexport").disabled = true;
    try {
      const r = await apiPost("/api/site/export", {
        graves: chosen, out_base: out, crs: site && site.crs, formats,
        photos: $("#siteexportphotos").checked && $("#sitephotos").checked,
        outlines: $("#siteoutlines").checked, centerlines: $("#sitecenter").checked,
        labels: $("#sitelabels").checked, title: $("#sitetitle").value,
      });
      const names = r.files.filter((f) => !/_photos[\\/]/.test(f)).map((f) => f.split(/[\\/]/).pop());
      $("#siteexportstatus").innerHTML =
        `Saved ${r.graves} graves: <code>${names.map(escapeHtml).join("</code>, <code>")}</code> ` +
        `<a href="#" id="sitereveal">open folder</a>`;
      $("#siteexportstatus").dataset.folder = r.folder;
      toast("Site plan saved.", "ok");
    } catch (err) {
      $("#siteexportstatus").textContent = "";
      toast(`Site export failed: ${err.message}`, "error", 9000);
    } finally {
      $("#siteexport").disabled = false;
    }
  });
  $("#siteexportstatus").addEventListener("click", (e) => {
    if (e.target.id !== "sitereveal") return;
    e.preventDefault();
    apiPost("/api/reveal", { path: $("#siteexportstatus").dataset.folder })
      .catch((err) => toast(err.message, "error"));
  });

  /* ---------------- restore ---------------- */
  $("#sitevec").value = store.get("vecdir", "");
  $("#sitebase").value = store.get("base", "esri");
  // v2 keys: the old defaults (outlines on, centerlines off) were wrong — the
  // centerlines ARE the drawing — so stored v1 choices must not carry over.
  for (const [id, def] of [["sitephotos", "1"], ["siteoutlines_v2", "0"], ["sitecenter_v2", "1"], ["sitelabels", "1"]]) {
    $("#" + id.replace("_v2", "")).checked = store.get(id, def) === "1";
  }
  renderSources();
})();
