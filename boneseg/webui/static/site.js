/* Site view — every georeferenced grave of one site on one map.
 *
 * Loads the photos of a folder (GCP sidecars / GeoTIFFs) plus the vectors
 * already exported for them, and stacks them in true position over an
 * optional Esri / OSM basemap (Leaflet, loaded on first use). The server
 * keeps no state: /api/site/scan returns everything the map needs, photos
 * come warped per grave from /api/site/photo, and the site DXF export gets
 * the grave list back.
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

  const panel = $("#sitepanel");
  const isOpen = () => !panel.hidden;

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
  function openSite() {
    if (batchPanelOpen()) $("#batchtoggle").click();
    panel.hidden = false;
    $("#canvaswrap").style.display = "none";
    $("#sidebar").hidden = true;  // segmentation controls are idle here — give the map the room
    $("#sitetoggle").classList.add("active");
    if (!$("#sitevec").value) $("#sitevec").value = $("#outdir").value;
    loadLeaflet().then(() => { if (map) map.invalidateSize(); })
      .catch((e) => toast(`Site view: ${e.message}`, "error", 9000));
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

  /* ---------------- map ---------------- */
  function ensureMap(mode) {
    if (map && mapMode === mode) return;
    if (map) { map.remove(); map = null; baseLayer = null; }
    mapMode = mode;
    const opts = { preferCanvas: true, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90 };
    map = mode === "map"
      ? L.map("sitemap", { ...opts, maxZoom: 26 })
      : L.map("sitemap", { ...opts, crs: L.CRS.Simple, minZoom: -6, maxZoom: 14 });
    L.control.scale({ imperial: false, maxWidth: 160 }).addTo(map);
    map.on("click", () => map.closePopup());
    setBasemap($("#sitebase").value);
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

  function photoUrl(g) {
    const q = new URLSearchParams({ path: g.photo, mode: site.mode, ox: site.origin[0],
                                    oy: site.origin[1], crs: g.crs || site.crs || "",
                                    size: 2048 });
    return `/api/site/photo?${q}`;
  }

  function buildLayers(it) {
    const g = it.g;
    it.photo = null; it.rings = null; it.lines = null; it.frame = null; it.label = null;
    if (!g.georeferenced) return;
    const c = it.color;
    if ((g.rings || []).length) {
      it.rings = L.polyline(g.rings, { color: c, weight: 1.6, opacity: 0.95, interactive: false });
    }
    if ((g.lines || []).length) {
      it.lines = L.polyline(g.lines, { color: c, weight: 1.2, opacity: 0.95, interactive: false });
    }
    const shape = g.footprint || hullOf(g);
    if (shape) {
      it.frame = L.polygon(shape, { color: c, weight: 1, opacity: 0.8, dashArray: "4 4",
                                    fill: true, fillOpacity: 0 });
      it.frame.bindPopup(() => popupHtml(it), { maxWidth: 320 });
      it.label = L.tooltip({ permanent: true, direction: "center", className: "sitelabel" })
        .setLatLng(L.polygon(shape).getBounds().getCenter()).setContent(escapeHtml(g.name));
    }
    if (g.photo && g.bounds) {
      it.photo = L.imageOverlay(photoUrl(g), g.bounds,
                                { opacity: +$("#siteopacity").value, interactive: false });
    }
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
    const showPhotos = $("#sitephotos").checked;
    const toggle = (layer, want) => {
      if (!layer) return;
      if (want && !map.hasLayer(layer)) layer.addTo(map);
      else if (!want && map.hasLayer(layer)) map.removeLayer(layer);
    };
    toggle(it.photo, on && showPhotos && it.photoOn);
    toggle(it.frame, on);
    toggle(it.label, on && $("#sitelabels").checked);
    toggle(it.rings, on && $("#siteoutlines").checked);
    toggle(it.lines, on && $("#sitecenter").checked);
    if (it.photo && map.hasLayer(it.photo)) it.photo.bringToBack();
    if (baseLayer) baseLayer.bringToBack();
  }
  const refreshAll = () => items.forEach(refreshItem);

  function zoomTo(list) {
    let b = null;
    for (const it of list) {
      if (!it.g.georeferenced) continue;
      const src = it.frame || it.photo;
      if (!src) continue;
      const bb = src.getBounds();
      b = b ? b.extend(bb) : L.latLngBounds(bb.getSouthWest(), bb.getNorthEast());
    }
    if (b) map.fitBounds(b.pad(0.08), { maxZoom: mapMode === "map" ? 24 : 9 });
  }

  /* ---------------- grave list ---------------- */
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function popupHtml(it) {
    const g = it.g;
    const notes = g.notes.length ? `<ul>${g.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : "";
    return `<b>${escapeHtml(g.name)}</b><br>
      <span class="hint">${escapeHtml(g.crs || "no CRS")} · ${g.n_rings} outlines · ${g.n_lines} centerlines` +
      `${g.dist_m != null ? ` · ${g.dist_m} m from site centre` : ""}</span>${notes}
      ${g.photo ? `<div class="btnrow"><button class="btn sm primary" data-open="${items.indexOf(it)}">✎ Open in editor</button></div>` : ""}`;
  }

  function renderList() {
    const wrap = $("#sitelist");
    if (!items.length) { wrap.innerHTML = ""; return; }
    wrap.innerHTML = items.map((it, i) => {
      const g = it.g;
      const warn = g.notes.length ? `<span class="sitewarn" title="${escapeHtml(g.notes.join("\n"))}">⚠</span>` : "";
      const off = g.georeferenced ? "" : " off";
      return `<div class="siterow${off}" data-i="${i}">
        <input type="checkbox" data-act="vis" ${it.visible ? "checked" : ""} ${g.georeferenced ? "" : "disabled"}>
        <span class="swatch" style="background:${it.color}"></span>
        <a href="#" data-act="zoom" title="${g.georeferenced ? "Zoom to grave" : "Not georeferenced — set GCPs in the editor (G)"}">${escapeHtml(g.name)}</a>
        ${warn}
        <span class="siteicons">
          ${g.photo && g.bounds ? `<button class="tbtn mini${it.photoOn ? " active" : ""}" data-act="photo" title="Show / hide this photo">📷</button>` : ""}
          ${g.photo ? `<button class="tbtn mini" data-act="open" title="Open in editor">✎</button>` : ""}
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
  });

  async function openInEditor(it) {
    const g = it.g;
    if (!g.photo) return;
    closeSite();
    await openBatchItem({ path: g.photo, out_dir: g.vec ? g.vec.dir : "", file: g.name });
  }

  /* ---------------- load ---------------- */
  async function loadSite() {
    const photoDir = $("#sitephotodir").value.trim();
    const vecDir = $("#sitevec").value.trim();
    store.set("photodir", photoDir);
    store.set("vecdir", vecDir);
    $("#sitesummary").textContent = "Reading the site…";
    $("#siteload").disabled = true;
    try {
      await loadLeaflet();
      const res = await apiPost("/api/site/scan", {
        photo_dir: photoDir, vec_dir: vecDir, recursive: $("#siterecursive").checked,
        vector_only: $("#sitevecs").checked,
      });
      site = res;
      ensureMap(res.mode);
      for (const it of items) for (const l of [it.photo, it.rings, it.lines, it.frame, it.label]) if (l) map.removeLayer(l);
      items = res.graves.map((g, i) => ({ g, color: COLORS[i % COLORS.length], visible: g.georeferenced, photoOn: true }));
      items.forEach(buildLayers);
      refreshAll();
      renderList();
      const placed = items.filter((it) => it.g.georeferenced);
      const warn = items.filter((it) => it.g.notes.length).length;
      $("#sitesummary").innerHTML =
        `<b>${placed.length}</b> of ${items.length} graves placed` +
        (res.crs ? ` · ${escapeHtml(res.crs)}` : " · local coordinates (no EPSG — no basemap)") +
        (warn ? ` · <span class="sitewarn">⚠ ${warn} to check</span>` : "") +
        (res.truncated ? " · <b>list truncated</b>" : "");
      if (!$("#siteout").value && photoDir) $("#siteout").value = photoDir.replace(/[\\/]+$/, "") + "\\site_plan.dxf";
      if (placed.length) zoomTo(placed);
      else toast("No georeferenced grave found — georeference the photos first (GCP tool, G).", "", 9000);
    } catch (err) {
      $("#sitesummary").textContent = "";
      toast(`Site view: ${err.message}`, "error", 9000);
    } finally {
      $("#siteload").disabled = false;
    }
  }
  $("#siteload").addEventListener("click", loadSite);
  for (const id of ["#sitephotodir", "#sitevec"]) {
    $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") loadSite(); });
  }

  /* ---------------- display controls ---------------- */
  $("#sitebase").addEventListener("change", (e) => setBasemap(e.target.value));
  for (const id of ["#sitephotos", "#siteoutlines", "#sitecenter", "#sitelabels"]) {
    $(id).addEventListener("change", refreshAll);
  }
  $("#siteopacity").addEventListener("input", (e) => {
    for (const it of items) if (it.photo) it.photo.setOpacity(+e.target.value);
  });
  $("#sitefit").addEventListener("click", () => zoomTo(items.filter((it) => it.visible)));
  $("#siteall").addEventListener("click", () => {
    const anyOff = items.some((it) => it.g.georeferenced && !it.visible);
    items.forEach((it) => { if (it.g.georeferenced) it.visible = anyOff; });
    refreshAll(); renderList();
  });

  /* ---------------- export ---------------- */
  $("#siteexport").addEventListener("click", async () => {
    const chosen = items.filter((it) => it.visible && it.g.georeferenced).map((it) => ({
      name: it.g.name, stem: it.g.stem, photo: it.g.photo, vec: it.g.vec,
    }));
    if (!chosen.length) { toast("No visible georeferenced graves to export.", "error"); return; }
    const out = $("#siteout").value.trim();
    if (!out) { toast("Enter where to save the site DXF.", "error"); return; }
    $("#siteexportstatus").textContent = `Writing ${chosen.length} graves…`;
    $("#siteexport").disabled = true;
    try {
      const r = await apiPost("/api/site/export_dxf", {
        graves: chosen, out_path: out, crs: site && site.crs, photos: $("#siteexportphotos").checked,
      });
      $("#siteexportstatus").innerHTML =
        `Saved <code>${escapeHtml(r.path)}</code> — ${r.graves} graves, ${r.photos} photos` +
        (r.skipped.length ? `, skipped: ${escapeHtml(r.skipped.join(", "))}` : "") +
        ` <a href="#" id="sitereveal">open folder</a>`;
      toast("Site DXF saved.", "ok");
    } catch (err) {
      $("#siteexportstatus").textContent = "";
      toast(`Site DXF export failed: ${err.message}`, "error", 9000);
    } finally {
      $("#siteexport").disabled = false;
    }
  });
  $("#siteexportstatus").addEventListener("click", (e) => {
    if (e.target.id !== "sitereveal") return;
    e.preventDefault();
    apiPost("/api/reveal", { path: $("#siteout").value.trim() }).catch((err) => toast(err.message, "error"));
  });

  /* ---------------- restore inputs ---------------- */
  $("#sitephotodir").value = store.get("photodir", "");
  $("#sitevec").value = store.get("vecdir", "");
  $("#sitebase").value = store.get("base", "esri");
})();
