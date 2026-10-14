"""Site view — every grave of one excavation on one map.

The editor works one photo at a time; this module puts the finished graves
back together in space so the whole site can be checked (and exported) as
one drawing:

  * **Photos** are found in a folder (optionally with subfolders). Each one
    that carries a georeference — a GCP sidecar from the in-app
    georeferencer, or a GeoTIFF transform — gets a footprint on the map and
    can be drawn rectified, in its true position.
  * **Vectors** come from the exports already on disk: per grave the
    ``<stem>_polygons.geojson`` / ``<stem>_centerlines.geojson`` pair the
    Export button writes, or the per-grave ``<stem>/<stem>.dxf``. Vectors
    exported BEFORE the photo was georeferenced are in pixel space; they
    are pushed through the photo's georeference so they still land right.
  * Graves with vectors but no photo in the folder are shown too.

Display coordinates:

  * ``map`` mode (the site has a known CRS): vectors go to the browser as
    lon/lat, photos are warped straight onto the Web-Mercator grid, so
    Leaflet can stack them over Esri / OSM tiles without any reprojection
    of its own.
  * ``local`` mode (georeferenced, but no EPSG anywhere): raw site
    coordinates minus a site origin (Leaflet CRS.Simple cannot cope with
    six-digit northings at sub-millimetre zoom), no basemap.

Everything here is stateless: the browser sends back the paths it got
from :func:`scan_site`, so nothing has to be kept on the server.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from boneseg.config import SUPPORTED_EXTENSIONS
from boneseg.data.readers import HAS_RASTERIO, GeoRef
from boneseg.logging_setup import get_logger

if HAS_RASTERIO:
    import rasterio
    from rasterio.warp import transform as warp_transform

logger = get_logger(__name__)

Polyline = list[tuple[float, float]]

# Coordinates below this (metres) are treated as pixel space, not a real
# projected CRS — same rule as the old SpojiGrobove script.
GEOREF_MIN = 10_000.0
# A grave this far from the site median is flagged as a likely
# georeferencing mistake (wrong points, wrong EPSG, wrong site).
MAX_SPREAD_M = 2_000.0
MAX_PHOTOS = 600
# Display-only line simplification (metres).
DISPLAY_TOL_M = 0.002
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}

_photo_cache: dict[tuple, bytes] = {}
_PHOTO_CACHE_MAX = 200


# --------------------------------------------------------------------------- #
# Georeference probing (no pixel read)                                         #
# --------------------------------------------------------------------------- #
def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def probe_photo(path: Path) -> tuple[GeoRef | None, int, int]:
    """Georeference + full-resolution size of a photo, without decoding it.

    Mirrors :func:`boneseg.data.readers.read_image`: a GeoTIFF's own
    transform wins, otherwise the GCP sidecar."""
    from boneseg.data.readers import _sidecar_georef

    if HAS_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(str(path)) as src:
                w, h = src.width, src.height
                if src.transform is not None and not src.transform.is_identity:
                    return GeoRef(src.transform, src.crs), w, h
            return _sidecar_georef(path), w, h
        except Exception:
            pass
    w, h = _image_size(path)
    return _sidecar_georef(path), w, h


def pixel_to_world_matrix(georef: GeoRef) -> np.ndarray:
    """3x3 matrix mapping full-res pixel (x, y, 1) -> world (homogeneous)."""
    if georef.homography is not None:
        return np.asarray(georef.homography, dtype=np.float64)
    t = georef.transform
    return np.array([[t.a, t.b, t.c], [t.d, t.e, t.f], [0, 0, 1.0]])


def _apply_h(h: np.ndarray, pts) -> np.ndarray:
    p = np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))])
    q = (h @ p.T).T
    return q[:, :2] / q[:, 2:3]


def _crs_key(crs) -> str | None:
    """'EPSG:3765' for a rasterio CRS / CRS string; None when unknown.

    A local engineering CRS (LOCAL_CS "Local Coordinates (m)") counts as
    unknown: it cannot be placed on a basemap or reprojected."""
    if crs is None:
        return None
    if isinstance(crs, str):
        m = re.fullmatch(r"\s*(?:urn:ogc:def:crs:)?EPSG:{1,2}(\d+)\s*", crs, re.I)
        if m:
            return f"EPSG:{m.group(1)}"
        if not HAS_RASTERIO:
            return None
        try:
            crs = rasterio.crs.CRS.from_user_input(crs)
        except Exception:
            return None
    try:
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else None
    except Exception:
        return None


def _transform_xy(src: str, dst: str, xs, ys) -> tuple[list[float], list[float]]:
    if src == dst:
        return list(xs), list(ys)
    ox, oy = warp_transform(src, dst, list(xs), list(ys))
    return list(ox), list(oy)


# --------------------------------------------------------------------------- #
# Vector discovery / loading                                                   #
# --------------------------------------------------------------------------- #
def _walk(root: Path, max_depth: int):
    root = Path(root)
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not d.startswith(".")] if depth < max_depth else []
        yield Path(dirpath), filenames


def index_vectors(vec_dir: Path | None) -> dict[str, dict]:
    """stem -> {"polygons", "centerlines", "dxf", "dir"} from export folders.

    When the same stem was exported to several folders the newest export
    wins (a re-export after fixing the georeference must replace the old
    one)."""
    found: dict[str, dict] = {}
    if vec_dir is None or not Path(vec_dir).is_dir():
        return found
    for d, files in _walk(Path(vec_dir), max_depth=4):
        for fn in files:
            low = fn.lower()
            if low.startswith("master") or low.startswith("site_"):
                continue
            kind = stem = None
            if low.endswith("_polygons.geojson"):
                kind, stem = "polygons", fn[: -len("_polygons.geojson")]
            elif low.endswith("_centerlines.geojson"):
                kind, stem = "centerlines", fn[: -len("_centerlines.geojson")]
            elif low.endswith(".dxf") and Path(fn).stem == d.name:
                # Per-grave export convention: <out>/<stem>/<stem>.dxf
                kind, stem = "dxf", Path(fn).stem
            if kind is None:
                continue
            p = d / fn
            mt = p.stat().st_mtime
            ent = found.get(stem)
            if ent is not None and ent["dir"] != str(d):
                if mt <= ent["mtime"]:
                    continue
                ent = None  # newer folder replaces the older export
            if ent is None:
                ent = found[stem] = {"dir": str(d), "mtime": mt}
            ent[kind] = str(p)
            ent["mtime"] = max(ent["mtime"], mt)
    return found


def _read_geojson(path: Path) -> tuple[list[Polyline], str | None]:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    crs = (gj.get("crs") or {}).get("properties", {}).get("name")
    out: list[Polyline] = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates")
        if not c:
            continue
        if t == "LineString":
            out.append([(float(x), float(y)) for x, y, *_ in c])
        elif t == "Polygon":
            out.extend([(float(x), float(y)) for x, y, *_ in ring] for ring in c)
        elif t == "MultiLineString":
            out.extend([(float(x), float(y)) for x, y, *_ in ln] for ln in c)
        elif t == "MultiPolygon":
            for poly in c:
                out.extend([(float(x), float(y)) for x, y, *_ in ring] for ring in poly)
    return out, (None if not crs or str(crs).startswith("pixel") else _crs_key(str(crs)))


def _read_dxf(path: Path) -> tuple[list[Polyline], list[Polyline]]:
    import ezdxf

    doc = ezdxf.readfile(str(path))
    rings: list[Polyline] = []
    lines: list[Polyline] = []
    for e in doc.modelspace().query("LWPOLYLINE"):
        pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
        if len(pts) < 2:
            continue
        (rings if e.closed else lines).append(pts)
    return rings, lines


def _looks_pixel(polys: list[Polyline]) -> bool:
    pts = [p for pl in polys for p in pl]
    if not pts:
        return False
    return max(abs(x) for x, _ in pts) < GEOREF_MIN and max(abs(y) for _, y in pts) < GEOREF_MIN


def load_vectors(vec: dict, georef: GeoRef | None, notes: list[str]
                 ) -> tuple[list[Polyline], list[Polyline], str | None, bool]:
    """(rings, lines, crs_key, georeferenced) in WORLD coords of the grave.

    Pixel-space exports (written before the photo got its GCPs) are mapped
    through the photo's georeference: export pixel space is (x, -y)."""
    rings: list[Polyline] = []
    lines: list[Polyline] = []
    crs: str | None = None
    if vec.get("polygons") or vec.get("centerlines"):
        if vec.get("polygons"):
            rings, crs = _read_geojson(Path(vec["polygons"]))
        if vec.get("centerlines"):
            lines, c2 = _read_geojson(Path(vec["centerlines"]))
            crs = crs or c2
    elif vec.get("dxf"):
        rings, lines = _read_dxf(Path(vec["dxf"]))

    if not rings and not lines:
        return [], [], None, False
    if not _looks_pixel(rings + lines):
        if crs is None and georef is not None:
            crs = _crs_key(georef.crs)
        return rings, lines, crs, True
    if georef is None:
        return rings, lines, None, False
    h = pixel_to_world_matrix(georef)

    def conv(pl: Polyline) -> Polyline:
        return [tuple(map(float, q)) for q in _apply_h(h, [(x, -y) for x, y in pl])]
    notes.append("vectors were exported before georeferencing — placed via the photo's GCPs")
    return [conv(r) for r in rings], [conv(l) for l in lines], _crs_key(georef.crs), True


# --------------------------------------------------------------------------- #
# Scan                                                                         #
# --------------------------------------------------------------------------- #
def _list_photos(photo_dir: Path, recursive: bool) -> list[Path]:
    if not photo_dir.is_dir():
        return []
    out: list[Path] = []
    for d, files in _walk(photo_dir, max_depth=6 if recursive else 0):
        for fn in sorted(files):
            p = d / fn
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and not fn.lower().endswith("_photo.jpg") \
                    and not re.search(r"_(mask|overlay|skeleton)\.(png|tif)$", fn.lower()):
                out.append(p)
                if len(out) >= MAX_PHOTOS:
                    return out
    return out


class _Frame:
    """Where a grave sits in display space (shared by scan + photo render)."""

    def __init__(self, mode: str, origin: tuple[float, float]):
        self.mode = mode
        self.ox, self.oy = origin

    def world_to_disp(self, crs: str | None, xs, ys) -> tuple[list[float], list[float]]:
        if self.mode == "map":
            return _transform_xy(crs, "EPSG:3857", xs, ys)
        return [x - self.ox for x in xs], [y - self.oy for y in ys]

    def disp_to_client(self, xs, ys) -> list[list[float]]:
        """Leaflet [lat, lng] pairs."""
        if self.mode == "map":
            lons, lats = _transform_xy("EPSG:3857", "EPSG:4326", xs, ys)
            return [[round(la, 9), round(lo, 9)] for lo, la in zip(lons, lats)]
        return [[round(y, 4), round(x, 4)] for x, y in zip(xs, ys)]

    def polylines_to_client(self, crs: str | None, polys: list[Polyline],
                            tol: float = DISPLAY_TOL_M) -> list[list]:
        """World polylines -> Leaflet [lat, lng] lists for DISPLAY.

        Lines are Douglas-Peucker simplified to ``tol`` metres first (a grave
        export easily holds 100k vertices — invisible below 2 mm, but heavy
        JSON for the browser), then projected in ONE call (per-line calls
        cost ~0.5 ms each in rasterio's env setup). Exports use the full
        geometry, never this."""
        arrs = [np.asarray(pl, dtype=np.float64) for pl in polys if len(pl) >= 2]
        if not arrs:
            return []
        c = arrs[0][0].copy()  # local origin: float32 approxPolyDP needs small numbers
        simp = []
        for a in arrs:
            if tol > 0 and len(a) > 3:
                closed = bool(np.allclose(a[0], a[-1]))
                q = cv2.approxPolyDP((a - c).astype(np.float32).reshape(-1, 1, 2),
                                     tol, closed).reshape(-1, 2).astype(np.float64) + c
                if closed and len(q) >= 3:
                    q = np.vstack([q, q[:1]])
                a = q if len(q) >= 2 else a
            simp.append(a)
        flat = np.concatenate(simp)
        if self.mode == "map":
            lons, lats = _transform_xy(crs, "EPSG:4326", flat[:, 0].tolist(), flat[:, 1].tolist())
            ll = np.round(np.column_stack([lats, lons]), 9)
        else:
            ll = np.round(np.column_stack([flat[:, 1] - self.oy, flat[:, 0] - self.ox]), 4)
        out, i = [], 0
        for a in simp:
            out.append(ll[i:i + len(a)].tolist())
            i += len(a)
        return out


def photo_homography(georef: GeoRef, w: int, h: int, crs: str | None,
                     frame: _Frame) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Full-res pixel -> display-space homography + footprint bbox.

    Map mode: the TM/UTM -> Web-Mercator step is not projective, but over
    a photo's footprint it is affine to well under a millimetre, so a
    homography fitted on a 7x7 grid of the photo is exact for display."""
    hw = pixel_to_world_matrix(georef)
    if frame.mode == "local":
        hd = np.array([[1, 0, -frame.ox], [0, 1, -frame.oy], [0, 0, 1.0]]) @ hw
    else:
        gx, gy = np.meshgrid(np.linspace(0, w, 7), np.linspace(0, h, 7))
        src = np.column_stack([gx.ravel(), gy.ravel()])
        world = _apply_h(hw, src)
        dx, dy = frame.world_to_disp(crs, world[:, 0], world[:, 1])
        dst = np.column_stack([dx, dy])
        hd, _ = cv2.findHomography(src, dst, 0)
    corners = _apply_h(hd, [(0, 0), (w, 0), (w, h), (0, h)])
    minx, miny = corners.min(axis=0)
    maxx, maxy = corners.max(axis=0)
    return hd, (float(minx), float(miny), float(maxx), float(maxy))


def scan_site(photo_dir: str, vec_dir: str, recursive: bool = True,
              vector_only: bool | None = None) -> dict:
    """Collect every grave of a site: georeference, vectors, warnings.

    ``vector_only``: also list exported graves whose photo is not in the
    photo folder. Defaults to on only when no photo folder is given — an
    export folder usually holds graves of EVERY site ever processed."""
    pdir = Path(photo_dir) if photo_dir else None
    vdir = Path(vec_dir) if vec_dir else None
    photos = _list_photos(pdir, recursive) if pdir else []
    vecs = index_vectors(vdir)

    graves: list[dict] = []
    used_vec: set[str] = set()
    stem_count = Counter(p.stem for p in photos)
    for p in photos:
        g: dict = {"name": p.stem if stem_count[p.stem] == 1 else f"{p.parent.name}/{p.stem}",
                   "stem": p.stem, "photo": str(p), "notes": []}
        try:
            georef, w, h = probe_photo(p)
        except Exception as exc:
            g["notes"].append(f"photo unreadable ({exc})")
            georef, w, h = None, 0, 0
        g.update(_georef=georef, w=w, h=h, crs=_crs_key(georef.crs) if georef else None)
        v = vecs.get(p.stem)
        if v is not None:
            used_vec.add(p.stem)
            g["vec"] = {k: v[k] for k in ("polygons", "centerlines", "dxf", "dir") if k in v}
        graves.append(g)
    if vector_only is None:
        vector_only = pdir is None
    for stem, v in sorted(vecs.items()) if vector_only else ():
        if stem in used_vec:
            continue
        graves.append({"name": stem, "stem": stem, "photo": None, "notes": [],
                       "_georef": None, "w": 0, "h": 0, "crs": None,
                       "vec": {k: v[k] for k in ("polygons", "centerlines", "dxf", "dir") if k in v}})

    # Load vectors in world coordinates (needs the photo georef for pixel exports).
    for g in graves:
        g["_rings"], g["_lines"] = [], []
        g["vec_georef"] = False
        if g.get("vec"):
            try:
                r, l, c, ok = load_vectors(g["vec"], g["_georef"], g["notes"])
                g["_rings"], g["_lines"], g["vec_georef"] = r, l, ok
                g["crs"] = g["crs"] or c
                if not ok:
                    g["notes"].append("vectors are in pixel space and the photo has no georeference")
            except Exception as exc:
                g["notes"].append(f"vectors unreadable ({exc})")
        g["georeferenced"] = g["_georef"] is not None or g["vec_georef"]

    # Raw centre of every placed grave, in its OWN coordinates.
    for g in graves:
        if not g["georeferenced"]:
            continue
        pts = [q for pl in g["_rings"] + g["_lines"] for q in pl]
        if not pts and g["_georef"] is not None:
            pts = [tuple(q) for q in _apply_h(pixel_to_world_matrix(g["_georef"]),
                                               [(0, 0), (g["w"], g["h"])])]
        if pts:
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            g["_raw"] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        else:
            g["georeferenced"] = False

    # Site CRS = most common known CRS. A grave whose GCPs carry no EPSG
    # takes the CRS of the nearest grave with similar raw numbers (same
    # survey grid), else the site CRS.
    crs_votes = Counter(g["crs"] for g in graves if g["georeferenced"] and g["crs"])
    site_crs = crs_votes.most_common(1)[0][0] if crs_votes else None
    if len(crs_votes) > 1:
        logger.warning("Site view: mixed CRS %s — showing everything in %s via reprojection",
                       dict(crs_votes), site_crs)
    known = [(g["_raw"], g["crs"]) for g in graves if g["georeferenced"] and g["crs"]]
    for g in graves:
        if g["georeferenced"] and not g["crs"] and site_crs:
            near = min(known, key=lambda k: math.dist(k[0], g["_raw"]))
            g["crs"] = near[1] if math.dist(near[0], g["_raw"]) < 50_000 else site_crs
            g["notes"].append(f"no EPSG set in its GCPs — assumed {g['crs']}")
    mode = "map" if (site_crs and HAS_RASTERIO) else "local"

    # Centres in the site CRS for the spread check + local origin.
    centres = []
    for g in graves:
        if not g["georeferenced"]:
            continue
        cx, cy = g["_raw"]
        if mode == "map" and g["crs"] != site_crs:
            try:
                (cx,), (cy,) = _transform_xy(g["crs"], site_crs, [cx], [cy])
            except Exception as exc:
                g["notes"].append(f"cannot convert {g['crs']} -> {site_crs} ({exc})")
                g["georeferenced"] = False
                continue
        g["_centre"] = (cx, cy)
        centres.append((cx, cy))
    origin = (0.0, 0.0)
    if centres:
        mx = float(np.median([c[0] for c in centres]))
        my = float(np.median([c[1] for c in centres]))
        origin = (round(mx), round(my))
        for g in graves:
            if "_centre" in g:
                d = math.hypot(g["_centre"][0] - mx, g["_centre"][1] - my)
                g["dist_m"] = round(d, 1)
                if d > MAX_SPREAD_M:
                    g["notes"].append(f"{d / 1000:.1f} km from the rest of the site — "
                                      "wrong points / EPSG, or another site?")
    frame = _Frame(mode, origin)

    # Display geometry for the browser.
    out = []
    for g in graves:
        item = {k: g[k] for k in ("name", "stem", "photo", "notes", "georeferenced",
                                  "crs", "w", "h") if k in g}
        item["vec"] = g.get("vec")
        item["dist_m"] = g.get("dist_m")
        item["n_rings"], item["n_lines"] = len(g["_rings"]), len(g["_lines"])
        if g["georeferenced"]:
            try:
                item["rings"] = frame.polylines_to_client(
                    g["crs"], [r for r in g["_rings"] if len(r) >= 3])
                item["lines"] = frame.polylines_to_client(
                    g["crs"], [l for l in g["_lines"] if len(l) >= 2])
                if g["_georef"] is not None and g["w"]:
                    hd, (x0, y0, x1, y1) = photo_homography(g["_georef"], g["w"], g["h"],
                                                           g["crs"], frame)
                    item["bounds"] = frame.disp_to_client([x0, x1], [y0, y1])
                    fp = _apply_h(hd, [(0, 0), (g["w"], 0), (g["w"], g["h"]), (0, g["h"])])
                    item["footprint"] = frame.disp_to_client(fp[:, 0], fp[:, 1])
            except Exception as exc:
                logger.exception("Site view: could not place %s", g["name"])
                item["notes"].append(f"could not place on the map ({exc})")
                item["georeferenced"] = False
        out.append(item)

    n_geo = sum(1 for g in out if g["georeferenced"])
    logger.info("Site view scan: %d graves (%d placed), mode=%s, crs=%s",
                len(out), n_geo, mode, site_crs)
    return {"graves": out, "mode": mode, "crs": site_crs, "origin": list(origin),
            "truncated": len(photos) >= MAX_PHOTOS}


# --------------------------------------------------------------------------- #
# Photo rendering                                                              #
# --------------------------------------------------------------------------- #
def _load_reduced(path: Path, max_dim: int) -> np.ndarray:
    """RGB at roughly ``max_dim`` on the long side (JPEG draft decode = fast)."""
    if HAS_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(str(path)) as src:
                f = max(1.0, max(src.width, src.height) / max_dim)
                oh, ow = max(1, int(src.height / f)), max(1, int(src.width / f))
                bands = [1, 2, 3] if src.count >= 3 else [1]
                arr = src.read(bands, out_shape=(len(bands), oh, ow))
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                return np.moveaxis(arr, 0, -1).astype(np.uint8)
        except Exception:
            pass
    with Image.open(path) as im:
        im.draft("RGB", (max_dim, max_dim))
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim), Image.BILINEAR)
        return np.asarray(im)


def _border_fill_alpha(img: np.ndarray) -> np.ndarray:
    """255 = photo, 0 = the flat white/black padding that orthophoto crops
    carry around the real image. Only padding CONNECTED TO THE IMAGE EDGE
    is cut, so a bright stone inside the grave stays opaque."""
    alpha = np.full(img.shape[:2], 255, np.uint8)
    for pad in ((img.min(axis=2) >= 250), (img.max(axis=2) <= 3)):
        if not (pad[0].any() or pad[-1].any() or pad[:, 0].any() or pad[:, -1].any()):
            continue
        n, lab = cv2.connectedComponents(pad.astype(np.uint8), connectivity=4)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        edge = edge[edge > 0]
        alpha[np.isin(lab, edge)] = 0
    return alpha


def render_photo(photo: str, mode: str, origin: tuple[float, float],
                 site_crs: str | None, max_dim: int = 2048) -> bytes:
    """The photo warped onto the display grid of its own footprint bbox,
    as WebP with transparency outside the (rectified) photo."""
    path = Path(photo)
    georef, w, h = probe_photo(path)
    if georef is None:
        raise ValueError("photo has no georeference")
    sidecar = path.with_suffix(path.suffix + ".gcps.json")
    key = (str(path), path.stat().st_mtime,
           sidecar.stat().st_mtime if sidecar.exists() else 0, mode, tuple(origin),
           site_crs, max_dim)
    if key in _photo_cache:
        return _photo_cache[key]

    crs = _crs_key(georef.crs) or site_crs
    frame = _Frame(mode, origin)
    hd, (x0, y0, x1, y1) = photo_homography(georef, w, h, crs, frame)
    gsd = max(x1 - x0, y1 - y0) / max_dim or 1.0
    out_w = max(1, math.ceil((x1 - x0) / gsd))
    out_h = max(1, math.ceil((y1 - y0) / gsd))

    img = _load_reduced(path, max(max_dim, 1) * 3 // 2)
    rh, rw = img.shape[:2]
    scale = np.diag([w / rw, h / rh, 1.0])                 # reduced px -> full px
    to_grid = np.array([[1 / gsd, 0, -x0 / gsd], [0, -1 / gsd, y1 / gsd], [0, 0, 1.0]])
    k = to_grid @ hd @ scale
    rgb = cv2.warpPerspective(img, k, (out_w, out_h), flags=cv2.INTER_AREA
                              if rw > out_w else cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(_border_fill_alpha(img), k, (out_w, out_h),
                                flags=cv2.INTER_NEAREST)
    rgba = np.dstack([rgb, alpha])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "WEBP", quality=82, method=3)
    data = buf.getvalue()
    if len(_photo_cache) >= _PHOTO_CACHE_MAX:
        _photo_cache.pop(next(iter(_photo_cache)))
    _photo_cache[key] = data
    return data


# --------------------------------------------------------------------------- #
# Site DXF export                                                              #
# --------------------------------------------------------------------------- #
_LAYER_COLORS = (1, 3, 4, 5, 6, 2, 30, 140, 40, 210, 92, 172)


def _dxf_layer(name: str, suffix: str) -> str:
    clean = re.sub(r'[<>/\\":;?*|,=`]+', "-", name).strip(" .-") or "GROB"
    return f"{clean}_{suffix}"


def export_site_dxf(graves: list[dict], out_path: str, site_crs: str | None,
                    include_photos: bool = True, photo_max_dim: int = 3000) -> dict:
    """One DXF for the whole site, in the site CRS (metres).

    Per grave: ``<NAME>_OUTLINE``, ``<NAME>_CENTERLINE`` and (optionally)
    ``<NAME>_PHOTO`` — the rectified photo as an IMAGE underlay, JPEGs in a
    ``<dxf name>_photos`` folder next to the DXF. Graves in another CRS are
    reprojected into the site CRS."""
    import ezdxf
    from rasterio.transform import Affine

    from boneseg.data.georef_fit import rectify_params

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    photo_dir = out.parent / f"{out.stem}_photos"
    if include_photos:
        doc.set_raster_variables(frame=1, quality=1, units="m")

    done, skipped, photos = [], [], 0
    for i, g in enumerate(graves):
        name = g.get("name") or g.get("stem") or f"GROB{i + 1}"
        notes: list[str] = []
        georef = None
        if g.get("photo"):
            try:
                georef, w, h = probe_photo(Path(g["photo"]))
            except Exception:
                georef = None
        rings: list[Polyline] = []
        lines: list[Polyline] = []
        crs = _crs_key(georef.crs) if georef else None
        if g.get("vec"):
            try:
                rings, lines, c, ok = load_vectors(g["vec"], georef, notes)
                if not ok:
                    rings, lines = [], []
                crs = crs or c
            except Exception as exc:
                logger.warning("Site DXF: vectors of %s unreadable (%s)", name, exc)
        crs = crs or site_crs
        if georef is None and not rings and not lines:
            skipped.append(name)
            continue

        def to_site(pl: Polyline) -> Polyline:
            if not site_crs or crs == site_crs:
                return pl
            xs, ys = _transform_xy(crs, site_crs, [q[0] for q in pl], [q[1] for q in pl])
            return list(zip(xs, ys))

        color = _LAYER_COLORS[i % len(_LAYER_COLORS)]
        lo, lc = _dxf_layer(name, "OUTLINE"), _dxf_layer(name, "CENTERLINE")
        for ln in (lo, lc):
            if ln not in doc.layers:
                doc.layers.add(ln, color=color)

        if include_photos and georef is not None:
            try:
                img = _load_reduced(Path(g["photo"]), photo_max_dim)
                rh, rw = img.shape[:2]
                s = np.diag([w / rw, h / rh, 1.0])
                hw = pixel_to_world_matrix(georef) @ s
                if site_crs and crs != site_crs:
                    # Reproject the rectification: fit pixel -> site CRS.
                    gx, gy = np.meshgrid(np.linspace(0, rw, 7), np.linspace(0, rh, 7))
                    src = np.column_stack([gx.ravel(), gy.ravel()])
                    wpts = _apply_h(hw, src)
                    xs, ys = _transform_xy(crs, site_crs, wpts[:, 0], wpts[:, 1])
                    hw, _ = cv2.findHomography(src, np.column_stack([xs, ys]), 0)
                t = Affine(hw[0, 0], hw[0, 1], hw[0, 2], hw[1, 0], hw[1, 1], hw[1, 2])
                small = GeoRef(t, None, homography=hw if abs(hw[2, 0]) + abs(hw[2, 1]) > 1e-12
                               else None)
                grid, k, ow, oh = rectify_params(small, rw, rh,
                                                 max_pixels=min(4 * rw * rh, 12_000_000))
                rect = cv2.warpPerspective(img, k, (ow, oh), flags=cv2.INTER_LINEAR)
                photo_dir.mkdir(exist_ok=True)
                safe = (re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or f"G{i}").upper()
                jpg = photo_dir / f"{safe}.jpg"
                Image.fromarray(rect).save(jpg, "JPEG", quality=88)
                lp = _dxf_layer(name, "PHOTO")
                if lp not in doc.layers:
                    doc.layers.add(lp, color=8)
                idef = doc.add_image_def(filename=str(jpg.resolve()), size_in_pixel=(ow, oh),
                                         name=f"{safe}_PHOTO")
                ix, iy = grid * (0.0, float(oh))
                ent = msp.add_image(image_def=idef, insert=(ix, iy, 0.0),
                                    size_in_units=(ow * grid.a, oh * -grid.e),
                                    dxfattribs={"layer": lp})
                ent.dxf.u_pixel = (grid.a, 0.0, 0.0)
                ent.dxf.v_pixel = (0.0, -grid.e, 0.0)
                photos += 1
            except Exception:
                logger.exception("Site DXF: photo underlay of %s failed", name)

        for r in rings:
            if len(r) >= 3:
                msp.add_lwpolyline(to_site(r), close=True, dxfattribs={"layer": lo})
        for ln_ in lines:
            if len(ln_) >= 2:
                msp.add_lwpolyline(to_site(ln_), close=False, dxfattribs={"layer": lc})
        done.append(name)

    if not done:
        raise ValueError("None of the selected graves is georeferenced — nothing to export.")
    doc.saveas(str(out))
    logger.info("Site DXF written: %s (%d graves, %d photos, %d skipped)",
                out, len(done), photos, len(skipped))
    return {"path": str(out), "graves": len(done), "photos": photos, "skipped": skipped}
