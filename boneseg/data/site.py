"""Site view — every grave of one excavation on one map.

The editor works one photo at a time; this module puts the finished graves
back together in space so the whole site can be checked and exported as
one drawing.

Sources (any mix, picked by the user):

  * **Photos** — a GCP sidecar from the in-app georeferencer or a GeoTIFF
    transform places them; their vectors are looked up in the export
    folder(s) by file name.
  * **Export folders / files** — what the Export button wrote per grave:
    ``<stem>_polygons.geojson`` / ``<stem>_centerlines.geojson`` and/or
    ``<stem>.dxf``, plus the photo as ``<stem>_photo.tif`` (GeoTIFF) or the
    DXF's IMAGE underlay. Picking ANY file of such a folder means "this
    grave".
  * **Folders** — every photo and every export found inside (optionally
    with subfolders).
  * **The grave open in the editor** — straight from memory, so a fresh
    georeference can be checked on the map before anything is exported.

Vectors exported BEFORE the photo was georeferenced are in pixel space;
they are pushed through the photo's georeference so they still land right.

Display coordinates:

  * ``map`` mode (the site has a known CRS): vectors go to the browser as
    lon/lat, photos are warped straight onto the Web-Mercator grid, so
    Leaflet can stack them over Esri / OSM tiles with no reprojection of
    its own.
  * ``local`` mode (georeferenced, but no EPSG anywhere): raw site
    coordinates minus a site origin (Leaflet CRS.Simple cannot cope with
    six-digit northings at sub-millimetre zoom), no basemap.

Stateless: the browser sends back the grave descriptions it got from
:func:`scan_site` for photo tiles and exports.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from boneseg.config import SUPPORTED_EXTENSIONS
from boneseg.data.readers import HAS_RASTERIO, GeoRef
from boneseg.logging_setup import get_logger

if HAS_RASTERIO:
    import rasterio
    from rasterio.transform import Affine
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
# Long side of the photo tiles drawn on the map.
PHOTO_TILE_DIM = 2048
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}
# Files the Export button writes next to the vectors — never "photos".
_ARTIFACT_RE = re.compile(r"_(photo|mask|overlay|skeleton|plate)\.(png|jpe?g|tiff?|pdf)$", re.I)

_photo_cache: dict[tuple, bytes] = {}
_PHOTO_CACHE_MAX = 200

# The grave open in the editor, supplied by the web server:
#   None, or {"name", "stem", "photo", "image", "georef", "rings", "lines", "key"}
CurrentProvider = Callable[[], "dict | None"]


def _norm(stem: str) -> str:
    """Matching key for grave names — Windows strips trailing dots from
    folder names ("Gr.41 a.k." exports into folder "Gr.41 a.k")."""
    return stem.rstrip(". ").lower()


# --------------------------------------------------------------------------- #
# Georeference helpers                                                         #
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


def _transform_xy(src: str | None, dst: str | None, xs, ys) -> tuple[list[float], list[float]]:
    if src == dst or not src or not dst:
        return list(xs), list(ys)
    ox, oy = warp_transform(src, dst, list(xs), list(ys))
    return list(ox), list(oy)


def _to_crs_matrix(hw: np.ndarray, w: int, h: int, src: str | None,
                   dst: str | None) -> np.ndarray:
    """Pixel -> ``dst`` homography from a pixel -> ``src`` one. Exact when
    the CRSs match; otherwise fitted on a 7x7 grid (over a photo's
    footprint the reprojection is affine to well under a millimetre)."""
    if not src or not dst or src == dst:
        return hw
    gx, gy = np.meshgrid(np.linspace(0, w, 7), np.linspace(0, h, 7))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    world = _apply_h(hw, grid)
    xs, ys = _transform_xy(src, dst, world[:, 0], world[:, 1])
    out, _ = cv2.findHomography(grid, np.column_stack([xs, ys]), 0)
    return out


# --------------------------------------------------------------------------- #
# Export-folder discovery                                                      #
# --------------------------------------------------------------------------- #
def _walk(root: Path, max_depth: int):
    root = Path(root)
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not d.startswith(".")] if depth < max_depth else []
        yield Path(dirpath), filenames


def _export_kind(fn: str) -> tuple[str, str] | None:
    """(kind, stem) for a file the Export button writes, else None."""
    low = fn.lower()
    for suffix, kind in (("_polygons.geojson", "polygons"),
                         ("_centerlines.geojson", "centerlines"),
                         ("_photo.tif", "photo_tif"), ("_photo.tiff", "photo_tif")):
        if low.endswith(suffix):
            return kind, fn[: -len(suffix)]
    if low.endswith(".dxf"):
        return "dxf", fn[:-4]
    return None


def index_exports(root: Path | None, max_depth: int = 4) -> dict[str, dict]:
    """norm-stem -> {"stem", "dir", "polygons", "centerlines", "dxf",
    "photo_tif"} for every per-grave export below ``root``.

    When the same grave was exported to several folders the newest export
    wins (a re-export after fixing the georeference replaces the old one).
    A DXF counts only as ``<stem>/<stem>.dxf`` or next to that grave's
    GeoJSON, so site-wide DXFs (master.dxf, site plans) are not graves."""
    found: dict[str, dict] = {}
    if root is None or not Path(root).is_dir():
        return found
    for d, files in _walk(Path(root), max_depth):
        local: dict[str, dict] = {}
        for fn in files:
            low = fn.lower()
            if low.startswith(("master", "site_")):
                continue
            k = _export_kind(fn)
            if k is None:
                continue
            kind, stem = k
            local.setdefault(_norm(stem), {"stem": stem, "dir": str(d)})[kind] = str(d / fn)
        for key, ent in local.items():
            if "dxf" in ent and not ({"polygons", "centerlines", "photo_tif"} & ent.keys()) \
                    and key != _norm(d.name):
                continue  # a loose DXF, not a per-grave export
            if not ({"polygons", "centerlines", "dxf"} & ent.keys()):
                continue
            ent["mtime"] = max(Path(p).stat().st_mtime for k2, p in ent.items()
                               if k2 in ("polygons", "centerlines", "dxf", "photo_tif"))
            old = found.get(key)
            if old is None or ent["mtime"] > old["mtime"]:
                found[key] = ent
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
        if e.dxf.layer.upper() == "PHOTO":
            continue
        pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
        if len(pts) < 2:
            continue
        (rings if e.closed else lines).append(pts)
    return rings, lines


def _dxf_image(path: Path) -> tuple[str, list[float]] | None:
    """(jpg path, affine a..f) of the photo underlay in an exported DXF.

    IMAGE entity: insert = lower-left corner, u/v = one pixel along the
    image's x / y (y up), so pixel (col, row-from-top) maps to
    insert + col*u + (h - row)*v."""
    import ezdxf

    try:
        doc = ezdxf.readfile(str(path))
    except Exception:
        return None
    for img in doc.modelspace().query("IMAGE"):
        try:
            idef = img.image_def
            fn = Path(idef.dxf.filename)
            if not fn.is_file():
                fn = path.parent / fn.name
            if not fn.is_file():
                continue
            h = float(idef.dxf.image_size[1])
            ix, iy = img.dxf.insert[0], img.dxf.insert[1]
            u, v = img.dxf.u_pixel, img.dxf.v_pixel
            return str(fn), [u[0], -v[0], ix + h * v[0], u[1], -v[1], iy + h * v[1]]
        except Exception:
            continue
    return None


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
    # Exports before 2026-10 traced raw pixel edges (zig-zag in CAD).
    from boneseg.postprocessing.vectorize import smooth_staircase_ring
    rings = [smooth_staircase_ring(r) for r in rings]
    if not _looks_pixel(rings + lines):
        return rings, lines, crs, True
    if georef is None:
        return rings, lines, None, False
    h = pixel_to_world_matrix(georef)

    def conv(pl: Polyline) -> Polyline:
        return [tuple(map(float, q)) for q in _apply_h(h, [(x, -y) for x, y in pl])]
    notes.append("vectors were exported before georeferencing — placed via the photo's GCPs")
    return [conv(r) for r in rings], [conv(l) for l in lines], _crs_key(georef.crs), True


# --------------------------------------------------------------------------- #
# Collecting graves                                                            #
# --------------------------------------------------------------------------- #
def _is_photo(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTENSIONS and not _ARTIFACT_RE.search(p.name)


def _list_photos(folder: Path, recursive: bool, export_stems: set[str]) -> list[Path]:
    out: list[Path] = []
    for d, files in _walk(folder, max_depth=6 if recursive else 0):
        for fn in sorted(files):
            p = d / fn
            # Inside an export folder the "<stem>.tif" is the binary mask.
            if not _is_photo(p) or _norm(p.stem) in export_stems:
                continue
            out.append(p)
            if len(out) >= MAX_PHOTOS:
                return out
    return out


def _new_grave(name: str, stem: str) -> dict:
    return {"name": name, "stem": stem, "photo": None, "photo_kind": None,
            "photo_affine": None, "vec": None, "notes": [], "current": False}


def _grave_from_export(ent: dict) -> dict:
    g = _new_grave(ent["stem"].rstrip(" "), ent["stem"])
    g["vec"] = {k: ent[k] for k in ("polygons", "centerlines", "dxf", "dir") if k in ent}
    if ent.get("photo_tif"):
        g["photo"], g["photo_kind"] = ent["photo_tif"], "file"
    elif ent.get("dxf"):
        img = _dxf_image(Path(ent["dxf"]))
        if img is not None:
            g["photo"], g["photo_kind"], g["photo_affine"] = img[0], "affine", img[1]
    return g


def collect_graves(sources: list[str], vec_dir: str | None, recursive: bool) -> tuple[list[dict], list[str]]:
    """Resolve user-picked files/folders into grave descriptions."""
    photos: list[Path] = []
    exports: dict[str, dict] = {}
    problems: list[str] = []
    seen_photo: set[str] = set()

    def add_export(key: str, ent: dict) -> None:
        old = exports.get(key)
        if old is None or ent["mtime"] > old["mtime"]:
            exports[key] = ent

    for raw in sources:
        p = Path(str(raw).strip().strip('"'))
        if p.is_dir():
            idx = index_exports(p, max_depth=6 if recursive else 1)
            for k, ent in idx.items():
                add_export(k, ent)
            for ph in _list_photos(p, recursive, set(idx)):
                if str(ph) not in seen_photo:
                    seen_photo.add(str(ph))
                    photos.append(ph)
        elif p.is_file():
            k = _export_kind(p.name)
            if k is not None or _ARTIFACT_RE.search(p.name):
                stem = k[1] if k else _ARTIFACT_RE.sub("", p.name)
                ent = index_exports(p.parent, max_depth=0).get(_norm(stem))
                if ent is None:
                    problems.append(f"{p.name}: not a per-grave export")
                else:
                    add_export(_norm(stem), ent)
            elif _is_photo(p):
                if str(p) not in seen_photo:
                    seen_photo.add(str(p))
                    photos.append(p)
            else:
                problems.append(f"{p.name}: unsupported file")
        else:
            problems.append(f"{p}: not found")

    # Vectors for picked photos: the picked exports + the export folder.
    lookup = index_exports(Path(vec_dir)) if vec_dir else {}
    lookup.update(exports)

    graves: list[dict] = []
    used: set[str] = set()
    stem_count = Counter(_norm(p.stem) for p in photos)
    for p in photos:
        key = _norm(p.stem)
        g = _new_grave(p.stem if stem_count[key] == 1 else f"{p.parent.name}/{p.stem}", p.stem)
        g["photo"], g["photo_kind"] = str(p), "file"
        ent = lookup.get(key)
        if ent is not None:
            used.add(key)
            g["vec"] = {k: ent[k] for k in ("polygons", "centerlines", "dxf", "dir") if k in ent}
        graves.append(g)
    for key, ent in sorted(exports.items()):
        if key not in used:
            graves.append(_grave_from_export(ent))
    return graves, problems


# --------------------------------------------------------------------------- #
# Materialising a grave (georef, size, pixels, vectors)                        #
# --------------------------------------------------------------------------- #
def _photo_georef(g: dict, current: dict | None) -> tuple[GeoRef | None, int, int]:
    kind = g.get("photo_kind")
    if g.get("current") and current is not None:
        h, w = current["image"].shape[:2]
        return current["georef"], w, h
    if kind == "file":
        return probe_photo(Path(g["photo"]))
    if kind == "affine" and g.get("photo_affine"):
        w, h = _image_size(Path(g["photo"]))
        return GeoRef(Affine(*g["photo_affine"]), None), w, h
    return None, 0, 0


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


def _pixels(g: dict, current: dict | None, max_dim: int) -> np.ndarray:
    if g.get("current") and current is not None:
        img = current["image"]
        h, w = img.shape[:2]
        f = max(1.0, max(w, h) / max_dim)
        if f > 1:
            img = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))),
                             interpolation=cv2.INTER_AREA)
        return img if img.ndim == 3 else np.stack([img] * 3, axis=-1)
    return _load_reduced(Path(g["photo"]), max_dim)


def _border_fill_alpha(img: np.ndarray) -> np.ndarray:
    """255 = photo, 0 = the flat white/black padding that orthophoto crops
    carry around the real image. Only padding CONNECTED TO THE IMAGE EDGE
    is cut, so a bright stone inside the grave stays opaque."""
    alpha = np.full(img.shape[:2], 255, np.uint8)
    for pad in ((img.min(axis=2) >= 250), (img.max(axis=2) <= 3)):
        if not (pad[0].any() or pad[-1].any() or pad[:, 0].any() or pad[:, -1].any()):
            continue
        _n, lab = cv2.connectedComponents(pad.astype(np.uint8), connectivity=4)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        edge = edge[edge > 0]
        alpha[np.isin(lab, edge)] = 0
    return alpha


def _warp_to_grid(img: np.ndarray, h_full: np.ndarray, w: int, h: int, max_dim: int
                  ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Warp a (possibly reduced) photo onto an axis-aligned grid in the
    target space of ``h_full`` (full-res pixel -> target). Returns RGBA
    (alpha = photo coverage) and the grid bbox (x0, y0, x1, y1)."""
    x0, y1, gsd, out_w, out_h = _grid(h_full, w, h, max_dim)
    rh, rw = img.shape[:2]
    scale = np.diag([w / rw, h / rh, 1.0])                 # reduced px -> full px
    to_grid = np.array([[1 / gsd, 0, -x0 / gsd], [0, -1 / gsd, y1 / gsd], [0, 0, 1.0]])
    k = to_grid @ h_full @ scale
    interp = cv2.INTER_AREA if rw > out_w else cv2.INTER_LINEAR
    rgb = cv2.warpPerspective(img, k, (out_w, out_h), flags=interp)
    alpha = cv2.warpPerspective(_border_fill_alpha(img), k, (out_w, out_h),
                                flags=cv2.INTER_NEAREST)
    return np.dstack([rgb, alpha]), _grid_bbox(x0, y1, gsd, out_w, out_h)


def _grid(h_full: np.ndarray, w: int, h: int, max_dim: int):
    """(x0, y1, gsd, out_w, out_h) of the axis-aligned grid covering the
    photo footprint in the target space of ``h_full``."""
    corners = _apply_h(h_full, [(0, 0), (w, 0), (w, h), (0, h)])
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    gsd = max(x1 - x0, y1 - y0) / max_dim or 1.0
    return (float(x0), float(y1), float(gsd), max(1, math.ceil((x1 - x0) / gsd)),
            max(1, math.ceil((y1 - y0) / gsd)))


def _grid_bbox(x0, y1, gsd, out_w, out_h) -> tuple[float, float, float, float]:
    return (x0, y1 - out_h * gsd, x0 + out_w * gsd, y1)


# --------------------------------------------------------------------------- #
# Display frame                                                                #
# --------------------------------------------------------------------------- #
class _Frame:
    """Where a grave sits in display space (shared by scan + photo render)."""

    def __init__(self, mode: str, origin: tuple[float, float]):
        self.mode = mode
        self.ox, self.oy = origin

    def disp_matrix(self, hw: np.ndarray, w: int, h: int, crs: str | None) -> np.ndarray:
        """Full-res pixel -> display coords (Web Mercator / local offset)."""
        if self.mode == "local":
            return np.array([[1, 0, -self.ox], [0, 1, -self.oy], [0, 0, 1.0]]) @ hw
        return _to_crs_matrix(hw, w, h, crs, "EPSG:3857")

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


# --------------------------------------------------------------------------- #
# Scan                                                                         #
# --------------------------------------------------------------------------- #
def scan_site(sources: list[str], vec_dir: str | None = None, recursive: bool = True,
              current: dict | None = None) -> dict:
    """Collect every grave of a site: georeference, vectors, warnings."""
    graves, problems = collect_graves(sources, vec_dir, recursive)
    if current is not None:
        # The editor's grave replaces its on-disk twin: live edits win.
        key = _norm(current["stem"])
        graves = [g for g in graves if _norm(g["stem"]) != key]
        cg = _new_grave(f"{current['name']} (editor)", current["stem"])
        cg.update(photo=current["photo"], photo_kind="current", current=True,
                  edited_lines=bool(current.get("edited_lines")))
        if cg["edited_lines"]:
            cg["notes"].append("bone lines edited by hand in the Skeleton view — always "
                               "shown and exported")
        graves.insert(0, cg)

    for g in graves:
        g["_rings"], g["_lines"] = [], []
        try:
            georef, w, h = _photo_georef(g, current)
        except Exception as exc:
            g["notes"].append(f"photo unreadable ({exc})")
            georef, w, h = None, 0, 0
        g.update(_georef=georef, w=w, h=h, crs=_crs_key(georef.crs) if georef else None)
        if g["current"]:
            g["_rings"], g["_lines"] = current["rings"], current["lines"]
            g["vec_georef"] = bool(g["_rings"] or g["_lines"])
        else:
            g["vec_georef"] = False
            if g.get("vec"):
                try:
                    r, l, c, ok = load_vectors(g["vec"], georef, g["notes"])
                    g["_rings"], g["_lines"], g["vec_georef"] = r, l, ok
                    g["crs"] = g["crs"] or c
                    if not ok:
                        g["notes"].append("vectors are in pixel space and the photo has no georeference")
                except Exception as exc:
                    g["notes"].append(f"vectors unreadable ({exc})")
            elif georef is not None:
                g["notes"].append("no exported vectors found — only the photo is shown")
        g["georeferenced"] = georef is not None or g["vec_georef"]
        if not g["georeferenced"] and g.get("photo") and not g["notes"]:
            g["notes"].append("not georeferenced — set GCPs in the editor (G)")

    # Raw centre of every placed grave, in its OWN coordinates.
    for g in graves:
        if not g["georeferenced"]:
            continue
        pts = [q for pl in g["_rings"] + g["_lines"] for q in pl]
        if not pts and g["_georef"] is not None:
            pts = [tuple(q) for q in _apply_h(pixel_to_world_matrix(g["_georef"]),
                                               [(0, 0), (g["w"], g["h"])])]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        g["_raw"] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    # Site CRS = most common known CRS. A grave whose GCPs carry no EPSG
    # takes the CRS of the nearest grave with similar raw numbers (same
    # survey grid); with no such grave, the CRS is guessed from the
    # numbers themselves (HTRS96/TM, Gauss-Krüger, UTM ...).
    crs_votes = Counter(g["crs"] for g in graves if g["georeferenced"] and g["crs"])
    site_crs = crs_votes.most_common(1)[0][0] if crs_votes else None
    known = [(g["_raw"], g["crs"]) for g in graves if g["georeferenced"] and g["crs"]]
    for g in graves:
        if not g["georeferenced"] or g["crs"]:
            continue
        near = min(known, key=lambda k: math.dist(k[0], g["_raw"])) if known else None
        if near is not None and math.dist(near[0], g["_raw"]) < 50_000:
            g["crs"] = near[1]
            if g["crs"]:
                g["notes"].append(f"no EPSG set in its GCPs — assumed {g['crs']} "
                                  "like the graves next to it")
        else:
            g["crs"] = _guess_epsg(*g["_raw"]) or site_crs
            if g["crs"]:
                g["notes"].append(f"no EPSG set in its GCPs — {g['crs']} guessed from the "
                                  "coordinates; set the EPSG in the GCP tool to be sure")
    crs_votes = Counter(g["crs"] for g in graves if g["georeferenced"] and g["crs"])
    site_crs = crs_votes.most_common(1)[0][0] if crs_votes else None
    if len(crs_votes) > 1:
        logger.warning("Site view: mixed CRS %s — showing everything in %s via reprojection",
                       dict(crs_votes), site_crs)
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
    for g in graves:
        if "_centre" in g and g["crs"]:
            note = _area_check(g["crs"], *g["_raw"])
            if note:
                g["notes"].insert(0, note)
    frame = _Frame(mode, origin)

    out = []
    for g in graves:
        item = {k: g.get(k) for k in ("name", "stem", "photo", "photo_kind", "photo_affine",
                                      "vec", "notes", "georeferenced", "crs", "current",
                                      "dist_m", "edited_lines")}
        item["n_rings"], item["n_lines"] = len(g["_rings"]), len(g["_lines"])
        if g["georeferenced"]:
            try:
                item["rings"] = frame.polylines_to_client(
                    g["crs"], [r for r in g["_rings"] if len(r) >= 3])
                item["lines"] = frame.polylines_to_client(
                    g["crs"], [l for l in g["_lines"] if len(l) >= 2])
                if g["_georef"] is not None and g["w"]:
                    hd = frame.disp_matrix(pixel_to_world_matrix(g["_georef"]),
                                           g["w"], g["h"], g["crs"])
                    fp = _apply_h(hd, [(0, 0), (g["w"], 0), (g["w"], g["h"]), (0, g["h"])])
                    item["footprint"] = frame.disp_to_client(fp[:, 0], fp[:, 1])
                    bx0, by0, bx1, by1 = _grid_bbox(*_grid(hd, g["w"], g["h"], PHOTO_TILE_DIM))
                    item["bounds"] = frame.disp_to_client([bx0, bx1], [by0, by1])
            except Exception as exc:
                logger.exception("Site view: could not place %s", g["name"])
                item["notes"].append(f"could not place on the map ({exc})")
                item["georeferenced"] = False
        out.append(item)

    logger.info("Site view scan: %d graves (%d placed), mode=%s, crs=%s",
                len(out), sum(1 for g in out if g["georeferenced"]), mode, site_crs)
    return {"graves": out, "mode": mode, "crs": site_crs, "origin": list(origin),
            "problems": problems}


def _area_check(crs: str, x: float, y: float) -> str | None:
    """Warn when a grave lies outside the area its CRS is made for — the
    sign of numbers recorded in one system and labelled as another."""
    try:
        from pyproj import CRS
        from boneseg.data.crs_guess import TM0_NAME, tm0_to_htrs96
        area = CRS.from_user_input(crs).area_of_use
        (lon,), (lat,) = _transform_xy(crs, "EPSG:4326", [x], [y])
    except Exception:
        return None
    if area is None:
        return None
    w, s_, e, n = area.bounds
    if w - 1 <= lon <= e + 1 and s_ - 1 <= lat <= n + 1:
        return None
    msg = (f"lands at {lat:.2f}°N {lon:.2f}°E — outside the area of {crs} "
           f"({area.name}): the points are not in {crs}")
    if tm0_to_htrs96([{"e": x, "n": y}]) is not None:
        msg += (f"; they look like a {TM0_NAME} — reload the point file in the "
                "GCP tool (it converts them now) and apply the GCPs again")
    return msg


def _guess_epsg(x: float, y: float) -> str | None:
    try:
        from boneseg.data.crs_guess import guess_crs
        res = guess_crs([{"e": x, "n": y}])
    except Exception:
        return None
    # "ambiguous" still names the preferred candidate (HTRS96/TM before the
    # UTM zones whose numbers overlap) — the same pre-fill the GCP tool uses.
    if res.get("confidence") in ("high", "ambiguous") and not res.get("swap") and res.get("epsg"):
        return f"EPSG:{res['epsg']}"
    return None


# --------------------------------------------------------------------------- #
# Photo tiles for the map                                                      #
# --------------------------------------------------------------------------- #
def render_photo(g: dict, mode: str, origin: tuple[float, float],
                 current: dict | None = None, max_dim: int = PHOTO_TILE_DIM) -> tuple[bytes, list]:
    """The photo warped onto the display grid of its own footprint bbox,
    as WebP with transparency outside the photo. Returns (bytes, bounds)
    with bounds as Leaflet [[lat, lng] SW, [lat, lng] NE]."""
    georef, w, h = _photo_georef(g, current)
    if georef is None:
        raise ValueError("photo has no georeference")
    if g.get("current") and current is not None:
        key = ("current", current["key"], mode, tuple(origin), g.get("crs"), max_dim)
    else:
        path = Path(g["photo"])
        sidecar = path.with_suffix(path.suffix + ".gcps.json")
        key = (str(path), path.stat().st_mtime,
               sidecar.stat().st_mtime if sidecar.exists() else 0, mode, tuple(origin),
               g.get("crs"), tuple(g.get("photo_affine") or ()), max_dim)
    if key in _photo_cache:
        return _photo_cache[key]

    crs = _crs_key(georef.crs) or g.get("crs")
    frame = _Frame(mode, origin)
    hd = frame.disp_matrix(pixel_to_world_matrix(georef), w, h, crs)
    img = _pixels(g, current, max(max_dim, 1) * 3 // 2)
    rgba, (x0, y0, x1, y1) = _warp_to_grid(img, hd, w, h, max_dim)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "WEBP", quality=82, method=3)
    res = (buf.getvalue(), frame.disp_to_client([x0, x1], [y0, y1]))
    if len(_photo_cache) >= _PHOTO_CACHE_MAX:
        _photo_cache.pop(next(iter(_photo_cache)))
    _photo_cache[key] = res
    return res


# --------------------------------------------------------------------------- #
# Site exports: DXF (CAD), GeoPackage + GeoTIFFs (GIS), PDF plan              #
# --------------------------------------------------------------------------- #
_ACI_COLORS = (1, 3, 4, 5, 6, 2, 30, 140, 40, 210, 92, 172)
_HEX_COLORS = ("#f97316", "#22d3ee", "#a3e635", "#f472b6", "#facc15", "#60a5fa",
               "#34d399", "#c084fc", "#fb7185", "#2dd4bf", "#fbbf24", "#818cf8")


def _layer_name(name: str, suffix: str) -> str:
    clean = re.sub(r'[<>/\\":;?*|,=`]+', "-", name).strip(" .-") or "GROB"
    return f"{clean}_{suffix}"


def _safe_file(name: str, i: int) -> str:
    return (re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or f"G{i}").upper()


def _site_graves(graves: list[dict], site_crs: str | None, current: dict | None,
                 want_photos: bool, photo_max_dim: int) -> list[dict]:
    """Every exportable grave in SITE coordinates: rings, lines and (when
    asked) the rectified photo as RGBA + bbox."""
    out = []
    for i, g in enumerate(graves):
        g = {**g, "notes": []}
        name = g.get("stem") if g.get("current") else g.get("name")
        name = name or g.get("stem") or f"GROB{i + 1}"
        try:
            georef, w, h = _photo_georef(g, current)
        except Exception:
            georef, w, h = None, 0, 0
        crs = (_crs_key(georef.crs) if georef else None) or g.get("crs") or site_crs
        rings: list[Polyline] = []
        lines: list[Polyline] = []
        if g.get("current") and current is not None:
            rings, lines = current["rings"], current["lines"]
        elif g.get("vec"):
            try:
                rings, lines, _c, ok = load_vectors(g["vec"], georef, g["notes"])
                if not ok:
                    rings, lines = [], []
            except Exception as exc:
                logger.warning("Site export: vectors of %s unreadable (%s)", name, exc)
        if georef is None and not rings and not lines:
            continue

        def to_site(pls: list[Polyline]) -> list[Polyline]:
            if not site_crs or crs == site_crs or not pls:
                return pls
            xs, ys = _transform_xy(crs, site_crs, [q[0] for pl in pls for q in pl],
                                   [q[1] for pl in pls for q in pl])
            res, k = [], 0
            for pl in pls:
                res.append(list(zip(xs[k:k + len(pl)], ys[k:k + len(pl)])))
                k += len(pl)
            return res

        rec = {"name": name, "index": i, "rings": to_site(rings), "lines": to_site(lines),
               "edited_lines": bool(g.get("edited_lines")),
               "photo": None, "footprint": None}
        if georef is not None and w:
            hs = _to_crs_matrix(pixel_to_world_matrix(georef), w, h, crs, site_crs)
            rec["footprint"] = _apply_h(hs, [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]).tolist()
            if want_photos:
                try:
                    img = _pixels(g, current, photo_max_dim)
                    rec["photo"] = _warp_to_grid(img, hs, w, h, photo_max_dim)
                except Exception:
                    logger.exception("Site export: photo of %s failed", name)
        out.append(rec)
    if not out:
        raise ValueError("None of the selected graves is georeferenced — nothing to export.")
    return out


def _write_dxf(recs: list[dict], out: Path, want_rings: bool, want_lines: bool) -> Path:
    import ezdxf

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    photo_dir = out.parent / f"{out.stem}_photos"
    if any(r["photo"] is not None for r in recs):
        doc.set_raster_variables(frame=0, quality=1, units="m")
    for r in recs:
        color = _ACI_COLORS[r["index"] % len(_ACI_COLORS)]
        if r["photo"] is not None:
            rgba, (x0, y0, x1, y1) = r["photo"]
            oh, ow = rgba.shape[:2]
            gsd = (x1 - x0) / ow
            photo_dir.mkdir(exist_ok=True)
            png = photo_dir / f"{_safe_file(r['name'], r['index'])}.png"
            Image.fromarray(rgba, "RGBA").save(png, optimize=False, compress_level=6)
            lp = _layer_name(r["name"], "PHOTO")
            if lp not in doc.layers:
                doc.layers.add(lp, color=8)
            idef = doc.add_image_def(filename=str(png.resolve()), size_in_pixel=(ow, oh),
                                     name=f"{_safe_file(r['name'], r['index'])}_PHOTO")
            ent = msp.add_image(image_def=idef, insert=(x0, y1 - oh * gsd, 0.0),
                                size_in_units=(ow * gsd, oh * gsd), dxfattribs={"layer": lp})
            ent.dxf.u_pixel = (gsd, 0.0, 0.0)
            ent.dxf.v_pixel = (0.0, gsd, 0.0)
            # flag 8 = transparency on: the empty corners of the rectified
            # photo (PNG alpha) don't cover the neighbouring graves.
            ent.dxf.flags = ent.dxf.flags | 8
    for r in recs:
        color = _ACI_COLORS[r["index"] % len(_ACI_COLORS)]
        lo, lc = _layer_name(r["name"], "MASK_EDGE"), _layer_name(r["name"], "LINES")
        r_lines = want_lines or r["edited_lines"]
        for ln, want in ((lo, want_rings), (lc, r_lines)):
            if want and ln not in doc.layers:
                doc.layers.add(ln, color=color)
        if want_rings:
            for ring in r["rings"]:
                if len(ring) >= 3:
                    msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": lo})
        if r_lines:
            for line in r["lines"]:
                if len(line) >= 2:
                    msp.add_lwpolyline(line, close=False, dxfattribs={"layer": lc})
    doc.saveas(str(out))
    return out


def _write_gpkg(recs: list[dict], out: Path, site_crs: str | None,
                want_rings: bool, want_lines: bool) -> list[Path]:
    """GeoPackage with three layers (``bone_lines``, ``mask_edges``,
    ``photo_footprints``; attribute ``grave``) + one GeoTIFF per photo."""
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    if out.exists():
        out.unlink()
    written = [out]
    layers = []
    if want_rings:
        layers.append(("mask_edges", [(r["name"], Polygon(ring)) for r in recs
                                     for ring in r["rings"] if len(ring) >= 4]))
    lines_rows = [(r["name"], LineString(l)) for r in recs
                  if want_lines or r["edited_lines"] for l in r["lines"] if len(l) >= 2]
    if lines_rows:
        layers.append(("bone_lines", lines_rows))
    layers.append(("photo_footprints", [(r["name"], Polygon(r["footprint"])) for r in recs
                                         if r["footprint"]]))
    n = 0
    for lname, rows in layers:
        if not rows:
            continue
        gpd.GeoDataFrame({"grave": [a for a, _ in rows]}, geometry=[b for _, b in rows],
                         crs=site_crs).to_file(out, layer=lname, driver="GPKG")
        n += 1
    if n == 0:
        raise ValueError("Nothing to write to the GeoPackage.")

    photo_dir = out.parent / f"{out.stem}_photos"
    for r in recs:
        if r["photo"] is None:
            continue
        rgba, (x0, y0, x1, y1) = r["photo"]
        oh, ow = rgba.shape[:2]
        gsd = (x1 - x0) / ow
        photo_dir.mkdir(exist_ok=True)
        tif = photo_dir / f"{_safe_file(r['name'], r['index'])}.tif"
        with rasterio.open(str(tif), "w", driver="GTiff", width=ow, height=oh, count=4,
                           dtype="uint8", crs=site_crs,
                           transform=Affine(gsd, 0, x0, 0, -gsd, y1),
                           photometric="RGB", alpha="YES", compress="deflate",
                           tiled=True, blockxsize=256, blockysize=256) as dst:
            for b in range(4):
                dst.write(rgba[..., b], b + 1)
        written.append(tif)
    return written


# Standard plan scales, ascending.
_PLAN_SCALES = (20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000)
_BAR_LENGTHS = (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)


def _write_pdf(recs: list[dict], out: Path, site_crs: str | None, title: str,
               want_rings: bool, want_lines: bool, labels: bool) -> Path:
    """One-page A3 landscape site plan at an exact standard scale."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    page_w, page_h, margin, head, foot = 420.0, 297.0, 12.0, 14.0, 30.0
    map_w, map_h = page_w - 2 * margin, page_h - 2 * margin - head - foot

    xs, ys = [], []
    for r in recs:
        for pl in r["rings"] + r["lines"] + ([r["footprint"]] if r["footprint"] else []):
            xs += [q[0] for q in pl]
            ys += [q[1] for q in pl]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ext_w, ext_h = max(x1 - x0, 0.5) * 1.03, max(y1 - y0, 0.5) * 1.03
    scale = next((s for s in _PLAN_SCALES
                  if ext_w * 1000 / s <= map_w and ext_h * 1000 / s <= map_h), _PLAN_SCALES[-1])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = map_w * scale / 2000, map_h * scale / 2000

    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    page = fig.add_axes([0, 0, 1, 1])
    page.set_xlim(0, page_w)
    page.set_ylim(0, page_h)
    page.axis("off")
    ax = fig.add_axes([margin / page_w, (margin + foot) / page_h, map_w / page_w, map_h / page_h])
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, length=2)
    ax.ticklabel_format(useOffset=False, style="plain")
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va("center")

    with_photos = any(r["photo"] is not None for r in recs)
    for r in recs:
        if r["photo"] is not None:
            rgba, (px0, py0, px1, py1) = r["photo"]
            ax.imshow(rgba, extent=(px0, px1, py0, py1), interpolation="bilinear", zorder=1)
    lw = 0.35 if scale <= 100 else 0.25
    for r in recs:
        color = _HEX_COLORS[r["index"] % len(_HEX_COLORS)] if with_photos else "black"
        if want_rings:   # mask edges: a checking aid, drawn thin and grey
            for ring in r["rings"]:
                a = np.asarray(ring)
                ax.plot(a[:, 0], a[:, 1], color="#888", lw=lw * 0.5, zorder=2)
        if want_lines or r["edited_lines"]:
            # The bone lines ARE the drawing (BoneSeg's Skeleton view).
            for line in r["lines"]:
                a = np.asarray(line)
                ax.plot(a[:, 0], a[:, 1], color=color, lw=lw, zorder=3)
        if not with_photos and r["footprint"] and not (r["rings"] or r["lines"]):
            a = np.asarray(r["footprint"])
            ax.plot(a[:, 0], a[:, 1], color="#999", lw=0.3, ls="--", zorder=2)
        if labels:
            pts = np.asarray([q for pl in r["rings"] + r["lines"] for q in pl]
                             or r["footprint"] or [[cx, cy]])
            ax.text(pts[:, 0].mean(), pts[:, 1].max(), r["name"], fontsize=6, ha="center",
                    va="bottom", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Header
    page.text(margin, page_h - margin - 6, title or "Site plan", fontsize=14, fontweight="bold")
    page.text(page_w - margin, page_h - margin - 6, f"{len(recs)} graves", fontsize=9,
              ha="right", color="#444")
    # Footer: scale bar, north arrow, metadata
    bar = max(b for b in _BAR_LENGTHS if b * 1000 / scale <= map_w * 0.3) \
        if any(b * 1000 / scale <= map_w * 0.3 for b in _BAR_LENGTHS) else _BAR_LENGTHS[0]
    bar_mm = bar * 1000 / scale
    bx, by = margin, margin + 7
    for k in range(4):
        page.add_patch(Rectangle((bx + k * bar_mm / 4, by), bar_mm / 4, 2,
                                 fc="black" if k % 2 == 0 else "white", ec="black", lw=0.5))
    page.text(bx, by + 3.2, "0", fontsize=7, ha="center")
    page.text(bx + bar_mm, by + 3.2, f"{bar:g} m", fontsize=7, ha="center")
    page.text(bx + bar_mm / 2, by - 5, f"M 1:{scale}", fontsize=9, ha="center", fontweight="bold")
    nx, ny = bx + bar_mm + 18, by - 4
    page.annotate("", xy=(nx, ny + 12), xytext=(nx, ny),
                  arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", mutation_scale=14))
    page.text(nx, ny + 15, "N", fontsize=10, ha="center", fontweight="bold")
    meta = [f"Coordinate system: {site_crs or 'local (no EPSG)'}",
            f"Vectors: {', '.join(v for v, w in (('bone lines', want_lines), ('mask edges', want_rings)) if w) or '—'}"
            + ("; photos rectified" if with_photos else ""),
            f"BoneSeg Studio · {date.today().isoformat()}"]
    for k, line in enumerate(meta):
        page.text(page_w - margin, margin + 14 - k * 4.5, line, fontsize=7, ha="right", color="#333")

    fig.savefig(out, dpi=300 if with_photos else 150)
    plt.close(fig)
    return out


def export_site(graves: list[dict], out_base: str, site_crs: str | None,
                formats: list[str], current: dict | None = None,
                photos: bool = True, outlines: bool = False, centerlines: bool = True,
                labels: bool = True, title: str = "") -> dict:
    """Write the chosen site exports next to ``out_base`` (extension ignored):
    ``.dxf`` (CAD), ``.gpkg`` + GeoTIFFs (GIS), ``.pdf`` (A3 plan)."""
    base = Path(out_base)
    if base.suffix.lower() in (".dxf", ".gpkg", ".pdf"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    if not (outlines or centerlines or photos):
        raise ValueError("Nothing selected: turn on bone lines, mask edges or photos.")
    formats = [f for f in formats if f in ("dxf", "gpkg", "pdf")]
    if not formats:
        raise ValueError("Choose at least one export format.")

    written: list[str] = []
    n_graves = 0
    # DXF/GIS get the photos at a CAD/GIS-friendly resolution; the PDF plan
    # needs less (a 3000 px photo of a 2 m grave is 0.7 mm per pixel).
    if "dxf" in formats or "gpkg" in formats:
        recs = _site_graves(graves, site_crs, current, photos, 3000)
        n_graves = len(recs)
        if "dxf" in formats:
            written.append(str(_write_dxf(recs, Path(f"{base}.dxf"), outlines, centerlines)))
        if "gpkg" in formats:
            written += [str(p) for p in _write_gpkg(recs, Path(f"{base}.gpkg"), site_crs,
                                                    outlines, centerlines)]
    if "pdf" in formats:
        recs = _site_graves(graves, site_crs, current, photos, 1600)
        n_graves = len(recs)
        written.append(str(_write_pdf(recs, Path(f"{base}.pdf"), site_crs, title,
                                      outlines, centerlines, labels)))
    logger.info("Site export: %d graves -> %s", n_graves, ", ".join(written))
    return {"files": written, "graves": n_graves, "folder": str(base.parent)}
