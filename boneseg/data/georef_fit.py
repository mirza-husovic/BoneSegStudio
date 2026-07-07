"""Ground-control-point georeferencing: fit an affine transform from
clicked image points + surveyed world coordinates.

Scope note (accuracy): an affine transform is EXACT for orthophotos and
flat, nadir shots. Hand-held oblique photos have perspective distortion an
affine cannot model — residuals reported back to the UI make that visible
(a few cm on a nadir grave photo is normal; tens of cm means the photo is
oblique or a point was mis-entered/swapped).

Coordinates follow the survey convention the user's total-station files
use: E (easting) and N (northing) in a projected CRS in metres — the same
numbers AutoCAD shows. AutoCAD itself is CRS-agnostic, so the EPSG code is
supplied by the user (HTRS96/TM = EPSG:3765 for Croatia); with it the same
coordinates load correctly in QGIS, without it exports still carry the
metric coordinates and CAD keeps working.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from boneseg.data.readers import HAS_RASTERIO, GeoRef
from boneseg.logging_setup import get_logger

if HAS_RASTERIO:
    import rasterio
    from rasterio.transform import Affine

logger = get_logger(__name__)

# World-file extension per image suffix (QGIS/ArcGIS sidecar convention).
WORLD_EXT = {".jpg": ".jgw", ".jpeg": ".jgw", ".png": ".pgw",
             ".tif": ".tfw", ".tiff": ".tfw"}


def fit_affine_gcps(
    gcps: list[dict],
) -> tuple["Affine", list[float], float]:
    """Least-squares affine from ≥3 GCPs.

    Each gcp: {"px": col, "py": row, "e": easting, "n": northing} in
    FULL-resolution pixel coords. Returns (transform, per-point residuals
    in metres, RMS). Exact for 3 points; overdetermined beyond.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for georeferencing")
    if len(gcps) < 3:
        raise ValueError("At least 3 control points are required")

    cols = np.array([float(g["px"]) for g in gcps])
    rows = np.array([float(g["py"]) for g in gcps])
    es = np.array([float(g["e"]) for g in gcps])
    ns = np.array([float(g["n"]) for g in gcps])

    m = np.column_stack([cols, rows, np.ones_like(cols)])
    # x = a*col + b*row + c ; y = d*col + e*row + f
    coef_e = np.linalg.lstsq(m, es, rcond=None)[0]
    coef_n = np.linalg.lstsq(m, ns, rcond=None)[0]
    transform = Affine(coef_e[0], coef_e[1], coef_e[2],
                       coef_n[0], coef_n[1], coef_n[2])

    fit_e = m @ coef_e
    fit_n = m @ coef_n
    residuals = [float(math.hypot(fe - e, fn - n))
                 for fe, e, fn, n in zip(fit_e, es, fit_n, ns)]
    rms = float(math.sqrt(sum(r * r for r in residuals) / len(residuals)))
    return transform, residuals, rms


def georef_from_gcps(gcps: list[dict], epsg: int | None) -> tuple[GeoRef, list[float], float]:
    """Fit + wrap in the pipeline's GeoRef (CRS optional = local grid)."""
    transform, residuals, rms = fit_affine_gcps(gcps)
    crs = rasterio.crs.CRS.from_epsg(int(epsg)) if epsg else None
    return GeoRef(transform, crs), residuals, rms


def write_world_file(image_path: Path, georef: GeoRef) -> list[Path]:
    """Write the ESRI world file (+ .prj when the CRS is known) next to the
    photo, so QGIS/ArcGIS open the ORIGINAL image already georeferenced.

    World file lines: A (x per col), D (y per col), B (x per row),
    E (y per row), C, F — where C/F are the coordinates of the CENTER of
    the top-left pixel (hence the half-pixel shift from the affine).
    """
    t = georef.transform
    cx, cy = t * (0.5, 0.5)
    ext = WORLD_EXT.get(image_path.suffix.lower(), ".wld")
    world = image_path.with_suffix(ext)
    world.write_text(
        "\n".join(f"{v:.10f}" for v in (t.a, t.d, t.b, t.e, cx, cy)) + "\n",
        encoding="ascii")
    written = [world]
    if georef.crs is not None:
        prj = image_path.with_suffix(".prj")
        prj.write_text(georef.crs.to_wkt(), encoding="ascii")
        written.append(prj)
    logger.info("World file written: %s", ", ".join(p.name for p in written))
    return written


def gcps_sidecar_path(image_path: Path) -> Path:
    """Sidecar JSON that lets the app re-apply GCPs on every reopen."""
    return image_path.with_suffix(image_path.suffix + ".gcps.json")
