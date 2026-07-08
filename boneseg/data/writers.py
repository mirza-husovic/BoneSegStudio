"""Output writers: binary mask raster, overlay PNG, skeleton PNG, GeoJSON, SVG.

Coordinate conventions (identical to predict.py so outputs drop into the
user's existing QGIS workflow):

  * Georeferenced input  -> GeoJSON in the input CRS (via the affine transform).
  * Plain image          -> GeoJSON in pixel coordinates with Y negated, which
                            matches where QGIS draws an ungeoreferenced raster
                            (world Y = 0..-H).
  * SVG is a screen format: always raw pixel coordinates, Y down, with a
    viewBox equal to the image size so it overlays the source image 1:1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from boneseg.data.readers import HAS_RASTERIO, GeoRef
from boneseg.logging_setup import get_logger

if HAS_RASTERIO:
    import rasterio

try:
    import ezdxf

    HAS_EZDXF = True
except Exception:  # pragma: no cover - optional dependency
    HAS_EZDXF = False

logger = get_logger(__name__)

Polyline = list[tuple[float, float]]


# ---------------------------------------------------------------------------
# Raster outputs
# ---------------------------------------------------------------------------
def save_mask_raster(mask01: np.ndarray, stem_path: Path, georef: GeoRef | None) -> Path:
    """Write the binary mask as {0,255} PNG, or geo-tagged GTiff if input was.

    Perspective georefs (oblique photo, GeoRef.homography) are warped
    (rectified) onto the north-up grid — a GeoTIFF can only carry an
    affine, and writing the sheared affine is exactly the "stretched"
    look this feature removes.

    ``stem_path`` is the output path without extension; the correct suffix
    is chosen here and the final path returned.
    """
    mask255 = (mask01 * 255).astype(np.uint8)
    if HAS_RASTERIO and georef is not None:
        transform = georef.transform
        if georef.homography is not None:
            from boneseg.data.georef_fit import rectify_params
            h, w = mask255.shape
            transform, k, out_w, out_h = rectify_params(georef, w, h)
            mask255 = cv2.warpPerspective(mask255, k, (out_w, out_h),
                                          flags=cv2.INTER_NEAREST)
        out = stem_path.with_suffix(".tif")
        with rasterio.open(
            str(out), "w",
            driver="GTiff",
            height=mask255.shape[0],
            width=mask255.shape[1],
            count=1,
            dtype="uint8",
            crs=georef.crs,
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(mask255, 1)
        return out
    out = stem_path.with_suffix(".png")
    Image.fromarray(mask255).save(out)
    return out


def rectify_photo(img_rgb: np.ndarray, georef: GeoRef,
                  ) -> tuple[np.ndarray, np.ndarray, "rasterio.Affine"]:
    """Warp the photo onto the north-up grid (perspective correction).

    Returns ``(rectified_rgb, valid_mask, grid_transform)``. ``valid``
    marks pixels covered by the source photo — the warped footprint is a
    quadrilateral, the rest of the grid is empty.
    """
    from boneseg.data.georef_fit import rectify_params
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    h, w = img_rgb.shape[:2]
    grid, k, out_w, out_h = rectify_params(georef, w, h)
    rect = cv2.warpPerspective(img_rgb, k, (out_w, out_h),
                               flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), k,
                                (out_w, out_h), flags=cv2.INTER_NEAREST)
    return rect, valid, grid


def save_photo_geotiff(img_rgb: np.ndarray, out_path: Path,
                       georef: GeoRef | None) -> Path | None:
    """Write the PHOTO itself as a georeferenced GeoTIFF.

    Unlike the world-file sidecar (which georeferences the original JPEG
    but only as long as the .jgw/.prj travel with it), a GeoTIFF is a
    single self-contained file — CRS and transform embedded — that any
    GIS opens correctly on its own.

    Perspective georefs (oblique photo) are RECTIFIED first: the photo is
    warped onto a north-up grid through the exact homography, so instead
    of the sheared/stretched affine look the GIS shows a flat, true-scale
    image (the uncovered corners of the grid are masked out).

    JPEG-in-TIFF compression keeps a 24 MP photo at roughly camera-JPEG
    size (a plain deflate GeoTIFF of the same photo is ~5× larger); tiling
    plus overviews make QGIS pan/zoom instant. Falls back to deflate if
    the local GDAL lacks JPEG support. Returns None (with a log) when
    there is no georeference — a pixel-space GeoTIFF would be meaningless.
    """
    if georef is None:
        logger.warning("GeoTIFF photo skipped: no georeference (set GCPs first)")
        return None
    if not HAS_RASTERIO:
        logger.warning("GeoTIFF photo skipped: rasterio not installed")
        return None
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)

    transform = georef.transform
    valid: np.ndarray | None = None
    if georef.homography is not None:
        img_rgb, valid, transform = rectify_photo(img_rgb, georef)

    h, w = img_rgb.shape[:2]
    base = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                crs=georef.crs, transform=transform,
                tiled=True, blockxsize=256, blockysize=256)
    attempts = (dict(compress="jpeg", photometric="ycbcr", jpeg_quality=92),
                dict(compress="deflate", predictor=2))
    last_exc: Exception | None = None
    for extra in attempts:
        try:
            env = rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True) if valid is not None \
                else rasterio.Env()
            with env, rasterio.open(str(out_path), "w", **base, **extra) as dst:
                for b in range(3):
                    dst.write(img_rgb[..., b], b + 1)
                if valid is not None:
                    try:
                        dst.write_mask(valid)
                    except Exception:
                        logger.warning("GeoTIFF validity mask not written "
                                       "(edges will show black)")
                factors = [f for f in (2, 4, 8, 16) if max(h, w) // f >= 512]
                if factors:
                    dst.build_overviews(
                        factors, rasterio.enums.Resampling.average)
            logger.info("GeoTIFF photo written: %s (%s%s)", out_path.name,
                        extra["compress"],
                        ", rectified" if georef.homography is not None else "")
            return out_path
        except Exception as exc:  # e.g. GDAL built without JPEG
            last_exc = exc
            logger.warning("GeoTIFF write with %s failed (%s) — trying next",
                           extra["compress"], exc)
    raise RuntimeError(f"GeoTIFF photo export failed: {last_exc}")


def render_overlay(
    img_rgb: np.ndarray,
    mask01: np.ndarray,
    opacity: float,
    color: tuple[int, int, int] = (0, 200, 220),
) -> np.ndarray:
    """Blend ``color`` over foreground pixels; ``opacity`` in [0, 1].

    Generalizes predict.py's fixed 0.55 blend to an adjustable one.
    Pure function — also used live by the UI opacity slider.
    """
    overlay = img_rgb.copy()
    fg = mask01 > 0
    if fg.any() and opacity > 0:
        tint = np.array(color, dtype=np.float32)
        base = overlay[fg].astype(np.float32)
        overlay[fg] = ((1.0 - opacity) * base + opacity * tint).astype(np.uint8)
    return overlay


def save_overlay_png(overlay_rgb: np.ndarray, out_path: Path) -> Path:
    """Write an already-rendered overlay image."""
    Image.fromarray(overlay_rgb).save(out_path)
    return out_path


def save_skeleton_png(skeleton: np.ndarray, out_path: Path) -> Path:
    """Write the skeleton as white-on-black PNG (bool or {0,1} input)."""
    Image.fromarray((skeleton > 0).astype(np.uint8) * 255).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Vector outputs
# ---------------------------------------------------------------------------
def save_geojson(
    polylines: Sequence[Polyline],
    out_path: Path,
    georef: GeoRef | None,
) -> Path:
    """Write centerline polylines as a GeoJSON FeatureCollection.

    Coordinates must already be in output space (geo or QGIS-pixel); this
    function only serializes.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in line],
            },
            "properties": {"id": i},
        }
        for i, line in enumerate(polylines)
        if len(line) >= 2
    ]
    gj: dict = {"type": "FeatureCollection", "features": features}
    crs_name = str(georef.crs) if (georef is not None and georef.crs) else "pixel"
    gj["crs"] = {"type": "name", "properties": {"name": crs_name}}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    return out_path


def _embed_dxf_photo(doc, msp, image_rgb: np.ndarray,
                     georef: GeoRef | None, dxf_path: Path) -> None:
    """Attach the photo as a DXF IMAGE entity on layer PHOTO, so ANY CAD
    (plain AutoCAD/LT included — no Map 3D needed) opens the drawing with
    the photo already sitting under the vectors.

    Georeferenced photos are RECTIFIED (north-up axis-aligned grid) before
    embedding — for two reasons: (1) AutoCAD refuses to display rasters
    with sheared/rotated u,v vectors (the frame shows, the image doesn't),
    and every least-squares affine carries some shear; (2) for oblique
    photos the rectified image is the geometrically correct one. The
    IMAGE entity then uses plain axis-aligned per-pixel vectors, which
    every CAD renders.

    The JPEG is written NEXT TO the DXF and referenced by bare filename,
    keeping the export folder self-contained. Without a georeference the
    original photo goes to insert (0, -h) with unit u/v (QGIS-pixel
    convention — same as the vectors).
    """
    target = dxf_path.parent / f"{dxf_path.stem}_photo.jpg"
    if georef is not None:
        rect, _valid, grid = rectify_photo(image_rgb, georef)
        h, w = rect.shape[:2]
        Image.fromarray(rect).save(target, "JPEG", quality=90)
        # North-up grid: lower-left corner, axis-aligned metre pixels.
        ix, iy = grid * (0.0, float(h))
        u = (grid.a, 0.0)
        v = (0.0, -grid.e)
    else:
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb] * 3, axis=-1)
        h, w = image_rgb.shape[:2]
        Image.fromarray(image_rgb).save(target, "JPEG", quality=90)
        ix, iy = 0.0, -float(h)
        u, v = (1.0, 0.0), (0.0, 1.0)

    doc.layers.add("PHOTO", color=8)  # grey
    image_def = doc.add_image_def(filename=target.name, size_in_pixel=(w, h))
    img = msp.add_image(
        image_def=image_def, insert=(ix, iy, 0.0),
        size_in_units=(w * math.hypot(*u), h * math.hypot(*v)),
        dxfattribs={"layer": "PHOTO"})
    img.dxf.u_pixel = (u[0], u[1], 0.0)
    img.dxf.v_pixel = (v[0], v[1], 0.0)
    logger.info("Photo embedded in DXF: %s (%dx%d px%s)", target.name, w, h,
                ", rectified" if georef is not None else "")


def save_dxf(
    rings: Sequence[Polyline],
    centerlines: Sequence[Polyline],
    out_path: Path,
    georef: GeoRef | None,
    image_rgb: np.ndarray | None = None,
) -> Path | None:
    """Write outlines + centerlines as a DXF for direct CAD ingestion.

    Same layer convention as predict.py:
      BONE_OUTLINE    — closed LWPOLYLINEs (predicted outline polygons)
      BONE_CENTERLINE — open LWPOLYLINEs (smoothed medial-axis lines)
      PHOTO           — the photo as an IMAGE underlay (optional, when
                        ``image_rgb`` is given; rectified when georeferenced)

    Coordinates are identical to the GeoJSON outputs (input CRS when
    georeferenced, else QGIS-pixel with Y negated). AutoCAD/BricsCAD open
    the R2010 DXF natively and can save it as DWG.

    Returns None if ezdxf is not installed (export silently skipped).
    """
    if not HAS_EZDXF:
        logger.warning("ezdxf not installed — DXF export skipped")
        return None

    doc = ezdxf.new(dxfversion="R2010")
    doc.layers.add("BONE_OUTLINE", color=3)      # green
    doc.layers.add("BONE_CENTERLINE", color=4)   # cyan
    doc.header["$INSUNITS"] = 6 if georef is not None else 0  # 6 = meters
    msp = doc.modelspace()

    # Image first: entities draw in file order, so the photo stays UNDER
    # the vectors in viewers that ignore draw-order tables.
    if image_rgb is not None:
        try:
            _embed_dxf_photo(doc, msp, image_rgb, georef, out_path)
        except Exception:
            logger.exception("DXF photo underlay failed (vectors unaffected)")

    for ring in rings:
        if len(ring) >= 3:
            msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": "BONE_OUTLINE"})
    for line in centerlines:
        if len(line) >= 2:
            msp.add_lwpolyline(line, close=False, dxfattribs={"layer": "BONE_CENTERLINE"})

    doc.saveas(str(out_path))
    return out_path


def save_svg(
    polylines: Sequence[Polyline],
    out_path: Path,
    width: int,
    height: int,
    stroke: str = "#111111",
    stroke_width: float = 1.5,
) -> Path:
    """Write centerlines as an SVG sized to the source image.

    Expects PIXEL coordinates (Y down). The viewBox matches the image, so
    the drawing overlays the photo 1:1 in Illustrator/Inkscape.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<g fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">',
    ]
    for line in polylines:
        if len(line) < 2:
            continue
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
        parts.append(f'<polyline points="{pts}"/>')
    parts.append("</g></svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
