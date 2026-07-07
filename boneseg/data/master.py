"""Cumulative "master" export — many graves, one GIS/CAD file.

Supports the one-image-at-a-time workflow: every exported grave is merged
into a single pair of GeoJSONs plus a single DXF in the output root, so a
whole site (e.g. 30 graves) opens as ONE file in QGIS or AutoCAD.

Files maintained in the master directory:

  master_centerlines.geojson   all centerlines, one feature per line,
                               ``properties.image`` = source image stem
                               (GeoJSON has no layers — filter/categorize
                               by the ``image`` attribute in QGIS)
  master_outlines.geojson      all outline polygons, same convention
  master.gpkg                  GeoPackage: TRUE per-grave layers
                               ``<stem>_outlines`` / ``<stem>_centerlines``
                               (requires geopandas; skipped otherwise)
  master.dxf                   one CAD file; per-grave layers
                               ``<STEM>_OUTLINE`` / ``<STEM>_CENTERLINE``
  master_layout.json           bookkeeping sidecar (mode, per-image offsets)

Semantics:

  * **Replace, not duplicate** — re-exporting the same image (e.g. after a
    threshold change) replaces its features/layers instead of appending a
    second copy.
  * **Georeferenced inputs** keep world coordinates: graves land at their
    true site positions and the master IS the site plan.
  * **Plain (pixel) images** have no shared frame — every grave would pile
    up at the origin. They are laid out in a horizontal strip with padding
    (offsets remembered in the sidecar, stable across re-exports). Useful
    for CAD editing, but positions are NOT real-world.
  * Mixing georeferenced and plain images in one master is refused.

The GeoJSONs are the source of truth; ``master.dxf`` is regenerated from
them on every append (avoids fragile in-place DXF entity surgery).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from boneseg.data.readers import GeoRef
from boneseg.data.writers import HAS_EZDXF
from boneseg.logging_setup import get_logger

if HAS_EZDXF:
    import ezdxf

logger = get_logger(__name__)

Polyline = list[tuple[float, float]]

MASTER_CENTERLINES = "master_centerlines.geojson"
MASTER_OUTLINES = "master_outlines.geojson"
MASTER_DXF = "master.dxf"
MASTER_GPKG = "master.gpkg"
MASTER_LAYOUT = "master_layout.json"

PIXEL_STRIP_PAD = 200.0  # px gap between graves in the pixel-mode strip

# ACI colors cycled per grave so layers are distinguishable in CAD
_LAYER_COLORS = (1, 3, 4, 5, 6, 2, 30, 140, 40, 210)


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Corrupt %s (%s) — starting fresh", path.name, exc)
        return default


def _layer_name(stem: str, suffix: str) -> str:
    """DXF-safe layer name: ``GR 23a`` -> ``GR_23A_OUTLINE``."""
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").upper() or "IMAGE"
    return f"{clean}_{suffix}"


def _merge_geojson(
    path: Path,
    stem: str,
    geoms: list[Polyline],
    geom_type: str,
    crs_name: str,
) -> Path:
    """Replace ``stem``'s features in a FeatureCollection file, keep the rest."""
    gj = _load_json(path, {"type": "FeatureCollection", "features": []})
    kept = [
        f for f in gj.get("features", [])
        if f.get("properties", {}).get("image") != stem
    ]
    for i, pts in enumerate(geoms):
        coords: object
        if geom_type == "Polygon":
            if len(pts) < 4:
                continue
            coords = [[[float(x), float(y)] for x, y in pts]]
        else:  # LineString
            if len(pts) < 2:
                continue
            coords = [[float(x), float(y)] for x, y in pts]
        kept.append({
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": coords},
            "properties": {"image": stem, "id": i},
        })
    out = {
        "type": "FeatureCollection",
        "features": kept,
        "crs": {"type": "name", "properties": {"name": crs_name}},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return path


def _rebuild_master_dxf(master_dir: Path, geo: bool) -> Path | None:
    """Regenerate master.dxf from the two master GeoJSONs."""
    if not HAS_EZDXF:
        logger.warning("ezdxf not installed — master DXF skipped")
        return None

    center = _load_json(master_dir / MASTER_CENTERLINES,
                        {"type": "FeatureCollection", "features": []})
    outline = _load_json(master_dir / MASTER_OUTLINES,
                         {"type": "FeatureCollection", "features": []})

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 6 if geo else 0  # 6 = meters
    msp = doc.modelspace()

    stems = sorted(
        {f["properties"]["image"] for f in center["features"]}
        | {f["properties"]["image"] for f in outline["features"]}
    )
    for i, stem in enumerate(stems):
        color = _LAYER_COLORS[i % len(_LAYER_COLORS)]
        for suffix in ("OUTLINE", "CENTERLINE"):
            name = _layer_name(stem, suffix)
            if name not in doc.layers:
                doc.layers.add(name, color=color)

    for f in outline["features"]:
        stem = f["properties"]["image"]
        ring = f["geometry"]["coordinates"][0]
        if len(ring) >= 3:
            msp.add_lwpolyline([(x, y) for x, y in ring], close=True,
                               dxfattribs={"layer": _layer_name(stem, "OUTLINE")})
    for f in center["features"]:
        stem = f["properties"]["image"]
        line = f["geometry"]["coordinates"]
        if len(line) >= 2:
            msp.add_lwpolyline([(x, y) for x, y in line], close=False,
                               dxfattribs={"layer": _layer_name(stem, "CENTERLINE")})

    out = master_dir / MASTER_DXF
    doc.saveas(str(out))
    return out


def _rebuild_master_gpkg(master_dir: Path, crs_name: str, geo: bool) -> Path | None:
    """Regenerate master.gpkg from the two master GeoJSONs.

    GeoPackage is the one open GIS format with real layers in a single
    file, so here every grave gets its own layer pair
    ``<stem>_outlines`` / ``<stem>_centerlines`` — the GIS twin of the
    per-grave DXF layers. Rebuilt from scratch on every append (fast at
    this size, and stale layers from removed graves can't linger).
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Polygon
    except Exception:
        logger.warning("geopandas not installed — master GeoPackage skipped")
        return None

    center = _load_json(master_dir / MASTER_CENTERLINES,
                        {"type": "FeatureCollection", "features": []})
    outline = _load_json(master_dir / MASTER_OUTLINES,
                         {"type": "FeatureCollection", "features": []})
    out = master_dir / MASTER_GPKG
    if out.exists():
        out.unlink()  # full rebuild: drop stale layers

    crs = crs_name if geo else None
    stems = sorted(
        {f["properties"]["image"] for f in center["features"]}
        | {f["properties"]["image"] for f in outline["features"]}
    )
    n_layers = 0
    for stem in stems:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "image"
        rings = [
            Polygon(f["geometry"]["coordinates"][0])
            for f in outline["features"]
            if f["properties"]["image"] == stem
            and len(f["geometry"]["coordinates"][0]) >= 4
        ]
        lines = [
            LineString(f["geometry"]["coordinates"])
            for f in center["features"]
            if f["properties"]["image"] == stem
            and len(f["geometry"]["coordinates"]) >= 2
        ]
        if rings:
            gpd.GeoDataFrame(
                {"image": [stem] * len(rings), "id": range(len(rings))},
                geometry=rings, crs=crs,
            ).to_file(out, layer=f"{safe}_outlines", driver="GPKG")
            n_layers += 1
        if lines:
            gpd.GeoDataFrame(
                {"image": [stem] * len(lines), "id": range(len(lines))},
                geometry=lines, crs=crs,
            ).to_file(out, layer=f"{safe}_centerlines", driver="GPKG")
            n_layers += 1

    if n_layers == 0:
        return None
    logger.info("master.gpkg rebuilt: %d layers (%d graves)", n_layers, len(stems))
    return out


def append_to_master(
    master_dir: Path,
    stem: str,
    rings_out: list[Polyline],
    polylines_out: list[Polyline],
    georef: GeoRef | None,
    image_width: int,
    image_height: int,
) -> list[Path]:
    """Merge one image's vectors into the master files; returns written paths.

    Raises ``ValueError`` when mixing georeferenced and plain images in the
    same master directory (their coordinates cannot coexist meaningfully).
    """
    master_dir = Path(master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)
    layout_path = master_dir / MASTER_LAYOUT
    layout = _load_json(layout_path, {"mode": None, "crs": None,
                                      "cursor_x": 0.0, "images": {}})

    mode = "geo" if georef is not None else "pixel"
    if layout["images"] and layout.get("mode") != mode:
        raise ValueError(
            f"Master file at {master_dir} was built from "
            f"{'georeferenced' if layout.get('mode') == 'geo' else 'plain'} images; "
            f"cannot add a {'georeferenced' if mode == 'geo' else 'plain'} one. "
            "Use a different output directory."
        )
    layout["mode"] = mode

    crs_name = str(georef.crs) if (georef is not None and georef.crs) else "pixel"
    if mode == "geo":
        if layout.get("crs") and layout["crs"] != crs_name:
            logger.warning(
                "Master CRS mismatch: file is %s, %s is %s — appended anyway, "
                "check positions in GIS", layout["crs"], stem, crs_name,
            )
        else:
            layout["crs"] = crs_name
        dx = dy = 0.0
        layout["images"][stem] = {"x_off": 0.0, "y_off": 0.0}
    else:
        entry = layout["images"].get(stem)
        if entry is None:
            entry = {"x_off": float(layout["cursor_x"]), "y_off": 0.0,
                     "width": image_width, "height": image_height}
            layout["images"][stem] = entry
            layout["cursor_x"] = float(layout["cursor_x"]) + image_width + PIXEL_STRIP_PAD
        dx, dy = float(entry["x_off"]), float(entry["y_off"])
        crs_name = "pixel (strip layout — positions not georeferenced)"

    rings = [[(x + dx, y + dy) for x, y in r] for r in rings_out]
    lines = [[(x + dx, y + dy) for x, y in l] for l in polylines_out]

    written = [
        _merge_geojson(master_dir / MASTER_CENTERLINES, stem, lines, "LineString", crs_name),
        _merge_geojson(master_dir / MASTER_OUTLINES, stem, rings, "Polygon", crs_name),
    ]
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    dxf = _rebuild_master_dxf(master_dir, geo=(mode == "geo"))
    if dxf is not None:
        written.append(dxf)
    gpkg = _rebuild_master_gpkg(
        master_dir,
        crs_name=str(georef.crs) if (georef is not None and georef.crs) else "",
        geo=(mode == "geo"),
    )
    if gpkg is not None:
        written.append(gpkg)

    logger.info(
        "Master updated with %s (%d outlines, %d centerlines, mode=%s) -> %s",
        stem, len(rings), len(lines), mode, master_dir,
    )
    return written
