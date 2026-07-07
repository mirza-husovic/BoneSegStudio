"""Vectorization: skeleton graph -> smoothed polylines, mask -> outline rings.

Spline smoothing is the same algorithm as predict.py (splprep with a
length-scaled ``s``, pre-thinned by keeping every Kth medial-axis point).
One deliberate refinement: the spline is always fitted in PIXEL space and
the smoothed points are mapped to geo coordinates afterwards. For plain
images this is bit-identical to predict.py; for georeferenced inputs it
makes the smoothing strength independent of the CRS units (predict.py fitted
in geo space, where a residual budget tuned for pixels over-smooths
meter-scale coordinates).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.interpolate import splev, splprep

from boneseg.data.readers import HAS_RASTERIO, GeoRef
from boneseg.logging_setup import get_logger
from boneseg.postprocessing.skeleton import iter_edges

if HAS_RASTERIO:
    import rasterio
    from rasterio.features import shapes as rio_shapes

try:
    import cv2

    HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    HAS_CV2 = False

logger = get_logger(__name__)

Polyline = list[tuple[float, float]]


# ---------------------------------------------------------------------------
# Centerlines
# ---------------------------------------------------------------------------
def graph_to_polylines_px(
    graph: nx.Graph | None,
    downsample_k: int,
    spline_smooth: float,
    density: int,
) -> list[Polyline]:
    """Convert skeleton-graph edges to smoothed polylines in pixel space.

    Returned coordinates are (x=col, y=row) with Y down — plain image
    coordinates, ready for SVG. Use :func:`polylines_px_to_output` to map
    them into GeoJSON space.
    """
    if graph is None:
        return []

    polylines: list[Polyline] = []
    for _, _, _, edge_data in iter_edges(graph):
        pts = edge_data["pts"]

        # Pre-thin the skeleton: medial-axis points are 1 px apart and the
        # zigzag they describe is exactly what makes splines wriggle. Keep
        # every Kth point (always keep the first and last so endpoints are
        # exact).
        if downsample_k > 1 and len(pts) > downsample_k * 3:
            keep = list(range(0, len(pts), downsample_k))
            if keep[-1] != len(pts) - 1:
                keep.append(len(pts) - 1)
            pts = pts[keep]

        # sknw stores (row, col); polylines are (x, y) = (col, row)
        coords = [(float(p[1]), float(p[0])) for p in pts]
        if len(coords) < 2:
            continue
        if len(coords) <= 8:
            # Too few points for a stable spline fit. predict.py dropped
            # these edges outright, but here pruning has already filtered
            # noise upstream, and hand-painted short strokes must survive:
            # keep the raw pixel chain so the exported vectors contain
            # every edge the skeleton view shows.
            polylines.append(coords)
            continue

        try:
            arr = np.array(coords, dtype=float)
            # Scale smoothing with path length: splprep's `s` is a sum of
            # squared residuals so it must grow with the number of points
            # or short segments get smoothed away.
            s_eff = max(spline_smooth, len(arr))
            tck, _ = splprep([arr[:, 0], arr[:, 1]], s=s_eff)
            unew = np.linspace(0.0, 1.0, len(arr) * density)
            xs, ys = splev(unew, tck)
            polylines.append(list(zip(xs.tolist(), ys.tolist())))
        except Exception as exc:
            logger.debug("spline fit failed for one edge (%s); kept raw", exc)
            polylines.append(coords)
            continue

    return polylines


def polylines_px_to_output(
    polylines_px: list[Polyline],
    georef: GeoRef | None,
) -> list[Polyline]:
    """Map pixel polylines into GeoJSON output coordinates.

    Georeferenced: through the affine transform into the input CRS.
    Plain image: Y negated, matching QGIS's placement of an ungeoreferenced
    raster at world Y = 0..-H (same convention as predict.py).
    """
    if georef is not None and HAS_RASTERIO:
        out: list[Polyline] = []
        for line in polylines_px:
            cols = [x for x, _ in line]
            rows = [y for _, y in line]
            xs, ys = rasterio.transform.xy(georef.transform, rows, cols)
            out.append(list(zip(map(float, xs), map(float, ys))))
        return out
    return [[(x, -y) for x, y in line] for line in polylines_px]


# ---------------------------------------------------------------------------
# Outline polygons (carried over from predict.py's polygons/ output)
# ---------------------------------------------------------------------------
def mask_to_rings(mask01: np.ndarray, georef: GeoRef | None) -> list[Polyline]:
    """Vectorize mask boundaries to closed exterior rings in output coords.

    Uses rasterio's polygonizer when the input is georeferenced, otherwise
    OpenCV contours in QGIS-pixel coordinates — exactly as predict.py did.
    """
    rings: list[Polyline] = []
    if HAS_RASTERIO and georef is not None:
        for geom, val in rio_shapes(mask01, mask=(mask01 > 0), transform=georef.transform):
            if val == 1:
                rings.append([(float(x), float(y)) for x, y in geom["coordinates"][0]])
    elif HAS_CV2:
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            raw = c.reshape(-1, 2).tolist()
            if len(raw) < 3:
                continue
            # Y-flip so the vectors align with the raster QGIS draws at Y=0..-H
            pts: Polyline = [(float(x), float(-y)) for x, y in raw]
            pts.append(pts[0])  # close ring
            rings.append(pts)
    else:
        logger.warning("no rasterio or opencv available for polygonization")
    return rings


def rings_to_geojson_dict(rings: list[Polyline], georef: GeoRef | None) -> dict:
    """Serialize closed rings as a Polygon FeatureCollection dict."""
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[float(x), float(y)] for x, y in ring]],
            },
            "properties": {"id": i},
        }
        for i, ring in enumerate(rings)
        if len(ring) >= 4  # closed ring = at least 3 vertices + repeat
    ]
    crs_name = str(georef.crs) if (georef is not None and georef.crs) else "pixel"
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs_name}},
    }
