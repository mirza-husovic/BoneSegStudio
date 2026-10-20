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


def rasterize_polylines(
    polylines_px: list[Polyline], shape: tuple[int, int], thickness: int = 1
) -> np.ndarray:
    """Draw pixel polylines onto a {0, 1} raster of ``shape`` (h, w).

    ``thickness=1`` keeps a hand-edited result's ``skeleton`` layer consistent
    with the polylines it was drawn from (skeleton PNG export, the
    mask-unchanged apply path); save_training dilates it to reproduce the
    UNET_DATASET_FINAL4 stroke profile.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    polys = [np.rint(np.asarray(line, dtype=np.float64)).astype(np.int32)
             for line in polylines_px if len(line) >= 2]
    if not polys:
        return img
    if HAS_CV2:
        # ONE call for every polyline: 400 separate cv2.polylines calls over a
        # 65 MP canvas cost ~0.7 s, which showed up on every "Apply edits".
        cv2.polylines(img, polys, isClosed=False, color=1, thickness=thickness)
    else:  # pragma: no cover - opencv is a hard dependency in practice
        from skimage.draw import line as sk_line
        for pts in polys:
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                rr, cc = sk_line(int(y0), int(x0), int(y1), int(x1))
                keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                img[rr[keep], cc[keep]] = 1
    return img


def polylines_px_to_output(
    polylines_px: list[Polyline],
    georef: GeoRef | None,
) -> list[Polyline]:
    """Map pixel polylines into GeoJSON output coordinates.

    Georeferenced: through the affine transform into the input CRS — or
    through the exact perspective homography when the GCP fit detected an
    oblique photo (GeoRef.homography), point by point.
    Plain image: Y negated, matching QGIS's placement of an ungeoreferenced
    raster at world Y = 0..-H (same convention as predict.py).
    """
    if georef is not None and georef.homography is not None:
        from boneseg.data.georef_fit import apply_homography
        out = []
        for line in polylines_px:
            if not line:
                out.append([])
                continue
            pts = apply_homography(georef.homography, line)
            out.append([(float(x), float(y)) for x, y in pts])
        return out
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
def smooth_closed_ring(pts, step: float = 1.0, sigma_steps: float = 1.5,
                       eps_steps: float = 0.2) -> np.ndarray | None:
    """Smooth a closed ring traced along a pixel grid.

    ``step`` is one pixel in the ring's units. The ring is resampled every
    half pixel, Gaussian-smoothed along its length (sigma 1.5 px, wrapping
    round the closure) and Douglas-Peucker simplified to 0.2 px: the pixel
    staircase disappears, while the line stays within a fraction of a pixel
    of the true bone edge (a 2 mm/px photo moves by well under 1 mm).
    Returns an (N, 2) closed array, or None for degenerate rings."""
    from scipy.ndimage import gaussian_filter1d

    a = np.asarray(pts, dtype=np.float64)
    if len(a) < 4:
        return None
    if not np.allclose(a[0], a[-1]):
        a = np.vstack([a, a[:1]])
    seg = np.linalg.norm(np.diff(a, axis=0), axis=1)
    total = float(seg.sum())
    if total < 4 * step:
        return a                          # a speck: nothing to smooth
    origin = a[0].copy()                  # float32 approxPolyDP needs small numbers
    a = a - origin
    t = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(total / (0.5 * step)))
    u = np.linspace(0.0, total, n, endpoint=False)
    r = np.column_stack([np.interp(u, t, a[:, 0]), np.interp(u, t, a[:, 1])])
    r = gaussian_filter1d(r, sigma=sigma_steps * 2.0, axis=0, mode="wrap")
    if HAS_CV2:
        r = cv2.approxPolyDP(r.astype(np.float32).reshape(-1, 1, 2),
                             eps_steps * step, True).reshape(-1, 2).astype(np.float64)
    if len(r) < 3:
        return None
    r = np.vstack([r, r[:1]]) + origin
    return r


def smooth_staircase_ring(ring: Polyline) -> Polyline:
    """Smooth a ring from an OLD export (raw pixel edges, before 2026-10).

    Detected by its geometry: nearly every segment runs along one of two
    perpendicular directions (the pixel grid, possibly rotated by the
    georeference). The pixel size is the shortest common segment. Rings
    that are already smooth come back unchanged."""
    a = np.asarray(ring, dtype=np.float64)
    if len(a) < 6:
        return ring
    d = np.diff(a, axis=0)
    seg = np.linalg.norm(d, axis=1)
    keep = seg > 0
    if keep.sum() < 5:
        return ring
    # Two dominant directions (mod 180): the pixel grid's rows and columns
    # in world space — not necessarily perpendicular, a GCP affine shears.
    ang = np.degrees(np.arctan2(d[keep, 1], d[keep, 0])) % 180.0
    hist, _ = np.histogram(ang, bins=180, range=(0.0, 180.0))
    first = int(np.argmax(hist))
    circ = lambda x, c: np.minimum(np.abs(x - c), 180.0 - np.abs(x - c))  # noqa: E731
    far = circ(np.arange(180) + 0.5, first + 0.5) >= 20.0
    second = int(np.argmax(np.where(far, hist, -1)))
    near = (circ(ang, first + 0.5) <= 2.0) | (circ(ang, second + 0.5) <= 2.0)
    if np.mean(near) < 0.9:
        return ring
    step = float(np.percentile(seg[keep], 10))
    sm = smooth_closed_ring(a, step=step)
    return ring if sm is None else [(float(x), float(y)) for x, y in sm]


def mask_to_rings(mask01: np.ndarray, georef: GeoRef | None) -> list[Polyline]:
    """Vectorize mask boundaries to smooth closed exterior rings in output coords.

    The boundary is traced with marching squares (the line runs exactly
    between bone and non-bone pixels, not along pixel centres), smoothed
    with :func:`smooth_closed_ring`, and mapped to output coordinates by
    the SAME function as the centerlines, so outlines and centerlines
    share one convention. Before 2026-10 the rings were the raw pixel
    edges (rasterio ``shapes``): a 1-pixel staircase that CAD showed as
    zig-zag lines.
    """
    from skimage.measure import find_contours

    m = (np.asarray(mask01) > 0).astype(np.uint8)
    if not m.any():
        return []
    rings_px: list[Polyline] = []
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        crop = (_lab[y:y + h, x:x + w] == i).astype(np.float32)
        for c in find_contours(np.pad(crop, 1), 0.5):
            if len(c) < 4:
                continue
            cx = c[:, 1] - 1 + x
            cy = c[:, 0] - 1 + y
            # positive signed area (x right, y down) = exterior; negative = hole
            if 0.5 * np.sum(cx * np.roll(cy, -1) - np.roll(cx, -1) * cy) <= 0:
                continue
            sm = smooth_closed_ring(np.column_stack([cx, cy]))
            if sm is not None:
                rings_px.append([(float(px), float(py)) for px, py in sm])
    return polylines_px_to_output(rings_px, georef)


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
