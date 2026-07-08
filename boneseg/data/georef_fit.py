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


# An affine that already fits this well means the photo is effectively
# nadir — perspective correction would add nothing but would cost the
# world-file workflow, so we stay affine below this RMS.
NADIR_RMS_M = 0.05


def fit_homography_gcps(
    gcps: list[dict],
) -> tuple[np.ndarray, list[float], float]:
    """Least-squares homography (3x3, pixel -> world) from >=4 GCPs.

    A homography models the perspective of an OBLIQUE photo of a flat
    plane exactly — what the affine cannot. Solved as a normalized DLT
    (Hartley normalization matters here: world coordinates are ~5e6 m, a
    raw DLT would be numerically useless). Exact for 4 points;
    overdetermined beyond. Returns (H, per-point residuals in m, RMS).
    """
    if len(gcps) < 4:
        raise ValueError("At least 4 control points are required for perspective")

    src = np.array([[float(g["px"]), float(g["py"])] for g in gcps])
    dst = np.array([[float(g["e"]), float(g["n"])] for g in gcps])

    def _norm(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean() or 1.0
        s = math.sqrt(2.0) / d
        t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        homog = np.column_stack([pts, np.ones(len(pts))])
        return (t @ homog.T).T[:, :2], t

    sn, t_src = _norm(src)
    dn, t_dst = _norm(dst)

    a = []
    for (x, y), (u, v) in zip(sn, dn):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _u, _s, vt = np.linalg.svd(np.asarray(a))
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    h = h / h[2, 2]

    proj = apply_homography(h, src)
    residuals = [float(math.hypot(px - ex, py - ey))
                 for (px, py), (ex, ey) in zip(proj, dst)]
    rms = float(math.sqrt(sum(r * r for r in residuals) / len(residuals)))
    return h, residuals, rms


def apply_homography(h: np.ndarray, pts) -> np.ndarray:
    """Project Nx2 points through a 3x3 homography."""
    p = np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))])
    q = (h @ p.T).T
    return q[:, :2] / q[:, 2:3]


def georef_from_gcps(gcps: list[dict], epsg: int | None) -> tuple[GeoRef, list[float], float]:
    """Fit + wrap in the pipeline's GeoRef (CRS optional = local grid).

    3 points -> affine. 4+ points -> affine first; if it fits to nadir
    accuracy (RMS <= NADIR_RMS_M) it stays — otherwise the photo is
    oblique and a homography is fitted on top (GeoRef.homography set),
    with the affine kept as the approximate transform for consumers that
    cannot handle perspective. Returned residuals belong to whichever
    model is ACTIVE.
    """
    transform, residuals, rms = fit_affine_gcps(gcps)
    crs = rasterio.crs.CRS.from_epsg(int(epsg)) if epsg else None
    if len(gcps) >= 4 and rms > NADIR_RMS_M:
        h, residuals_h, rms_h = fit_homography_gcps(gcps)
        logger.info("Oblique photo detected (affine rms %.3f m) — "
                    "homography rms %.3f m", rms, rms_h)
        return GeoRef(transform, crs, homography=h), residuals_h, rms_h
    return GeoRef(transform, crs), residuals, rms


def rectify_params(
    georef: GeoRef, width: int, height: int, max_pixels: int | None = None,
) -> tuple["Affine", np.ndarray, int, int]:
    """North-up target grid for warping (rectifying) the photo/mask.

    Returns ``(grid_transform, K, out_w, out_h)``: an axis-aligned
    world grid covering the image footprint at the image-centre ground
    sample distance, and the 3x3 ``K`` mapping ORIGINAL pixel -> GRID
    pixel for cv2.warpPerspective. Works for both perspective and
    plain-affine georefs (the DXF underlay rectifies affine shear too —
    AutoCAD cannot display sheared rasters at all).
    """
    if georef.homography is not None:
        h = np.asarray(georef.homography, dtype=np.float64)
    else:
        t = georef.transform
        h = np.array([[t.a, t.b, t.c], [t.d, t.e, t.f], [0, 0, 1.0]])

    w, hh = float(width), float(height)
    corners = apply_homography(h, [(0, 0), (w, 0), (w, hh), (0, hh)])
    minx, miny = corners.min(axis=0)
    maxx, maxy = corners.max(axis=0)

    # Ground sample distance at the image centre (metres per pixel).
    cx, cy = w / 2.0, hh / 2.0
    steps = apply_homography(h, [(cx, cy), (cx + 1, cy), (cx, cy + 1)])
    gsd = float((np.linalg.norm(steps[1] - steps[0])
                 + np.linalg.norm(steps[2] - steps[0])) / 2.0) or 1.0

    # Cap the output size — extreme obliques blow the footprint up.
    max_pixels = max_pixels or int(4 * width * height)
    out_w = max(1, math.ceil((maxx - minx) / gsd))
    out_h = max(1, math.ceil((maxy - miny) / gsd))
    if out_w * out_h > max_pixels:
        f = math.sqrt(out_w * out_h / max_pixels)
        gsd *= f
        out_w = max(1, math.ceil((maxx - minx) / gsd))
        out_h = max(1, math.ceil((maxy - miny) / gsd))
        logger.warning("Rectified grid capped: gsd coarsened %.2fx", f)

    grid = Affine.translation(minx, maxy) * Affine.scale(gsd, -gsd)
    g3 = np.array([[grid.a, grid.b, grid.c], [grid.d, grid.e, grid.f],
                   [0, 0, 1.0]])
    k = np.linalg.inv(g3) @ h
    return grid, k, out_w, out_h


MAX_AUTO_POINTS = 40  # seed search is O(m^3); 40 -> ~60k seeds, still <1 s


def auto_assign_gcps(
    clicks: list[tuple[float, float]],
    points: list[tuple[float, float]],
) -> tuple[list[int], float, bool, float]:
    """Match clicked pixel positions to surveyed points in ANY order.

    The user clicks n>=3 spots on the photo and supplies m>=n surveyed
    points (a per-grave file often holds extra points that are not in the
    frame — those are simply left unused). The correct pairing is found
    geometrically: three well-spread clicks are seeded against every
    ordered triple of surveyed points, the exact affine of each seed
    predicts where the remaining clicks land in world space, remaining
    clicks are greedily matched to the nearest unused points, and the
    assignment with the smallest full-refit RMS wins.

    Orientation disambiguates symmetric layouts: pixel rows grow DOWN
    while northing grows UP, so a genuine photo->world affine has a
    negative determinant. A best fit with POSITIVE determinant means the
    world points are mirrored — i.e. E and N are swapped in the file
    (the classic geodetic Y/X trap); the caller swaps and retries.

    Statistically tied candidates (symmetric layouts — the classic
    4-corner rectangle) are tie-broken by least distortion, then by click
    order; see the comment at the ranking below.

    Returns ``(assignment, rms_m, mirrored, second_rms_m)`` where
    ``assignment[i]`` is the index into ``points`` matched to
    ``clicks[i]`` and ``second_rms_m`` is the RMS of the best DIFFERENT
    assignment with the same orientation (close to ``rms_m`` = the layout
    is ambiguous, e.g. a near-perfect rectangle — worth a UI warning).
    """
    n, m = len(clicks), len(points)
    if n < 3:
        raise ValueError("At least 3 clicked points are required")
    if m < n:
        raise ValueError(f"The point list has only {m} point(s) for {n} clicks "
                         "— every click needs a surveyed point to match.")
    if m > MAX_AUTO_POINTS:
        raise ValueError(f"Too many points for auto-matching ({m} > {MAX_AUTO_POINTS}) "
                         "— trim the file to the points on this photo.")

    c = np.asarray(clicks, dtype=np.float64)          # (n, 2)
    w = np.asarray(points, dtype=np.float64)          # (m, 2)

    # Seed on the most spread-out click triple (a fat triangle keeps the
    # exact 3-point affine well conditioned).
    from itertools import combinations, permutations
    best_area, seed = 0.0, (0, 1, 2)
    for tri in combinations(range(n), 3):
        a, b, d = c[tri[0]], c[tri[1]], c[tri[2]]
        area = abs((b[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (b[1] - a[1]))
        if area > best_area:
            best_area, seed = area, tri
    span = float(max(c[:, 0].ptp(), c[:, 1].ptp())) or 1.0
    if best_area < 1e-4 * span * span:
        raise ValueError("The clicked points are (nearly) collinear — "
                         "spread them across the photo.")

    a_mat = np.column_stack([c[list(seed)], np.ones(3)])  # (3, 3), fixed
    a_inv = np.linalg.inv(a_mat)
    rest = [i for i in range(n) if i not in seed]

    # All ordered point triples at once: coeffs (K,3,2), then every click
    # projected through every candidate affine in one einsum.
    triples = np.array(list(permutations(range(m), 3)), dtype=np.intp)  # (K,3)
    coeffs = np.einsum("ab,kbe->kae", a_inv, w[triples])                # (K,3,2)
    ones = np.column_stack([c, np.ones(n)])                            # (n,3)
    pred = np.einsum("na,kae->kne", ones, coeffs)                      # (K,n,2)
    # Distance of every predicted click to every surveyed point, then a
    # cheap lower bound (nearest point, reuse allowed) ranks candidates so
    # only the plausible few get the exact greedy + refit treatment.
    d2 = ((pred[:, :, None, :] - w[None, None, :, :]) ** 2).sum(-1)    # (K,n,m)
    lower = d2.min(-1).sum(-1)                                         # (K,)
    order = np.argsort(lower)[:200]

    def _greedy(k: int) -> list[int] | None:
        assign = [-1] * n
        used = set()
        for s_pos, p_idx in zip(seed, triples[k]):
            assign[s_pos] = int(p_idx)
            used.add(int(p_idx))
        for i in rest:
            cand = np.argsort(d2[k, i])
            pick = next((int(j) for j in cand if int(j) not in used), None)
            if pick is None:
                return None
            assign[i] = pick
            used.add(pick)
        return assign

    def _fit(key: tuple[int, ...]):
        gcps = [{"px": c[i, 0], "py": c[i, 1],
                 "e": w[j, 0], "n": w[j, 1]} for i, j in enumerate(key)]
        return fit_affine_gcps(gcps)

    def _aniso(fit) -> float:
        """Distortion of the fitted map: ratio of the affine's singular
        values (1 = isotropic nadir look, big = photo squashed)."""
        sv = np.linalg.svd(np.array([[fit.a, fit.b], [fit.d, fit.e]]),
                           compute_uv=False)
        return float(sv[0] / sv[1]) if sv[1] > 0 else float("inf")

    scored: dict[tuple[int, ...], tuple[float, bool, float]] = {}
    for k in order:
        assign = _greedy(int(k))
        if assign is None:
            continue
        key = tuple(assign)
        if key in scored:
            continue
        fit, _res, rms = _fit(key)
        det = fit.a * fit.e - fit.b * fit.d
        scored[key] = (rms, det > 0, _aniso(fit))

    if not scored:
        raise ValueError("Auto-matching failed — no consistent assignment found.")
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0])
    best_rms0 = ranked[0][1][0]
    best_mir = ranked[0][1][1]

    # SYMMETRIC LAYOUTS (the common 4-corner rectangle!): several pairings
    # fit within measurement noise — an affine maps a rectangle onto any
    # rotated/cycled version of itself EXACTLY, so RMS cannot decide.
    # Tie-break among the statistically tied candidates:
    #   1. least distortion — a wrong (cycled) pairing only fits by
    #      squashing the photo by aspect² (the "stretched" QGIS look),
    #      the true pairing is the most isotropic one;
    #   2. RMS in 5 mm buckets — a genuinely better fit still wins;
    #   3. click order — users overwhelmingly click points in file order,
    #      so among truly equal candidates (a 180° rotation of a perfect
    #      rectangle is geometrically indistinguishable) prefer the
    #      identity-like assignment.
    near = [(key, rms, an) for key, (rms, mir, an) in ranked
            if mir == best_mir and rms <= max(2 * best_rms0, best_rms0 + 0.02)]
    near.sort(key=lambda t: (round(t[2], 1),
                             round(t[1] / 0.005),
                             -sum(1 for i, j in enumerate(t[0]) if i == j),
                             t[1]))
    best_key, best_rms, aniso = near[0]
    mirrored = best_mir
    second = near[1][1] if len(near) > 1 else next(
        (rms for key, (rms, mir, _a) in ranked
         if mir == mirrored and key != best_key), float("inf"))
    logger.info("GCP auto-match: %d clicks vs %d points, rms %.3f m, "
                "foreshortening x%.2f (runner-up %.3f m, %d tied)%s",
                n, m, best_rms, aniso,
                second if math.isfinite(second) else -1.0, len(near),
                " — MIRRORED (E/N swapped?)" if mirrored else "")
    return list(best_key), best_rms, mirrored, second


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
