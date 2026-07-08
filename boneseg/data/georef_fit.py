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

    scored: dict[tuple[int, ...], tuple[float, bool]] = {}
    for k in order:
        assign = _greedy(int(k))
        if assign is None:
            continue
        key = tuple(assign)
        if key in scored:
            continue
        gcps = [{"px": c[i, 0], "py": c[i, 1],
                 "e": w[j, 0], "n": w[j, 1]} for i, j in enumerate(assign)]
        fit, _res, rms = fit_affine_gcps(gcps)
        det = fit.a * fit.e - fit.b * fit.d
        scored[key] = (rms, det > 0)

    if not scored:
        raise ValueError("Auto-matching failed — no consistent assignment found.")
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0])
    (best_key, (best_rms, mirrored)) = ranked[0]
    second = next((rms for key, (rms, mir) in ranked[1:] if mir == mirrored),
                  float("inf"))
    logger.info("GCP auto-match: %d clicks vs %d points, rms %.3f m "
                "(runner-up %.3f m)%s", n, m, best_rms,
                second if math.isfinite(second) else -1.0,
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
