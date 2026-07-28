"""Best-effort EPSG auto-detection from raw GCP coordinate magnitudes.

Field surveys hand this tool raw projected E/N pairs with no CRS metadata
attached (see ``gcp_parse.py``'s docstring) — the numbers alone are the only
signal. Each candidate CRS below carries a plausible raw easting/northing
range (a Gauss-Krüger zone-prefixed easting like 6,402,736 looks nothing
like HTRS96/TM's 519,351 for the same physical point) as a fast pre-filter;
a range match alone isn't proof though — e.g. UTM33N and HTRS96/TM eastings
overlap — so each survivor is also transformed to WGS84 and checked against
a loose Balkan-region bounding box before being accepted.

The E/N columns are also tried swapped, since total-station exports in the
wild disagree on column order (some lead with northing) and the existing
parser has no way to tell without this kind of external check.

This is a *pre-fill*, never a hard override — the caller always leaves the
EPSG field and swap checkbox editable.
"""

from __future__ import annotations

from boneseg.data.readers import HAS_RASTERIO
from boneseg.logging_setup import get_logger

if HAS_RASTERIO:
    import rasterio
    from rasterio.warp import transform as warp_transform

logger = get_logger(__name__)

# Loose WGS84 sanity box a candidate's transformed centroid must land in to
# be considered plausible — wide enough for Croatia/Bosnia/Serbia/Montenegro/
# Slovenia and the Hungarian border strip, tight enough to reject a
# numerically-plausible but geographically-wrong candidate.
_REGION_LON = (12.0, 23.5)
_REGION_LAT = (41.0, 47.0)

# (epsg, name, easting_range, northing_range) — ranges are raw-value
# pre-filters, not exact zone bounds. Listed in rough preference order:
# when two candidates both pass (e.g. 3908 vs. 31276 — same zone-6 numbers,
# different datum realisation, a few hundred metres apart), the earlier one
# wins the "best" slot but both are reported as candidates so the ambiguity
# is visible.
_CANDIDATES: list[tuple[int, str, tuple[float, float], tuple[float, float]]] = [
    (3765, "HTRS96/TM (Hrvatska)", (150_000, 850_000), (4_600_000, 5_250_000)),
    (3908, "MGI 1901 / Balkans zone 6", (6_150_000, 6_850_000), (4_600_000, 5_250_000)),
    (31276, "MGI / Balkan zone 6", (6_150_000, 6_850_000), (4_600_000, 5_250_000)),
    (31275, "MGI / Balkan zone 5", (5_150_000, 5_850_000), (4_600_000, 5_250_000)),
    (23700, "EOV (Mađarska)", (400_000, 950_000), (25_000, 350_000)),
    (32634, "UTM zone 34N (WGS84)", (150_000, 850_000), (4_600_000, 5_250_000)),
    (32633, "UTM zone 33N (WGS84)", (150_000, 850_000), (4_600_000, 5_250_000)),
]


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def guess_crs(points: list[dict]) -> dict:
    """Guess the EPSG code (+ E/N swap) of raw survey points from magnitude.

    Returns ``{"epsg", "name", "swap", "confidence", "candidates"}`` where
    confidence is "high" (exactly one plausible match), "ambiguous" (more
    than one — e.g. 3908 vs. 31276), or "none". ``candidates`` lists every
    plausible match (each with its own "swap" flag) for the UI to surface.
    """
    empty = {"epsg": None, "name": None, "swap": False,
              "confidence": "none", "candidates": []}
    if not points or not HAS_RASTERIO:
        return empty

    e0 = _median([p["e"] for p in points])
    n0 = _median([p["n"] for p in points])

    passed = []
    for swap, (a, b) in ((False, (e0, n0)), (True, (n0, e0))):
        for epsg, name, (emin, emax), (nmin, nmax) in _CANDIDATES:
            if not (emin <= a <= emax and nmin <= b <= nmax):
                continue
            try:
                lons, lats = warp_transform(
                    rasterio.crs.CRS.from_epsg(epsg),
                    rasterio.crs.CRS.from_epsg(4326), [a], [b])
                lon, lat = lons[0], lats[0]
            except Exception:
                logger.debug("crs_guess: EPSG:%d transform failed", epsg, exc_info=True)
                continue
            if _REGION_LON[0] <= lon <= _REGION_LON[1] and _REGION_LAT[0] <= lat <= _REGION_LAT[1]:
                passed.append({"epsg": epsg, "name": name, "swap": swap,
                                "lon": round(lon, 4), "lat": round(lat, 4)})

    if not passed:
        return empty
    best = passed[0]
    return {"epsg": best["epsg"], "name": best["name"], "swap": best["swap"],
             "confidence": "high" if len(passed) == 1 else "ambiguous",
             "candidates": passed}


def region_check(epsg: int | None, e: float, n: float) -> str | None:
    """Sanity-check a FITTED georeference (called at Apply time, not just at
    load time): reproject the world coordinate to WGS84 and warn when it
    falls outside the Balkan region — this catches a wrong EPSG or a
    mis-applied E/N swap (e.g. the load-time guess already swapped and the
    user toggled the checkbox again, cancelling it out) regardless of how
    the wrong orientation happened, before the user finds out in QGIS.
    """
    if epsg is None or not HAS_RASTERIO:
        return None
    try:
        lons, lats = warp_transform(
            rasterio.crs.CRS.from_epsg(epsg), rasterio.crs.CRS.from_epsg(4326), [e], [n])
        lon, lat = lons[0], lats[0]
    except Exception:
        logger.debug("region_check: EPSG:%s transform failed", epsg, exc_info=True)
        return None
    if _REGION_LON[0] <= lon <= _REGION_LON[1] and _REGION_LAT[0] <= lat <= _REGION_LAT[1]:
        return None
    return (f"Computed location is lon={lon:.2f}, lat={lat:.2f} — far outside "
            "the expected region for this site. Check the EPSG code and the "
            "swap E/N checkbox (a load-time auto-swap may have been "
            "toggled back off).")
