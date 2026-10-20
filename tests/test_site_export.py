"""Weight-free tests for the site view and the vector output (CI-safe).

Covers what the CAD/GIS drawing depends on:
  * outlines are smooth curves on the bone edge, not pixel staircases;
  * old staircase exports are smoothed when the site view loads them;
  * GNSS-default "TM 0" points are recognised and converted to HTRS96;
  * point IDs with "_" are not parsed as numbers;
  * a site export writes the bone lines as the visible drawing
    (DXF <grave>_LINES on, <grave>_MASK_EDGE off; GPKG bone_lines; PDF).

Run: python tests/test_site_export.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boneseg.data.crs_guess import tm0_to_htrs96  # noqa: E402
from boneseg.data.gcp_parse import parse_gcp_text  # noqa: E402
from boneseg.data.site import export_site, scan_site  # noqa: E402
from boneseg.postprocessing.vectorize import (  # noqa: E402
    mask_to_rings,
    smooth_staircase_ring,
)


def _axis_aligned_fraction(ring) -> float:
    a = np.asarray(ring, dtype=np.float64)
    d = np.diff(a, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 0]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    return float(np.mean((ang < 1) | (ang > 89)))


def test_outlines_are_smooth_and_on_the_edge() -> None:
    m = np.zeros((400, 400), np.uint8)
    cv2.ellipse(m, (200, 200), (150, 40), 20, 0, 360, 1, -1)
    ring = np.array(max(mask_to_rings(m, None), key=len))
    ring[:, 1] *= -1                       # plain image: output Y is negated
    assert _axis_aligned_fraction(ring) < 0.2, "outline still follows the pixel grid"
    c, s = np.cos(np.radians(20)), np.sin(np.radians(20))
    d = ring - [200, 200]
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    err = np.abs(np.sqrt((u / 150) ** 2 + (v / 40) ** 2) - 1) * 40
    assert err.mean() < 0.6, f"outline {err.mean():.2f} px off the edge"


def test_old_staircase_rings_are_smoothed() -> None:
    m = np.zeros((120, 120), np.uint8)
    cv2.circle(m, (60, 60), 40, 1, -1)
    from rasterio.features import shapes
    from rasterio.transform import Affine
    # exactly how exports before 2026-10 traced it: raw pixel edges through
    # a (slightly sheared) GCP affine
    t = Affine(0.001, 0.0002, 574800.0, 0.00005, -0.001, 5028180.0)
    stair = [tuple(p) for geom, val in shapes(m, mask=m > 0, transform=t) if val == 1
             for p in [geom["coordinates"][0]]][0]
    out = smooth_staircase_ring(stair)
    assert out is not stair, "staircase not detected"
    assert _axis_aligned_fraction(out) < 0.2
    smooth = mask_to_rings(m, None)[0]
    assert smooth_staircase_ring(smooth) is smooth, "smooth rings must be left alone"


def test_tm0_points_become_htrs96() -> None:
    pts = [{"id": "37", "e": 1366377.3765, "n": 5178740.9672},
           {"id": "38", "e": 1366377.0259, "n": 5178740.7938}]
    conv = tm0_to_htrs96(pts)
    assert conv is not None
    assert abs(conv[0]["e"] - 574804.26) < 0.05 and abs(conv[0]["n"] - 5028185.95) < 0.05
    assert tm0_to_htrs96([{"e": 574786.6, "n": 5028184.6}]) is None


def test_underscore_ids_are_not_numbers() -> None:
    pts = parse_gcp_text("157_1,574809.3360,5028183.3005\n157_2,574809.9172,5028183.3723\n")
    assert pts[0]["id"] == "157_1" and abs(pts[0]["e"] - 574809.336) < 1e-6


def test_site_export_writes_bone_lines() -> None:
    import ezdxf
    import pyogrio

    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / "GR1"
        g.mkdir()
        x0, y0 = 574800.0, 5028180.0
        line = [[x0 + i * 0.01, y0 + 0.2 * np.sin(i / 20)] for i in range(150)]
        ring = [[x0, y0 - 0.5], [x0 + 1, y0 - 0.5], [x0 + 1, y0 - 0.3], [x0, y0 - 0.3], [x0, y0 - 0.5]]
        crs = {"type": "name", "properties": {"name": "EPSG:3765"}}
        (g / "GR1_centerlines.geojson").write_text(json.dumps({"type": "FeatureCollection", "crs": crs,
            "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": line}}]}))
        (g / "GR1_polygons.geojson").write_text(json.dumps({"type": "FeatureCollection", "crs": crs,
            "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
        site = scan_site([str(g)], None, True)
        assert site["mode"] == "map" and site["crs"] == "EPSG:3765"
        assert site["graves"][0]["georeferenced"]
        res = export_site(site["graves"], str(Path(tmp) / "plan"), site["crs"], ["dxf", "gpkg", "pdf"],
                          photos=False, outlines=True, centerlines=True)
        doc = ezdxf.readfile(str(Path(tmp) / "plan.dxf"))
        layers = {l.dxf.name: l.is_on() for l in doc.layers if l.dxf.name.startswith("GR1")}
        assert layers.get("GR1_LINES") is True, layers   # the drawing, visible
        assert any(e.dxf.layer == "GR1_LINES" for e in doc.modelspace().query("LWPOLYLINE"))
        names = {n for n, _ in pyogrio.list_layers(str(Path(tmp) / "plan.gpkg"))}
        assert {"bone_lines", "mask_edges"} <= names
        assert (Path(tmp) / "plan.pdf").stat().st_size > 1000
        assert res["graves"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all site/vector tests passed")
