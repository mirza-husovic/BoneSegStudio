"""Unit test: catalog plate PDF (georeferenced + plain + scale/north math).

Run:  python tests/test_plate.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.transform import Affine

from boneseg.data.plate import (_north_angle_page_deg, _pick_scale,
                                save_plate_pdf)
from boneseg.data.readers import GeoRef

rng = np.random.default_rng(0)
img = (rng.normal(140, 25, (1200, 1600, 3))).clip(0, 255).astype(np.uint8)
mask = np.zeros((1200, 1600), np.uint8)
mask[300:330, 200:900] = 1
mask[500:700, 700:740] = 1
polys = [[(200.0, 315.0), (900.0, 315.0)], [(720.0, 500.0), (720.0, 700.0)]]

with tempfile.TemporaryDirectory() as td:
    out = Path(td)

    # --- georeferenced: 1 px = 1.2 mm, EPSG:3765, 15 deg rotated ---------
    t = Affine.translation(457000, 4795000) * Affine.rotation(15) * Affine.scale(0.0012, -0.0012)
    georef = GeoRef(t, rasterio.crs.CRS.from_epsg(3765))

    scale = _pick_scale(1600 * 0.0012, 1200 * 0.0012)
    assert scale == 20, scale                       # 1.92 x 1.44 m -> 1:20
    ang = _north_angle_page_deg(georef, 1600, 1200)
    # Image axes rotated CCW vs world => north tips CW (negative) on page.
    assert abs(ang + 15.0) < 0.01, ang
    print(f"scale 1:{scale}, north angle {ang:.1f} deg OK")

    p1 = save_plate_pdf(img, mask, polys, georef, out / "plate_geo.pdf",
                        label="GROB 12", site="Test site", note="nota",
                        model_name="model367b3")
    assert p1 is not None and p1.stat().st_size > 10_000
    print(f"geo plate OK ({p1.stat().st_size} bytes)")

    # --- north-up affine: angle 0 -----------------------------------------
    t2 = Affine.translation(457000, 4795000) * Affine.scale(0.0012, -0.0012)
    assert abs(_north_angle_page_deg(GeoRef(t2, None), 1600, 1200)) < 0.01
    print("north-up angle 0 OK")

    # --- plain image (no georef): plate still renders ---------------------
    p2 = save_plate_pdf(img, mask, polys, None, out / "plate_plain.pdf",
                        label="GROB 13", site="Test site")
    assert p2 is not None and p2.stat().st_size > 10_000
    print(f"plain plate OK ({p2.stat().st_size} bytes)")

print("PLATE TEST PASSED")
