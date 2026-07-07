"""Unit test: GCP affine fitting + world file round-trip.

Run:  python tests/test_georef_fit.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rasterio.transform import Affine

from boneseg.data.georef_fit import (fit_affine_gcps, georef_from_gcps,
                                     write_world_file)

# Ground truth: rotated + scaled affine like a real georeferenced photo.
T = Affine.translation(457123.45, 4795321.98) * Affine.rotation(-25) * Affine.scale(0.0015, -0.0015)

px_pts = [(100, 120), (1500, 200), (800, 1000), (300, 900)]
gcps = []
for px, py in px_pts:
    e, n = T * (px, py)
    gcps.append({"px": px, "py": py, "e": e, "n": n})

# --- exact fit from 4 perfect points (1e-4 m = float64 precision at ~5e6) ---
fit, res, rms = fit_affine_gcps(gcps)
assert rms < 1e-4, rms
for k in "abcdef":
    assert abs(getattr(fit, k) - getattr(T, k)) < 1e-4, k
print(f"exact fit OK, rms={rms:.2e} m")

# --- noisy fit: 2 cm noise -> residuals ~cm ---
rng = np.random.default_rng(1)
noisy = [{**g, "e": g["e"] + rng.normal(0, 0.02), "n": g["n"] + rng.normal(0, 0.02)}
         for g in gcps]
fit2, res2, rms2 = fit_affine_gcps(noisy)
print(f"noisy fit rms={rms2*100:.1f} cm, per-point={[round(r*100,1) for r in res2]} cm")
assert rms2 < 0.05

# --- 3-point minimum is exact ---
fit3, res3, rms3 = fit_affine_gcps(gcps[:3])
assert rms3 < 1e-4
print("3-point exact fit OK")

# --- too few points rejected ---
try:
    fit_affine_gcps(gcps[:2])
    raise AssertionError("should have raised")
except ValueError:
    print("2-point rejection OK")

# --- GeoRef wrapper: EPSG + local grid ---
g, _, _ = georef_from_gcps(gcps, 3765)
assert g.crs is not None and g.crs.to_epsg() == 3765
g_local, _, _ = georef_from_gcps(gcps, None)
assert g_local.crs is None
print("GeoRef wrapper OK (EPSG:3765 + local grid)")

# --- world file round-trip: rasterio must read back the same transform ---
import rasterio
from PIL import Image

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "wf_test.png"
    Image.new("RGB", (50, 40)).save(tmp)
    write_world_file(tmp, g)
    with rasterio.open(tmp) as src:
        for k in "abcdef":
            assert abs(getattr(src.transform, k) - getattr(T, k)) < 1e-4, (k, src.transform)
print("world file round-trip OK (rasterio re-reads identical transform)")

print("GEOREF FIT TEST PASSED")
