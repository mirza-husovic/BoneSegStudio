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

# --- auto-assignment: shuffled points + extras, any click order ---------- #
from boneseg.data.georef_fit import auto_assign_gcps

rng2 = np.random.default_rng(7)
click_pts = [(120.0, 90.0), (1500.0, 260.0), (760.0, 1040.0),
             (340.0, 880.0), (1210.0, 700.0)]
world = [T * c for c in click_pts]
# Extra surveyed points that are NOT on the photo + shuffled order.
extras = [T * (2600.0, 1800.0), T * (-500.0, 2400.0), T * (3000.0, -400.0)]
pool = [(e + rng2.normal(0, 0.01), n + rng2.normal(0, 0.01))
        for e, n in world] + list(extras)
order = rng2.permutation(len(pool))
shuffled = [pool[i] for i in order]
truth = {int(np.where(order == i)[0][0]): i for i in range(len(world))}

assign, rms_a, mirrored, second = auto_assign_gcps(click_pts, shuffled)
assert not mirrored
for click_i, pool_j in enumerate(assign):
    assert shuffled[pool_j] == pool[click_i], (click_i, pool_j)
assert rms_a < 0.05, rms_a
assert second > 5 * rms_a  # unambiguous layout
print(f"auto-assign OK (rms={rms_a*100:.1f} cm, runner-up {second:.2f} m)")

# --- swapped E/N is detected as a mirrored best fit ---------------------- #
swapped = [(n, e) for e, n in shuffled]
_a, _r, mirrored_sw, _s = auto_assign_gcps(click_pts, swapped)
assert mirrored_sw, "swapped E/N should look mirrored"
print("swapped E/N detection OK")

# --- guard rails ---------------------------------------------------------- #
try:
    auto_assign_gcps(click_pts[:2], shuffled)
    raise AssertionError("should have raised (2 clicks)")
except ValueError:
    pass
try:
    auto_assign_gcps(click_pts, shuffled[:3])
    raise AssertionError("should have raised (fewer points than clicks)")
except ValueError:
    pass
try:
    auto_assign_gcps([(0, 0), (100, 0), (200, 0), (300, 0)], shuffled)
    raise AssertionError("should have raised (collinear clicks)")
except ValueError:
    pass
print("auto-assign guard rails OK")

print("GEOREF FIT TEST PASSED")
