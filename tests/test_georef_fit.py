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

# --- REAL user case (2026-07-08): 4-corner near-rectangle GR30 ------------ #
# A rectangle is affine-symmetric: cycled pairings fit with the SAME rms,
# only by squashing the photo by aspect² (the reported "stretched" QGIS
# look). The least-distortion + click-order tie-break must pick the truth.
GR30 = [(576362.2191, 5016382.8409), (576363.1558, 5016383.1340),
        (576363.0224, 5016383.5597), (576362.0796, 5016383.2474)]
GR30_5 = GR30 + [(576362.3657, 5016383.0922)]
# Simulated photo: 1000 px/m, rotated 30 deg, y down (det<0), tiny noise.
_ang = np.radians(30)
_rot = np.array([[np.cos(_ang), -np.sin(_ang)], [np.sin(_ang), np.cos(_ang)]])
_org = np.array([576362.0, 5016382.5])
_rng3 = np.random.default_rng(11)


def _to_px(pts):
    out = []
    for e, nn in pts:
        v = _rot @ (np.array([e, nn]) - _org) * 1000.0
        out.append((float(v[0] + 300 + _rng3.normal(0, 2)),
                    float(-v[1] + 1500 + _rng3.normal(0, 2))))
    return out


clicks4 = _to_px(GR30)                      # clicked in file order 1-4
for pool in (GR30, GR30_5):                 # without and with the extra 5th
    a4, rms4, mir4, sec4 = auto_assign_gcps(clicks4, pool)
    assert a4 == [0, 1, 2, 3], f"wrong pairing chosen: {a4} (pool={len(pool)})"
    assert not mir4
    _fitA, _resA, rmsA = fit_affine_gcps(
        [{"px": px, "py": py, "e": pool[j][0], "n": pool[j][1]}
         for (px, py), j in zip(clicks4, a4)])
    assert rmsA < 0.01
print("symmetric-rectangle tie-break OK (correct pairing, no squash)")

# clicks in a DIFFERENT order than the file still resolve correctly
perm = [2, 0, 3, 1]
clicks_shuffled = [clicks4[i] for i in perm]
a5, _r5, _m5, _s5 = auto_assign_gcps(clicks_shuffled, GR30_5)
assert a5 == perm, a5
print("shuffled-click rectangle OK")

# --- homography: oblique photo (perspective) ------------------------------ #
from boneseg.data.georef_fit import (apply_homography, fit_homography_gcps,
                                     georef_from_gcps as gfg, rectify_params)

# Ground-truth homography with a real perspective term (oblique camera).
H_TRUE = np.array([
    [0.0015, -0.0006, 457120.0],
    [0.0004,  0.0011, 4795320.0],
    [1.2e-5,  3.0e-5, 1.0],
])
px6 = [(100, 100), (1800, 150), (1700, 1200), (200, 1100), (950, 600), (500, 300)]
w6 = apply_homography(H_TRUE, px6)
gcps_h = [{"px": x, "py": y, "e": e, "n": n}
          for (x, y), (e, n) in zip(px6, w6)]

h_fit, res_h, rms_h = fit_homography_gcps(gcps_h)
assert rms_h < 1e-4, rms_h
proj = apply_homography(h_fit, [(400, 700)])
truth = apply_homography(H_TRUE, [(400, 700)])
assert np.allclose(proj, truth, atol=1e-4)
print(f"homography exact fit OK (rms={rms_h:.2e} m)")

# georef_from_gcps: oblique -> homography attached, affine kept as approx
g_obl, res_obl, rms_obl = gfg(gcps_h, 3765)
assert g_obl.homography is not None, "oblique photo should get a homography"
assert rms_obl < 1e-4 and "perspective" in str(g_obl)
# nadir (pure affine) points -> NO homography even with 4+ points
g_nad, _res_n, rms_n = gfg(gcps, 3765)   # gcps = exact affine points above
assert g_nad.homography is None and rms_n < 1e-4
print("mode selection OK (oblique->homography, nadir->affine)")

# rectified grid: north-up, K maps photo corners inside the grid
grid, K, ow, oh = rectify_params(g_obl, 1920, 1280)
assert grid.b == 0 and grid.d == 0 and grid.a > 0 and grid.e < 0
gc = apply_homography(K, [(0, 0), (1920, 0), (1920, 1280), (0, 1280)])
assert gc[:, 0].min() > -1.5 and gc[:, 0].max() < ow + 1.5
assert gc[:, 1].min() > -1.5 and gc[:, 1].max() < oh + 1.5
print(f"rectify grid OK ({ow}x{oh}, gsd={grid.a*1000:.1f} mm)")

# vectors go through the homography EXACTLY
from boneseg.postprocessing import polylines_px_to_output
line_px = [(150.0, 200.0), (900.0, 640.0), (1500.0, 1000.0)]
out_line = polylines_px_to_output([line_px], g_obl)[0]
expect = apply_homography(H_TRUE, line_px)
assert np.allclose(np.array(out_line), expect, atol=1e-3)
print("polylines through homography OK")

print("GEOREF FIT TEST PASSED")
