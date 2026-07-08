"""Unit test: photo embedded in DXF as a georeferenced IMAGE underlay.

Run:  python tests/test_dxf_image.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ezdxf
import rasterio
from PIL import Image
from rasterio.transform import Affine

from boneseg.data.readers import GeoRef
from boneseg.data.writers import save_dxf

W, H = 400, 300
T = Affine.translation(576362.0, 5016383.0) * Affine.rotation(-25) * Affine.scale(0.002, -0.002)
GEOREF = GeoRef(T, rasterio.crs.CRS.from_epsg(3765))

rings = [[T * p for p in [(10, 10), (100, 10), (100, 80), (10, 80)]]]
lines = [[T * p for p in [(20, 20), (90, 70)]]]


def read_image_entity(dxf_path: Path):
    doc = ezdxf.readfile(str(dxf_path))
    imgs = doc.modelspace().query("IMAGE")
    assert len(imgs) == 1, f"expected 1 IMAGE, got {len(imgs)}"
    assert "PHOTO" in {ly.dxf.name for ly in doc.layers}
    return doc, imgs[0]


import numpy as np

PHOTO_RGB = np.full((H, W, 3), (120, 100, 80), dtype=np.uint8)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # --- georeferenced: underlay is RECTIFIED (north-up, axis-aligned u/v;
    # AutoCAD cannot display sheared/rotated rasters at all) --------------- #
    out = td / "geo" / "grave.dxf"
    out.parent.mkdir()
    save_dxf(rings, lines, out, GEOREF, image_rgb=PHOTO_RGB)
    doc, img = read_image_entity(out)
    assert (out.parent / "grave_photo.jpg").is_file(), "underlay JPEG missing"
    # AutoCAD resolves ABSOLUTE image paths most reliably (ezdxf docs);
    # raster variables must exist with visible frames.
    idef = doc.objects.query("IMAGEDEF")[0]
    assert Path(idef.dxf.filename).is_absolute(), idef.dxf.filename
    assert Path(idef.dxf.filename).is_file()
    rv = doc.objects.query("RASTERVARIABLES")[0]
    assert rv.dxf.frame == 1
    # axis-aligned: no rotation/shear components at all
    assert img.dxf.u_pixel.y == 0.0 and img.dxf.v_pixel.x == 0.0
    assert img.dxf.u_pixel.x > 0 and img.dxf.v_pixel.y > 0
    # the underlay must cover the world footprint of the warped photo corners
    from boneseg.data.georef_fit import apply_homography, rectify_params
    grid, k, out_w, out_h = rectify_params(GEOREF, W, H)
    assert abs(img.dxf.u_pixel.x - grid.a) < 1e-9
    corners = [T * (0, 0), T * (W, 0), T * (W, H), T * (0, H)]
    minx = min(c[0] for c in corners); miny = min(c[1] for c in corners)
    maxx = max(c[0] for c in corners); maxy = max(c[1] for c in corners)
    x0, y0 = img.dxf.insert.x, img.dxf.insert.y
    x1 = x0 + out_w * img.dxf.u_pixel.x
    y1 = y0 + out_h * img.dxf.v_pixel.y
    gsd = grid.a
    assert abs(x0 - minx) < 2 * gsd and abs(y0 - miny) < 2 * gsd, (x0, minx, y0, miny)
    assert abs(x1 - maxx) < 2 * gsd and abs(y1 - maxy) < 2 * gsd
    # vectors still present, image entity written FIRST (stays underneath)
    ents = list(doc.modelspace())
    assert ents[0].dxftype() == "IMAGE"
    assert sum(1 for e in ents if e.dxftype() == "LWPOLYLINE") == 2
    print("georeferenced IMAGE underlay OK (rectified, axis-aligned, covers footprint)")

    # --- pixel mode: insert (0,-h), unit u/v — QGIS-pixel convention ------ #
    out2 = td / "px" / "grave.dxf"
    out2.parent.mkdir()
    px_rings = [[(10.0, -10.0), (100.0, -10.0), (100.0, -80.0)]]
    save_dxf(px_rings, [], out2, None, image_rgb=PHOTO_RGB)
    _doc2, img2 = read_image_entity(out2)
    assert img2.dxf.insert.x == 0.0 and img2.dxf.insert.y == -float(H)
    assert img2.dxf.u_pixel.x == 1.0 and img2.dxf.v_pixel.y == 1.0
    print("pixel-mode IMAGE underlay OK")

    # --- no image args -> classic DXF, no IMAGE entity -------------------- #
    out3 = td / "plain.dxf"
    save_dxf(rings, lines, out3, GEOREF)
    doc3 = ezdxf.readfile(str(out3))
    assert len(doc3.modelspace().query("IMAGE")) == 0
    print("plain DXF (no image) unchanged OK")

print("DXF IMAGE TEST PASSED")
