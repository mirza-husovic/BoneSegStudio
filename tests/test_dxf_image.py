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


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    photo_dir = td / "src"
    photo_dir.mkdir()
    photo = photo_dir / "DSCF_test.jpg"
    Image.new("RGB", (W, H), (120, 100, 80)).save(photo)

    # --- georeferenced: insert = T*(0,h), u = (a,d), v = (-b,-e) ---------- #
    out = td / "geo" / "grave.dxf"
    out.parent.mkdir()
    save_dxf(rings, lines, out, GEOREF, image_path=photo, image_size=(W, H))
    doc, img = read_image_entity(out)
    ix, iy = T * (0.0, float(H))
    assert abs(img.dxf.insert.x - ix) < 1e-9 and abs(img.dxf.insert.y - iy) < 1e-9
    assert abs(img.dxf.u_pixel.x - T.a) < 1e-12 and abs(img.dxf.u_pixel.y - T.d) < 1e-12
    assert abs(img.dxf.v_pixel.x + T.b) < 1e-12 and abs(img.dxf.v_pixel.y + T.e) < 1e-12
    assert (out.parent / photo.name).is_file(), "photo not copied next to the DXF"
    # top-right pixel corner must land where the affine puts it:
    tr_world = T * (W, 0.0)
    tr_dxf = (img.dxf.insert.x + W * img.dxf.u_pixel.x + H * img.dxf.v_pixel.x,
              img.dxf.insert.y + W * img.dxf.u_pixel.y + H * img.dxf.v_pixel.y)
    assert abs(tr_dxf[0] - tr_world[0]) < 1e-6 and abs(tr_dxf[1] - tr_world[1]) < 1e-6
    # vectors still present, image entity written FIRST (stays underneath)
    ents = list(doc.modelspace())
    assert ents[0].dxftype() == "IMAGE"
    assert sum(1 for e in ents if e.dxftype() == "LWPOLYLINE") == 2
    print("georeferenced IMAGE underlay OK (corners match affine)")

    # --- pixel mode: insert (0,-h), unit u/v — QGIS-pixel convention ------ #
    out2 = td / "px" / "grave.dxf"
    out2.parent.mkdir()
    px_rings = [[(10.0, -10.0), (100.0, -10.0), (100.0, -80.0)]]
    save_dxf(px_rings, [], out2, None, image_path=photo, image_size=(W, H))
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
