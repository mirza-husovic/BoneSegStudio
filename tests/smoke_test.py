"""Headless smoke test — exercises the whole pipeline without the web UI.

Loads the real model, runs inference on a small synthetic image, runs
postprocessing (mask cleanup + skeleton + vectors) and exports every
artifact type. Verifies the layers are wired together and the model
checkpoint loads; it does NOT assert prediction quality.

Run:
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boneseg.config import (  # noqa: E402
    AppConfig, DisplaySettings, ExportSettings, InferenceSettings, PostprocessSettings,
)
from boneseg.pipeline import BonePipeline  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="boneseg_smoke_"))

    # synthetic 600x800 image with a few bright strokes (bone-like)
    rng = np.random.default_rng(0)
    img = (rng.integers(40, 90, size=(600, 800, 3))).astype(np.uint8)
    img[250:350, 100:700] = 200
    img[100:500, 380:430] = 190
    src = tmp / "synthetic.png"
    Image.fromarray(img).save(src)
    print(f"[1/7] test image written: {src}")

    pipeline = BonePipeline(AppConfig())
    print(f"[2/7] device: {pipeline.engine.device_info.label}")

    # Permissive postprocessing: the synthetic image is not a real grave, so
    # a low threshold guarantees SOME foreground for the export assertions.
    pp = PostprocessSettings(threshold=0.05, min_component_px=0,
                             prune_branch_px=0, min_skeleton_px=0)
    result = pipeline.process(src, InferenceSettings(use_tta=False), pp)
    print(f"[3/7] inference {result.inference_seconds:.1f}s, "
          f"postprocess {result.postprocess_seconds:.1f}s")
    print(f"      mask fg={result.fg_pixels} px, components={result.n_components}, "
          f"centerlines={len(result.polylines_out)}")

    assert result.mask.shape == img.shape[:2], "mask shape mismatch"
    assert result.skeleton.shape == img.shape[:2], "skeleton shape mismatch"
    assert result.prob.dtype == np.float32, "prob should be float32"
    print("[4/7] shape/dtype assertions passed")

    written = pipeline.export(
        result,
        ExportSettings(output_dir=tmp / "out", save_svg=True),
        DisplaySettings(),
        out_dir=tmp / "out",
    )
    for p in written:
        assert Path(p).exists() and Path(p).stat().st_size > 0, f"empty export: {p}"
    assert any(Path(p).suffix == ".dxf" for p in written), "DXF missing from export"
    print(f"[5/7] exported {len(written)} files, all non-empty:")
    for p in written:
        print(f"      {Path(p).name}  ({Path(p).stat().st_size:,} B)")

    # ---- DXF round-trip ----
    import ezdxf  # noqa: E402
    dxf_path = next(p for p in written if Path(p).suffix == ".dxf")
    doc = ezdxf.readfile(str(dxf_path))
    layers = {ly.dxf.name for ly in doc.layers}
    assert {"BONE_OUTLINE", "BONE_CENTERLINE"} <= layers, f"bad DXF layers: {layers}"
    n_ent = len(doc.modelspace())
    assert n_ent > 0, "DXF has no entities"
    print(f"[6/7] DXF round-trip OK: layers {sorted(layers)}, {n_ent} entities")

    # ---- master append: two images, then re-export first (replace) ----
    import json  # noqa: E402
    master_dir = tmp / "master"
    exp_master = ExportSettings(output_dir=master_dir, save_mask=False,
                                save_overlay=False, save_skeleton=False,
                                save_geojson=False, save_dxf=False,
                                save_svg=False, append_master=True)
    src2 = tmp / "synthetic2.png"
    Image.fromarray(np.rot90(img).copy()).save(src2)
    result2 = pipeline.process(src2, InferenceSettings(use_tta=False), pp)
    for res in (result, result2, result):   # 3rd call re-exports image 1
        pipeline.export(res, exp_master, DisplaySettings(),
                        out_dir=master_dir / res.source_path.stem)

    mdxf = ezdxf.readfile(str(master_dir / "master.dxf"))
    mlayers = {ly.dxf.name for ly in mdxf.layers}
    assert any(n.startswith("SYNTHETIC_") for n in mlayers), f"master layers: {mlayers}"
    assert any(n.startswith("SYNTHETIC2_") for n in mlayers), f"master layers: {mlayers}"
    gj = json.loads((master_dir / "master_centerlines.geojson").read_text(encoding="utf-8"))
    stems = {f["properties"]["image"] for f in gj["features"]}
    assert stems == {"synthetic", "synthetic2"}, f"master stems: {stems}"
    n1 = sum(1 for f in gj["features"] if f["properties"]["image"] == "synthetic")
    assert n1 == len(result.polylines_out), "re-export duplicated features"
    print(f"[7/7] master OK: {len(mdxf.modelspace())} entities, "
          f"stems {sorted(stems)}, replace semantics hold")

    print(f"\nSMOKE TEST PASSED. Artifacts in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
