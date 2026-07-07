"""Central configuration for BoneSeg Studio.

All defaults mirror the values used by the original ``model330b3/predict.py``
so that this application reproduces the validated inference pipeline exactly:

    patch 512 / stride 256 (overlap-averaging), threshold 0.5,
    min component 32 px, optional 4-way TTA, AMP on CUDA,
    spline smoothing 200 / density 4 / downsample 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH: Path = Path(
    r"C:\Users\mirza\Desktop\modeli\model367b3\best_bone_model.pth"
)

LOGS_DIR: Path = PROJECT_ROOT / "logs"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Staging folder for corrected training pairs saved from the app ("Save to
# training set"). Kept SEPARATE from UNET_DATASET_FINAL4 on purpose — the
# user merges staged pairs into the real dataset manually. Layout mirrors
# the dataset: images/ + masks/ + overlays/ + dataset_manifest.csv.
TRAINING_STAGING_DIR: Path = Path.home() / "Desktop" / "UNET_STAGING"

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
)

# The browser editor works at this capped long-side resolution: full-res
# orthophotos (100+ MP) would exhaust browser canvas memory. The Canvas 2D
# editor handles 4096 comfortably (no WebGL texture budget involved), and
# edits are applied back to the FULL-resolution mask as a sparse diff —
# untouched regions keep their original full-res detail. Skeleton
# centerlines are width-invariant (medial axis), so the edit-resolution cap
# doesn't degrade the vector deliverable.
MAX_EDIT_DIM: int = 4096


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------
@dataclass
class InferenceSettings:
    """Parameters of the sliding-window forward pass (expensive stage)."""

    patch: int = 512            # window size the model was trained on
    stride: int = 256           # overlap-averaging stride (training-eval value)
    use_tta: bool = False       # 4-way flip TTA (+~0.006 Dice, ~4x slower)
    use_amp: bool = True        # mixed precision on CUDA
    batch_size: int = 8         # tiles per forward pass (eval mode -> identical output)


@dataclass
class PostprocessSettings:
    """Parameters of mask cleanup + skeleton + vectors (cheap stage).

    These can be re-applied to a cached probability map without re-running
    the network, which is what the UI's "Apply settings" button does.
    """

    threshold: float = 0.5          # sigmoid threshold (training-eval value)
    min_component_px: int = 32      # drop mask blobs smaller than this

    # Skeletonization
    prune_branch_px: int = 20       # remove terminal skeleton branches shorter than this
    min_skeleton_px: int = 40       # remove isolated skeleton fragments shorter than this

    # Centerline spline smoothing (identical to predict.py)
    spline_smooth: float = 200.0    # lower bound for splprep's `s`
    spline_density: int = 4         # resampled points per input point
    spline_downsample: int = 5      # keep every Kth medial-axis point pre-fit


@dataclass
class DisplaySettings:
    """Purely visual parameters — never affect exported geometry."""

    overlay_opacity: float = 0.55   # predict.py blended mask at 0.55
    overlay_color: tuple[int, int, int] = (0, 200, 220)  # cyan


@dataclass
class ExportSettings:
    """Which artifacts to write and where."""

    output_dir: Path = OUTPUTS_DIR
    save_mask: bool = True
    save_overlay: bool = True
    save_skeleton: bool = True
    save_geojson: bool = True
    save_dxf: bool = True
    save_svg: bool = False
    save_plate: bool = False        # A4 catalog plate PDF (scale bar + north arrow)

    # Merge every exported grave into ONE site-wide GeoJSON pair + DXF
    # (master_*.geojson / master.dxf in output_dir) — see boneseg/data/master.py
    append_master: bool = False

    # Plate title block
    plate_site: str = ""            # site / excavation name printed on plates
    plate_note: str = ""            # free-text note printed under the photo


@dataclass
class AppConfig:
    """Top-level bundle handed to the pipeline and the UI."""

    model_key: str = "model367b3"
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    postprocess: PostprocessSettings = field(default_factory=PostprocessSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    export: ExportSettings = field(default_factory=ExportSettings)


def ensure_runtime_dirs() -> None:
    """Create logs/ and outputs/ if missing (safe to call repeatedly)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
