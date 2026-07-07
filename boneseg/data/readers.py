"""Image loading.

Same behavior as ``predict.py``: geo-tagged TIFFs are read with rasterio and
keep their affine transform + CRS; everything else (JPG/PNG/plain TIFF) is
read with PIL and processed in pixel coordinates. rasterio is optional — if
it is not importable, TIFFs simply fall back to PIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from boneseg.config import SUPPORTED_EXTENSIONS
from boneseg.logging_setup import get_logger

logger = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = None  # allow >89-megapixel orthophotos

try:
    import rasterio

    HAS_RASTERIO = True
except Exception:  # pragma: no cover - optional dependency
    HAS_RASTERIO = False


@dataclass(frozen=True)
class GeoRef:
    """Georeferencing carried from input raster to all outputs.

    ``transform`` is a rasterio affine transform; ``crs`` a rasterio CRS.
    Typed as Any so the rest of the codebase does not import rasterio.
    """

    transform: Any
    crs: Any

    def __str__(self) -> str:
        return f"CRS={self.crs}" if self.crs else "affine only"


class UnsupportedImageError(ValueError):
    """Raised for files the application cannot process."""


def is_supported(path: Path) -> bool:
    """True if the extension is one the app accepts."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _sidecar_georef(path: Path) -> GeoRef | None:
    """GeoRef from a user GCP sidecar (written by the in-app georeferencer),
    so plain photos come back georeferenced on EVERY read — single open,
    batch and batch-reopen alike. Lazy import avoids a module cycle."""
    from boneseg.data.georef_fit import gcps_sidecar_path, georef_from_gcps

    sidecar = gcps_sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        import json

        data = json.loads(sidecar.read_text(encoding="utf-8"))
        g, _res, rms = georef_from_gcps(data["gcps"], data.get("epsg"))
        logger.info("GCP sidecar applied: %s (rms %.3f m)", sidecar.name, rms)
        return g
    except Exception:
        logger.exception("Ignoring unreadable GCP sidecar %s", sidecar)
        return None


def read_image(path: Path) -> tuple[np.ndarray, GeoRef | None]:
    """Read an image as RGB uint8.

    Returns
    -------
    (array, georef)
        ``georef`` is None for non-georeferenced inputs (unless a GCP
        sidecar exists next to the file); downstream outputs otherwise
        fall back to pixel coordinates.

    Raises
    ------
    UnsupportedImageError
        For unsupported extensions.
    OSError / ValueError
        For corrupted files (caller converts these to a UI message).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not is_supported(path):
        raise UnsupportedImageError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    ext = path.suffix.lower()
    if HAS_RASTERIO and ext in (".tif", ".tiff"):
        try:
            with rasterio.open(str(path)) as src:
                bands = src.count
                if bands >= 3:
                    arr = src.read([1, 2, 3])
                elif bands == 1:
                    g = src.read(1)
                    arr = np.stack([g, g, g], axis=0)
                else:
                    arr = src.read(list(range(1, min(bands, 3) + 1)))
                    if arr.shape[0] < 3:
                        # pad missing channels with the last one
                        pad = np.repeat(arr[-1:, ...], 3 - arr.shape[0], axis=0)
                        arr = np.concatenate([arr, pad], axis=0)
                arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
                has_transform = (
                    src.transform is not None and not src.transform.is_identity
                )
                georef = GeoRef(src.transform, src.crs) if has_transform else None
                if georef is None:
                    georef = _sidecar_georef(path)
                return arr, georef
        except Exception as exc:
            logger.warning("rasterio read failed for %s (%s); trying PIL", path.name, exc)

    img = np.array(Image.open(path).convert("RGB"))
    return img, _sidecar_georef(path)
