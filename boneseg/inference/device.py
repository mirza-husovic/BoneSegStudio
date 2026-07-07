"""Automatic CUDA detection with CPU fallback.

The user never configures the device: CUDA is used whenever available and
functional, otherwise the application silently runs on CPU (slower but
correct).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from boneseg.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device plus a human-readable label for the UI."""

    device: torch.device
    label: str          # e.g. "GPU — NVIDIA GeForce RTX 4060" or "CPU"
    is_cuda: bool


def detect_device() -> DeviceInfo:
    """Return the best available device.

    A CUDA build with no working driver can make ``is_available()`` raise or
    the first allocation fail, so the probe is wrapped defensively — any
    problem degrades to CPU instead of crashing the app.
    """
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            info = DeviceInfo(torch.device("cuda"), f"GPU — {name}", True)
            logger.info("CUDA available: %s", name)
            return info
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CUDA probe failed (%s); falling back to CPU", exc)

    logger.info("CUDA not available — using CPU")
    return DeviceInfo(torch.device("cpu"), "CPU", False)
