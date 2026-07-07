"""Device selection and sliding-window inference."""

from boneseg.inference.device import DeviceInfo, detect_device
from boneseg.inference.engine import InferenceEngine, predict_full

__all__ = ["DeviceInfo", "detect_device", "InferenceEngine", "predict_full"]
