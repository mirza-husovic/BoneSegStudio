"""BoneSeg Studio — local desktop application for archaeological bone
segmentation, skeletonization and vectorization.

The package is organized as independent layers:

    config          — dataclasses with all tunable settings and paths
    models          — model registry + weight loading (extensible to YOLO/SAM2)
    inference       — device selection and sliding-window inference engine
    data            — image readers (PIL / rasterio) and output writers
    postprocessing  — mask cleanup, skeletonization, vectorization
    pipeline        — orchestrates one image end-to-end
    webui           — FastAPI backend + vanilla Canvas 2D frontend

Nothing in ``boneseg`` imports FastAPI except ``boneseg.webui``, so the
whole processing pipeline is usable headless (scripts, notebooks, CLI).
"""

__version__ = "1.0.0"
APP_NAME = "BoneSeg Studio"
