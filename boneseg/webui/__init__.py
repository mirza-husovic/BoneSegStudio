"""Local web UI (FastAPI backend + vanilla Canvas 2D frontend).

This package replaced the original Gradio interface: the Gradio ImageEditor
(WebGL) crashed with a white screen on tool switches and lost the user's
edits, and its canvas had no reliable pan. The frontend here is plain
HTML/JS drawing on a 2D canvas — no WebGL, no framework — and the backend
is a thin FastAPI layer over :class:`boneseg.pipeline.BonePipeline`.
"""

from boneseg.webui.server import create_app

__all__ = ["create_app"]
