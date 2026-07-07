"""FastAPI backend for BoneSeg Studio.

Design rules (stability first):

* Every mutating endpoint acquires ``Studio.lock`` with a short timeout and
  answers **409 busy** instead of queueing behind a long inference — the
  frontend disables its buttons during a job, so a 409 only happens if the
  user races the UI.
* Long work (inference, batch) runs in ONE background thread; its state is
  polled via ``GET /api/job``. A cooperative cancel flag aborts between
  tile batches / batch items.
* The mask travels to the browser at a capped edit resolution
  (``MAX_EDIT_DIM``), but edits come back as a **sparse diff** against the
  exact mask the browser was given (``mask_basis``): only changed pixels
  are upscaled and applied to the full-resolution mask, so untouched
  regions keep full-resolution detail.
* ``mask_version`` guards against applying edits over a stale basis (e.g.
  the user re-ran postprocessing after loading the editor): a mismatch is
  a clear 409, never silent corruption.

Everything heavy is delegated to :class:`boneseg.pipeline.BonePipeline`;
no science lives in this file.
"""

from __future__ import annotations

import io
import json
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from boneseg import APP_NAME, __version__
from boneseg.config import (
    MAX_EDIT_DIM,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
    TRAINING_STAGING_DIR,
    AppConfig,
    DisplaySettings,
    ExportSettings,
    InferenceSettings,
    PostprocessSettings,
)
from boneseg.data import GeoRef, read_image
from boneseg.data.georef_fit import (
    gcps_sidecar_path,
    georef_from_gcps,
    write_world_file,
)
from boneseg.logging_setup import get_logger
from boneseg.models import MODEL_REGISTRY
from boneseg.pipeline import BonePipeline, PipelineResult
from boneseg.postprocessing import mask_to_rings, polylines_px_to_output

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

EXPORT_KEYS = ("mask", "overlay", "skeleton", "geojson", "dxf", "svg", "plate")


class Cancelled(Exception):
    """Raised inside a job thread when the user pressed Cancel."""


class Studio:
    """All server-side state for the single local user."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pipeline = BonePipeline(config)
        self.lock = threading.RLock()

        # Loaded image (pre-inference)
        self.source_path: Path | None = None
        self.image: np.ndarray | None = None
        self.georef: GeoRef | None = None

        # Last pipeline result
        self.result: PipelineResult | None = None

        # Edit-resolution artifacts
        self.edit_scale: float = 1.0
        self.edit_size: tuple[int, int] = (0, 0)        # (w, h)
        self.photo_jpeg: bytes | None = None            # edit-res photo
        self.mask_basis: np.ndarray | None = None       # bool, edit-res, as served
        self.mask_version: int = 0

        # Cumulative direct centerline edits, FULL resolution (bool).
        # Re-applied on every rebuild so pen strokes / skeleton erasures
        # survive further mask edits; reset by inference / postprocess.
        self.cl_add: np.ndarray | None = None
        self.cl_remove: np.ndarray | None = None

        # Cumulative user-ADDED mask pixels, FULL resolution (bool). Skeleton
        # filters (branch pruning / min fragment) are suppressed inside these
        # regions on rebuilds: hand-painted bones must never be filtered away
        # as noise, however small. Reset by inference / postprocess.
        self.mask_added: np.ndarray | None = None

        # Postprocess settings actually in force (set by inference and by the
        # explicit "Apply settings" button). Manual-edit applies reuse THESE —
        # the filters run once, not on every apply.
        self.pp_applied = PostprocessSettings()

        # Background job
        self.job_lock = threading.Lock()
        self.job_thread: threading.Thread | None = None
        self.cancel = threading.Event()
        self.job: dict = {"kind": None, "status": "idle", "progress": 0.0,
                          "message": "", "error": None, "rows": None}

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def job_running(self) -> bool:
        return self.job["status"] == "running"

    def set_image(self, path: Path) -> None:
        """Load an image from disk and precompute its edit-res JPEG."""
        img, georef = read_image(path)
        self.source_path = path
        self.image = img
        self.georef = georef
        # A GCP sidecar (written by /api/georef) re-applies user
        # georeferencing on every reopen of a plain photo.
        if self.georef is None:
            sidecar = gcps_sidecar_path(path)
            if sidecar.is_file():
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    g, _res, rms = georef_from_gcps(data["gcps"], data.get("epsg"))
                    self.georef = g
                    logger.info("GCP sidecar applied: %s (rms %.3f m)", sidecar.name, rms)
                except Exception:
                    logger.exception("Ignoring unreadable GCP sidecar %s", sidecar)
        self.result = None
        self.mask_basis = None
        self.cl_add = None
        self.cl_remove = None
        self.mask_added = None
        self.mask_version += 1

        h, w = img.shape[:2]
        self.edit_scale = min(1.0, MAX_EDIT_DIM / max(h, w))
        ew = max(1, round(w * self.edit_scale))
        eh = max(1, round(h * self.edit_scale))
        self.edit_size = (ew, eh)
        small = img if self.edit_scale >= 1.0 else np.asarray(
            Image.fromarray(img).resize((ew, eh), Image.BILINEAR)
        )
        buf = io.BytesIO()
        Image.fromarray(small).save(buf, "JPEG", quality=88)
        self.photo_jpeg = buf.getvalue()
        logger.info("Image loaded: %s (%dx%d, edit %dx%d, georef=%s)",
                    path.name, w, h, ew, eh, georef is not None)

    def downscale_binary(self, full01: np.ndarray) -> np.ndarray:
        """Full-res {0,1} -> edit-res bool. INTER_AREA + >0 keeps thin lines
        visible (any covered source pixel survives the downscale)."""
        ew, eh = self.edit_size
        if full01.shape == (eh, ew):
            return full01 > 0
        small = cv2.resize((full01 * 255).astype(np.uint8), (ew, eh),
                           interpolation=cv2.INTER_AREA)
        return small > 0

    def upscale_binary(self, small_bool: np.ndarray) -> np.ndarray:
        """Edit-res bool -> full-res bool (NEAREST blocks)."""
        assert self.image is not None
        h, w = self.image.shape[:2]
        if small_bool.shape == (h, w):
            return small_bool
        up = cv2.resize(small_bool.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)
        return up > 0

    def result_summary(self) -> dict:
        r = self.result
        ew, eh = self.edit_size
        base = {
            "has_image": self.image is not None,
            "has_result": r is not None,
            "mask_version": self.mask_version,
            "edit": {"width": ew, "height": eh, "scale": self.edit_scale},
        }
        if self.image is not None and self.source_path is not None:
            h, w = self.image.shape[:2]
            base["image"] = {
                "name": self.source_path.name,
                "width": w, "height": h,
                "megapixels": round(w * h / 1e6, 1),
                "georef": str(self.georef) if self.georef else None,
            }
        if r is not None:
            base["result"] = {
                "inference_seconds": round(r.inference_seconds, 1),
                "postprocess_seconds": round(r.postprocess_seconds, 1),
                "n_components": r.n_components,
                "n_centerlines": len(r.polylines_out),
                "fg_pixels": r.fg_pixels,
                "fg_fraction": round(r.fg_fraction, 5),
                "edited": r.stats.get("edited"),
            }
        return base


# --------------------------------------------------------------------------- #
# Request parsing helpers                                                      #
# --------------------------------------------------------------------------- #
def _pp_from(payload: dict) -> PostprocessSettings:
    d = PostprocessSettings()
    return PostprocessSettings(
        threshold=float(payload.get("threshold", d.threshold)),
        min_component_px=int(payload.get("min_component_px", d.min_component_px)),
        prune_branch_px=int(payload.get("prune_branch_px", d.prune_branch_px)),
        min_skeleton_px=int(payload.get("min_skeleton_px", d.min_skeleton_px)),
    )


def _export_from(payload: dict) -> ExportSettings:
    out_dir = str(payload.get("out_dir") or "").strip().strip('"')
    choices = payload.get("choices") or []
    return ExportSettings(
        output_dir=Path(out_dir) if out_dir else OUTPUTS_DIR,
        save_mask="mask" in choices,
        save_overlay="overlay" in choices,
        save_skeleton="skeleton" in choices,
        save_geojson="geojson" in choices,
        save_dxf="dxf" in choices,
        save_svg="svg" in choices,
        save_plate="plate" in choices,
        append_master=bool(payload.get("append_master", False)),
        plate_site=str(payload.get("plate_site", "")).strip(),
        plate_note=str(payload.get("plate_note", "")).strip(),
    )


def _png_rgba(mask_bool: np.ndarray, rgb: tuple[int, int, int] = (255, 255, 255),
              alpha: np.ndarray | None = None) -> bytes:
    h, w = mask_bool.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask_bool, 0] = rgb[0]
    arr[mask_bool, 1] = rgb[1]
    arr[mask_bool, 2] = rgb[2]
    arr[..., 3] = alpha if alpha is not None else mask_bool.astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _rasterize_polylines(polylines_px: list, shape: tuple[int, int],
                         thickness: int = 1) -> np.ndarray:
    """Draw full-res pixel polylines onto a {0,1} raster.

    thickness=1 keeps ``result.skeleton`` (skeleton PNG export, mask-unchanged
    apply path) consistent with hand-edited vectors; save_training dilates the
    1 px raster to reproduce the UNET_DATASET_FINAL4 mask stroke profile.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    for line in polylines_px:
        if len(line) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in line], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=1, thickness=thickness)
    return img


NO_CACHE = {"Cache-Control": "no-store"}


# --------------------------------------------------------------------------- #
# App factory                                                                  #
# --------------------------------------------------------------------------- #
def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    state = Studio(config)
    app = FastAPI(title=APP_NAME, version=__version__)
    app.state.studio = state

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    def _acquire():
        """Short-timeout lock so a long job answers 409 instead of hanging."""
        if not state.lock.acquire(timeout=1.0):
            raise HTTPException(409, "Server is busy (inference or batch is running).")

    # ------------------------------------------------------------------ #
    # Static frontend                                                     #
    # ------------------------------------------------------------------ #
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def _no_cache_static(request, call_next):  # noqa: ANN001
        """Force revalidation of the frontend files: a stale cached app.js
        combined with a newer backend is a recipe for silent breakage."""
        resp = await call_next(request)
        if request.url.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/")
    async def index():
        # Cache-bust the static assets by file mtime. index.html itself is
        # served no-store (always fresh), so a versioned ?v= URL guarantees the
        # browser fetches the app.js/app.css that MATCH this backend — a stale
        # cached app.js paired with a newer backend silently breaks the skeleton
        # view (it fetches endpoints the refactor removed).
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for asset in ("app.js", "app.css"):
            try:
                ver = int((STATIC_DIR / asset).stat().st_mtime)
            except OSError:
                continue
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={ver}")
        return Response(html, media_type="text/html", headers=NO_CACHE)

    # ------------------------------------------------------------------ #
    # Status / metadata                                                   #
    # ------------------------------------------------------------------ #
    @app.get("/api/status")
    async def status():
        d = PostprocessSettings()
        disp = DisplaySettings()
        return {
            "app": APP_NAME,
            "version": __version__,
            "device": state.pipeline.engine.device_info.label,
            "models": [
                {"key": k, "name": s.display_name, "description": s.description}
                for k, s in MODEL_REGISTRY.items()
            ],
            "model_key": config.model_key,
            "defaults": {
                "threshold": d.threshold,
                "min_component_px": d.min_component_px,
                "prune_branch_px": d.prune_branch_px,
                "min_skeleton_px": d.min_skeleton_px,
                "opacity": disp.overlay_opacity,
                "use_tta": config.inference.use_tta,
                "out_dir": str(OUTPUTS_DIR),
                "train_dir": str(TRAINING_STAGING_DIR),
            },
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            **state.result_summary(),
        }

    @app.get("/api/job")
    async def job_status():
        return state.job

    @app.post("/api/cancel")
    async def cancel_job():
        if state.job_running():
            state.cancel.set()
            return {"cancelling": True}
        return {"cancelling": False}

    # ------------------------------------------------------------------ #
    # Image loading                                                       #
    # ------------------------------------------------------------------ #
    @app.post("/api/open")
    async def open_image(file: UploadFile | None = File(None),
                         path: str | None = Form(None)):
        if state.job_running():
            raise HTTPException(409, "A job is running — cancel it or wait before loading a new image.")
        if file is None and not path:
            raise HTTPException(400, "Provide a file upload or a local path.")

        if path:
            src = Path(path.strip().strip('"'))
            if not src.exists():
                raise HTTPException(400, f"File not found: {src}")
        else:
            name = Path(file.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type: {name}. "
                                         f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            src = UPLOADS_DIR / name
            with open(src, "wb") as f:
                while chunk := await file.read(4 * 1024 * 1024):
                    f.write(chunk)

        _acquire()
        try:
            state.set_image(src)
        except Exception as exc:
            logger.exception("Failed to load image")
            raise HTTPException(400, f"Could not read image: {exc}")
        finally:
            state.lock.release()
        return state.result_summary()

    @app.get("/api/image/photo.jpg")
    async def photo():
        if state.photo_jpeg is None:
            raise HTTPException(404, "No image loaded.")
        return Response(state.photo_jpeg, media_type="image/jpeg", headers=NO_CACHE)

    @app.get("/api/image/mask.png")
    async def mask_png():
        """Current mask at edit resolution. Serving this SETS the diff basis:
        the next /api/apply_mask is interpreted relative to this exact image."""
        _acquire()
        try:
            if state.result is None:
                raise HTTPException(404, "No result yet — run inference first.")
            small = state.downscale_binary(state.result.mask)
            state.mask_basis = small
            data = _png_rgba(small)
            version = state.mask_version
        finally:
            state.lock.release()
        return Response(data, media_type="image/png",
                        headers={**NO_CACHE, "X-Mask-Version": str(version)})

    @app.get("/api/vectors")
    async def vectors():
        """Centerline polylines in EDIT-resolution pixel coordinates.

        These are the exact spline-smoothed polylines the exports use, so
        the browser view and the GeoJSON/DXF output are always identical.
        """
        _acquire()
        try:
            if state.result is None:
                raise HTTPException(404, "No result yet — run inference first.")
            s = state.edit_scale
            polylines = [
                [[round(x * s, 2), round(y * s, 2)] for x, y in line]
                for line in state.result.polylines_px
            ]
            ew, eh = state.edit_size
            return {"version": state.mask_version, "width": ew, "height": eh,
                    "polylines": polylines}
        finally:
            state.lock.release()

    @app.post("/api/set_vectors")
    async def set_vectors(payload: dict = Body(...)):
        """Replace the centerlines with hand-edited vector polylines.

        The client sends polylines in EDIT-resolution coordinates (the same
        space :func:`/api/vectors` serves). They become the authoritative
        geometry outright — no skeletonization, no spline re-fitting — so what
        the user draws is exactly what exports. Coordinates are mapped back to
        full-resolution pixels via ``edit_scale``; the skeleton raster is
        redrawn to match. These overrides live until the next inference /
        postprocess / mask-apply rebuilds from the mask.
        """
        _acquire()
        try:
            if state.result is None:
                raise HTTPException(400, "Run inference first.")
            lines_edit = payload.get("polylines") or []
            s = state.edit_scale or 1.0
            polylines_px = [
                [(float(x) / s, float(y) / s) for x, y in line]
                for line in lines_edit if len(line) >= 2
            ]
            r = state.result
            r.polylines_px = polylines_px
            r.polylines_out = polylines_px_to_output(polylines_px, r.georef)
            r.skeleton = _rasterize_polylines(polylines_px, r.mask.shape)
            r.stats["edited"] = "vector"
            state.mask_version += 1
            state.mask_basis = None
            logger.info("Vectors replaced by hand: %d centerlines", len(polylines_px))
            return state.result_summary()
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # Inference (background job)                                          #
    # ------------------------------------------------------------------ #
    @app.post("/api/infer")
    async def infer(payload: dict = Body(...)):
        if state.image is None:
            raise HTTPException(400, "Load an image first.")
        with state.job_lock:
            if state.job_running():
                raise HTTPException(409, "A job is already running.")
            state.cancel.clear()
            state.job = {"kind": "infer", "status": "running", "progress": 0.0,
                         "message": "Starting…", "error": None, "rows": None}

        model_key = str(payload.get("model_key", config.model_key))
        inf = InferenceSettings(use_tta=bool(payload.get("use_tta", False)))
        pp = _pp_from(payload)
        src = state.source_path

        def progress_cb(f: float) -> None:
            if state.cancel.is_set():
                raise Cancelled()
            state.job["progress"] = round(f * 0.92, 4)
            state.job["message"] = f"Inference… {f * 100:.0f}%"

        def work() -> None:
            try:
                with state.lock:
                    state.pipeline.engine.set_model(model_key)
                    state.job["message"] = "Loading model…"
                    img2, georef2, prob, dt = state.pipeline.run_inference(
                        src, inf, progress_cb=progress_cb)
                    state.job["progress"] = 0.94
                    state.job["message"] = "Postprocessing…"
                    result = state.pipeline.postprocess(
                        src, img2, georef2, prob, pp, inference_seconds=dt)
                    state.result = result
                    state.mask_version += 1
                    state.mask_basis = None
                    state.cl_add = None
                    state.cl_remove = None
                    state.mask_added = None
                    state.pp_applied = pp
                state.job.update(status="done", progress=1.0, message="Done")
            except Cancelled:
                state.job.update(status="cancelled", message="Cancelled")
            except Exception as exc:
                logger.exception("Inference job failed")
                state.job.update(status="error", error=f"{type(exc).__name__}: {exc}")

        state.job_thread = threading.Thread(target=work, daemon=True, name="infer-job")
        state.job_thread.start()
        return {"started": True}

    # ------------------------------------------------------------------ #
    # Fast postprocessing (sync)                                          #
    # ------------------------------------------------------------------ #
    @app.post("/api/postprocess")
    async def postprocess(payload: dict = Body(...)):
        _acquire()
        try:
            if state.result is None:
                raise HTTPException(400, "Run inference first.")
            r = state.result
            pp = _pp_from(payload)
            state.result = state.pipeline.postprocess(
                r.source_path, r.image, r.georef, r.prob, pp,
                inference_seconds=r.inference_seconds)
            state.mask_version += 1
            state.mask_basis = None
            state.cl_add = None
            state.cl_remove = None
            state.mask_added = None
            state.pp_applied = pp
            return state.result_summary()
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # Manual edits (mask diff vs served basis; centerline diff computed   #
    # client-side against the rendered vectors)                           #
    # ------------------------------------------------------------------ #
    @app.post("/api/apply_mask")
    async def apply_mask(file: UploadFile = File(...),
                         skel_add: UploadFile | None = File(None),
                         skel_rem: UploadFile | None = File(None),
                         mask_version: int = Form(...),
                         prune_branch_px: int = Form(20),
                         min_skeleton_px: int = Form(40)):
        data = await file.read()
        skel_add_data = await skel_add.read() if skel_add is not None else None
        skel_rem_data = await skel_rem.read() if skel_rem is not None else None
        _acquire()
        try:
            if state.result is None or state.mask_basis is None:
                raise HTTPException(400, "No editable mask on the server — run inference first.")
            if int(mask_version) != state.mask_version:
                raise HTTPException(409, "The mask changed on the server since the editor "
                                         "loaded it — reload the mask and redo the edit.")
            edited = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
            if edited.shape[:2] != state.mask_basis.shape:
                raise HTTPException(400, "Edited mask has the wrong resolution.")
            edited_bool = edited[..., 3] > 10

            add = edited_bool & ~state.mask_basis
            rem = ~edited_bool & state.mask_basis
            mask_changed = bool(add.any() or rem.any())
            new_mask = state.result.mask.copy()
            if add.any():
                add_full = state.upscale_binary(add)
                new_mask[add_full] = 1
                # Remember user-added regions so skeleton filters never touch
                # them on this or any later rebuild.
                if state.mask_added is None:
                    state.mask_added = np.zeros(new_mask.shape, dtype=bool)
                state.mask_added |= add_full
            if rem.any():
                rem_full = state.upscale_binary(rem)
                new_mask[rem_full] = 0
                if state.mask_added is not None:
                    state.mask_added &= ~rem_full

            # Direct centerline edits (line pen / skeleton-view eraser):
            # the CLIENT diffs its rendered centerlines against what it drew
            # and sends explicit add/remove masks at edit resolution.
            # Accumulated so they survive subsequent mask edits/re-applies.
            def _read_edit_mask(blob: bytes) -> np.ndarray:
                arr = np.asarray(Image.open(io.BytesIO(blob)).convert("RGBA"))
                if arr.shape[:2] != state.mask_basis.shape:
                    raise HTTPException(400, "Centerline edit mask has the wrong resolution.")
                return arr[..., 3] > 10

            sk_add = _read_edit_mask(skel_add_data) if skel_add_data else None
            sk_rem = _read_edit_mask(skel_rem_data) if skel_rem_data else None
            if (sk_add is not None and sk_add.any()) or (sk_rem is not None and sk_rem.any()):
                logger.info("centerline diff: add=%d px, remove=%d px",
                            int(sk_add.sum()) if sk_add is not None else 0,
                            int(sk_rem.sum()) if sk_rem is not None else 0)
                h, w = state.result.mask.shape
                if sk_rem is not None and sk_rem.any():
                    # The displayed splines deviate from the raw medial axis
                    # by a couple of px, and skeletonization itself is not
                    # perfectly deterministic between runs — dilate erasures
                    # so they reliably catch the underlying centerline.
                    sk_rem = cv2.dilate(
                        sk_rem.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    ) > 0
                add_full = state.upscale_binary(sk_add) if sk_add is not None else \
                    np.zeros((h, w), dtype=bool)
                rem_full = state.upscale_binary(sk_rem) if sk_rem is not None else \
                    np.zeros((h, w), dtype=bool)
                if state.cl_add is None:
                    state.cl_add = np.zeros((h, w), dtype=bool)
                if state.cl_remove is None:
                    state.cl_remove = np.zeros((h, w), dtype=bool)
                state.cl_add = (state.cl_add | add_full) & ~rem_full
                state.cl_remove = (state.cl_remove | rem_full) & ~add_full

            # Filters run ONCE (at inference / "Apply settings") — applies
            # reuse those exact values, whatever the sliders say now, and the
            # user's own additions are protected from them entirely.
            pp = PostprocessSettings(
                threshold=0.5, min_component_px=0,
                prune_branch_px=state.pp_applied.prune_branch_px,
                min_skeleton_px=state.pp_applied.min_skeleton_px)
            # Unchanged mask -> keep the exact skeleton the user edited
            # (re-deriving would shift pixels and break their erasures).
            base_skel = None if mask_changed else state.result.skeleton
            state.result = state.pipeline.apply_manual_mask(
                state.result, new_mask, pp,
                cl_add=state.cl_add, cl_remove=state.cl_remove,
                base_skeleton=base_skel, protect=state.mask_added)
            state.mask_version += 1
            state.mask_basis = None
            return state.result_summary()
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # Export                                                              #
    # ------------------------------------------------------------------ #
    @app.post("/api/export")
    async def export(payload: dict = Body(...)):
        _acquire()
        try:
            if state.result is None:
                raise HTTPException(400, "Nothing to export yet — run inference first.")
            exp = _export_from(payload)
            if not any([exp.save_mask, exp.save_overlay, exp.save_skeleton,
                        exp.save_geojson, exp.save_dxf, exp.save_svg,
                        exp.append_master]):
                raise HTTPException(400, "Select at least one export format.")
            display = DisplaySettings(overlay_opacity=float(payload.get("opacity", 0.55)))
            target = Path(exp.output_dir) / state.result.source_path.stem
            try:
                written = state.pipeline.export(state.result, exp, display, out_dir=target)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            return {"files": [str(p) for p in written], "target": str(target),
                    "master": str(Path(exp.output_dir) / "master.dxf") if exp.append_master else None}
        finally:
            state.lock.release()

    @app.post("/api/save_training")
    async def save_training(payload: dict = Body(...)):
        """Save the current photo + corrected mask as a training pair.

        Writes into a STAGING folder (not the live dataset) using the exact
        UNET_DATASET_FINAL4 layout: images/<stem>.png (RGB), masks/<stem>.png
        (L, {0,255}), overlays/<stem>.png, and an update-or-append row in
        dataset_manifest.csv — so a later merge is a plain folder copy.
        """
        import csv

        from boneseg.data import render_overlay as _render_overlay

        _acquire()
        try:
            if state.result is None:
                raise HTTPException(400, "Nothing to save yet — run inference first.")
            r = state.result
            out_dir = Path(str(payload.get("out_dir") or "").strip().strip('"')
                           or TRAINING_STAGING_DIR)
            stem = r.source_path.stem
            for sub in ("images", "masks", "overlays"):
                (out_dir / sub).mkdir(parents=True, exist_ok=True)

            img_path = out_dir / "images" / f"{stem}.png"
            mask_path = out_dir / "masks" / f"{stem}.png"
            ov_path = out_dir / "overlays" / f"{stem}.png"
            Image.fromarray(r.image).save(img_path)
            # FINAL4 mask convention: centerline VECTORS rasterized as 1 px
            # polylines dilated with a 5x5 ellipse ({0,255}, mode L) — this
            # reproduces the measured stroke-width profile of the dataset
            # (median 5.7, p25 4.5, p75 6.0 via distance transform) exactly,
            # unlike the raw model mask whose width wobbles. Includes all
            # manual edits, since the vectors are the authoritative geometry.
            thin = _rasterize_polylines(r.polylines_px, r.mask.shape)
            mask_canon = cv2.dilate(
                thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            Image.fromarray(mask_canon * 255, "L").save(mask_path)
            overlay = _render_overlay(r.image, mask_canon,
                                      float(payload.get("opacity", 0.55)),
                                      config.display.overlay_color)
            Image.fromarray(overlay).save(ov_path)

            # manifest: update-or-append by image_name (re-saving replaces)
            manifest = out_dir / "dataset_manifest.csv"
            header = ["image_name", "mask_name", "source_dataset", "width", "height"]
            rows: list[dict] = []
            if manifest.exists():
                with open(manifest, newline="", encoding="utf-8") as f:
                    rows = [row for row in csv.DictReader(f)
                            if row.get("image_name") != f"{stem}.png"]
            rows.append({"image_name": f"{stem}.png", "mask_name": f"{stem}.png",
                         "source_dataset": "boneseg_studio",
                         "width": str(r.width), "height": str(r.height)})
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

            logger.info("Training pair staged: %s -> %s (edited=%s)",
                        stem, out_dir, r.stats.get("edited"))
            return {"files": [str(img_path), str(mask_path), str(ov_path), str(manifest)],
                    "target": str(out_dir), "stem": stem, "n_pairs": len(rows),
                    "edited": r.stats.get("edited")}
        finally:
            state.lock.release()

    @app.post("/api/reveal")
    async def reveal(payload: dict = Body(...)):
        """Open a folder in Windows Explorer (local desktop app convenience)."""
        import os
        p = Path(str(payload.get("path", "")).strip().strip('"'))
        if not p.exists():
            raise HTTPException(400, f"Path does not exist: {p}")
        os.startfile(str(p if p.is_dir() else p.parent))  # noqa: S606
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # GCP georeferencing                                                  #
    # ------------------------------------------------------------------ #
    @app.post("/api/georef")
    async def georef_apply(payload: dict = Body(...)):
        """Fit an affine from user GCPs (edit-res px + surveyed E/N), apply
        it to the current image/result, persist a sidecar for reopens and
        write a world file so the ORIGINAL photo loads georeferenced in
        QGIS. ``clear: true`` removes user georeferencing again."""
        if state.image is None:
            raise HTTPException(400, "Load an image first.")
        _acquire()
        try:
            def _rewire(g: GeoRef | None) -> None:
                if state.result is None:
                    return
                state.result.georef = g
                state.result.polylines_out = polylines_px_to_output(
                    state.result.polylines_px, g)
                state.result.rings_out = mask_to_rings(state.result.mask, g)

            sidecar = gcps_sidecar_path(state.source_path)
            if payload.get("clear"):
                state.georef = None
                _rewire(None)
                if sidecar.is_file():
                    sidecar.unlink()
                return {**state.result_summary(), "cleared": True}

            swap = bool(payload.get("swap", False))
            epsg_raw = payload.get("epsg")
            try:
                epsg = int(str(epsg_raw).strip()) if str(epsg_raw or "").strip() else None
            except ValueError:
                raise HTTPException(400, f"EPSG must be a number, got: {epsg_raw}")
            scale = state.edit_scale
            gcps = []
            for g in payload.get("gcps") or []:
                try:
                    e, n = float(g["e"]), float(g["n"])
                except (KeyError, TypeError, ValueError):
                    raise HTTPException(400, "Every point needs numeric E and N.")
                if swap:
                    e, n = n, e
                gcps.append({"px": float(g["px"]) / scale,
                             "py": float(g["py"]) / scale, "e": e, "n": n})
            try:
                georef2, residuals, rms = georef_from_gcps(gcps, epsg)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(400, str(exc))

            state.georef = georef2
            _rewire(georef2)
            sidecar.write_text(json.dumps(
                {"version": 1, "epsg": epsg, "gcps": gcps}, indent=1),
                encoding="utf-8")
            world: list[str] = []
            try:
                world = [p.name for p in write_world_file(state.source_path, georef2)]
            except Exception:
                logger.exception("World file write failed (georef still applied)")
            logger.info("Georef applied: %d GCPs, rms %.3f m, epsg=%s",
                        len(gcps), rms, epsg)
            return {**state.result_summary(),
                    "residuals_m": [round(r, 3) for r in residuals],
                    "rms_m": round(rms, 3),
                    "world_files": world}
        finally:
            state.lock.release()

    # ------------------------------------------------------------------ #
    # Batch (background job)                                              #
    # ------------------------------------------------------------------ #
    @app.post("/api/batch_open")
    async def batch_open(payload: dict = Body(...)):
        """Open one batch item in the editor, seeding the result from the
        mask the batch already exported — no re-inference needed."""
        if state.job_running():
            raise HTTPException(409, "A job is running — wait for it to finish first.")
        src = Path(str(payload.get("path", "")).strip().strip('"'))
        if not src.is_file():
            raise HTTPException(400, f"Image not found: {src}")
        item_dir = Path(str(payload.get("out_dir", "")).strip().strip('"'))
        pp = _pp_from(payload)

        # save_mask_raster writes .tif for georeferenced inputs, .png otherwise.
        mask_path = next((p for p in (item_dir / f"{src.stem}_mask.tif",
                                      item_dir / f"{src.stem}_mask.png")
                          if p.is_file()), None)

        _acquire()
        try:
            state.set_image(src)
            if mask_path is not None:
                mask01 = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
                if mask01.shape != state.image.shape[:2]:
                    raise HTTPException(400,
                        f"Exported mask {mask_path.name} does not match the image size "
                        f"({mask01.shape[1]}x{mask01.shape[0]} vs {state.image.shape[1]}x{state.image.shape[0]}).")
                # The binary mask doubles as the prob map: any threshold keeps
                # it intact, and skeleton/vector derivation matches the batch
                # (same mask, same settings).
                state.result = state.pipeline.postprocess(
                    src, state.image, state.georef, mask01.astype(np.float32), pp)
                state.result.stats["edited"] = None
                state.pp_applied = pp
                state.mask_version += 1
                logger.info("Batch item reopened with exported mask: %s", mask_path)
            else:
                logger.info("Batch item reopened WITHOUT mask (none exported): %s", src.name)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to reopen batch item")
            raise HTTPException(400, f"Could not reopen batch item: {exc}")
        finally:
            state.lock.release()
        return {**state.result_summary(), "mask_loaded": mask_path is not None}

    @app.post("/api/batch_upload")
    async def batch_upload(files: list[UploadFile] = File(...)):
        """Stage drag-&-dropped images into a fresh folder under uploads/,
        so the regular folder-based /api/batch can run on them unchanged."""
        if state.job_running():
            raise HTTPException(409, "A job is running — wait for it to finish before staging a batch.")
        stage = UPLOADS_DIR / "batch" / datetime.now().strftime("%Y%m%d_%H%M%S")
        saved: list[str] = []
        skipped: list[str] = []
        for uf in files:
            name = Path(uf.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped.append(name)
                continue
            stage.mkdir(parents=True, exist_ok=True)
            dst = stage / name
            n = 1
            # A folder drop can contain the same basename in different
            # subfolders — keep both rather than silently overwriting.
            while dst.exists():
                dst = stage / f"{Path(name).stem}_{n}{Path(name).suffix}"
                n += 1
            with open(dst, "wb") as f:
                while chunk := await uf.read(4 * 1024 * 1024):
                    f.write(chunk)
            saved.append(dst.name)
        if not saved:
            raise HTTPException(400, "No supported images in the drop. Supported: "
                                     + ", ".join(SUPPORTED_EXTENSIONS))
        logger.info("Staged %d dropped image(s) in %s (%d skipped)",
                    len(saved), stage, len(skipped))
        return {"dir": str(stage), "count": len(saved), "skipped": skipped}

    @app.post("/api/batch")
    async def batch(payload: dict = Body(...)):
        in_dir = Path(str(payload.get("in_dir", "")).strip().strip('"'))
        if not in_dir.is_dir():
            raise HTTPException(400, f"Input folder does not exist: {in_dir}")
        out_dir = str(payload.get("out_dir") or "").strip().strip('"')
        out_path = Path(out_dir) if out_dir else OUTPUTS_DIR / "batch"

        with state.job_lock:
            if state.job_running():
                raise HTTPException(409, "A job is already running.")
            state.cancel.clear()
            state.job = {"kind": "batch", "status": "running", "progress": 0.0,
                         "message": "Starting batch…", "error": None, "rows": None}

        model_key = str(payload.get("model_key", config.model_key))
        inf = InferenceSettings(use_tta=bool(payload.get("use_tta", False)))
        pp = _pp_from(payload)
        exp = _export_from({**payload, "out_dir": str(out_path)})
        display = DisplaySettings(overlay_opacity=float(payload.get("opacity", 0.55)))

        def file_cb(i: int, n: int, name: str) -> None:
            if state.cancel.is_set():
                raise Cancelled()
            state.job["progress"] = round(i / max(1, n), 4)
            state.job["message"] = f"[{i + 1}/{n}] {name}"

        def work() -> None:
            try:
                with state.lock:
                    state.pipeline.engine.set_model(model_key)
                    rows = state.pipeline.process_folder(
                        in_dir, out_path, inf, pp, exp, display, file_cb=file_cb)
                state.job.update(status="done", progress=1.0, rows=rows,
                                 message=f"Batch finished — outputs in {out_path}")
            except Cancelled:
                state.job.update(status="cancelled", message="Batch cancelled")
            except Exception as exc:
                logger.exception("Batch job failed")
                state.job.update(status="error", error=f"{type(exc).__name__}: {exc}")

        state.job_thread = threading.Thread(target=work, daemon=True, name="batch-job")
        state.job_thread.start()
        return {"started": True}

    logger.info("FastAPI app created (device: %s)", state.pipeline.engine.device_info.label)
    return app
