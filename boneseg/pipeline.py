"""End-to-end processing pipeline.

The pipeline is split into two stages with different costs:

  * :meth:`BonePipeline.run_inference`  — expensive (network forward passes);
    produces a float probability map that is cached by the UI.
  * :meth:`BonePipeline.postprocess`    — cheap (thresholding, skeleton,
    vectors); can be re-applied to a cached probability map when the user
    moves the threshold / pruning sliders, without re-running the model.

:meth:`BonePipeline.export` writes artifacts, and
:meth:`BonePipeline.process_folder` implements batch mode on top of the same
two stages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from boneseg.config import (
    SUPPORTED_EXTENSIONS,
    AppConfig,
    DisplaySettings,
    ExportSettings,
    InferenceSettings,
    PostprocessSettings,
)
from boneseg.data import (
    GeoRef,
    append_to_master,
    read_image,
    render_overlay,
    save_dxf,
    save_geojson,
    save_mask_raster,
    save_overlay_png,
    save_photo_geotiff,
    save_plate_pdf,
    save_skeleton_png,
    save_svg,
)
from boneseg.inference import InferenceEngine
from boneseg.logging_setup import get_logger
from boneseg.postprocessing import (
    apply_centerline_edits,
    build_skeleton_graph,
    count_components,
    graph_to_image,
    graph_to_polylines_px,
    mask_to_rings,
    polylines_px_to_output,
    prune_graph,
    remove_component_at,
)
from boneseg.postprocessing.vectorize import Polyline, rings_to_geojson_dict

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Everything produced for one image; the UI keeps this in session state."""

    source_path: Path
    image: np.ndarray                   # RGB uint8
    georef: GeoRef | None
    prob: np.ndarray                    # float32 probability map (cached)
    mask: np.ndarray                    # uint8 {0,1} after cleanup
    skeleton: np.ndarray                # uint8 {0,1} pruned skeleton
    polylines_px: list[Polyline]        # centerlines, pixel coords (SVG space)
    polylines_out: list[Polyline]       # centerlines, GeoJSON coords
    rings_out: list[Polyline]           # outline polygons, GeoJSON coords
    n_components: int
    fg_pixels: int
    inference_seconds: float
    postprocess_seconds: float
    stats: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def fg_fraction(self) -> float:
        return self.fg_pixels / max(1, self.height * self.width)


class BonePipeline:
    """Stateful facade over the inference engine + postprocessing stack."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.engine = InferenceEngine(self.config.model_key)

    # ------------------------------------------------------------------ #
    # Stage 1: inference                                                  #
    # ------------------------------------------------------------------ #
    def run_inference(
        self,
        path: Path,
        settings: InferenceSettings,
        progress_cb: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, GeoRef | None, np.ndarray, float]:
        """Load one image and predict its probability map.

        Returns (image_rgb, georef, prob, seconds).
        """
        img, georef = read_image(Path(path))
        logger.info(
            "Inference: %s (%dx%d, georef=%s, tta=%s)",
            Path(path).name, img.shape[1], img.shape[0],
            georef is not None, settings.use_tta,
        )
        t0 = time.time()
        prob = self.engine.predict(img, settings, progress_cb=progress_cb)
        dt = time.time() - t0
        logger.info("Inference done in %.1fs", dt)
        return img, georef, prob, dt

    # ------------------------------------------------------------------ #
    # Stage 2: postprocessing                                             #
    # ------------------------------------------------------------------ #
    def postprocess(
        self,
        source_path: Path,
        image: np.ndarray,
        georef: GeoRef | None,
        prob: np.ndarray,
        settings: PostprocessSettings,
        inference_seconds: float = 0.0,
    ) -> PipelineResult:
        """Mask cleanup + skeleton + vectors from a (possibly cached) prob map."""
        from boneseg.postprocessing.cleanup import threshold_and_clean

        t0 = time.time()
        mask = threshold_and_clean(prob, settings.threshold, settings.min_component_px)
        return self._assemble(
            source_path, image, georef, prob, mask, settings, inference_seconds, time.time() - t0
        )

    # ------------------------------------------------------------------ #
    # Manual editing (paint corrections / click-to-delete)                #
    # ------------------------------------------------------------------ #
    def apply_manual_mask(
        self,
        result: PipelineResult,
        edited_mask01: np.ndarray,
        settings: PostprocessSettings,
        cl_add: np.ndarray | None = None,
        cl_remove: np.ndarray | None = None,
        base_skeleton: np.ndarray | None = None,
        protect: np.ndarray | None = None,
    ) -> PipelineResult:
        """Rebuild skeleton/vectors from hand-made corrections.

        ``edited_mask01`` replaces the model's mask outright (no thresholding
        or small-object removal — the user's strokes are the ground truth).
        ``cl_add``/``cl_remove`` are direct centerline edits (line pen /
        skeleton-view eraser) merged into the derived skeleton AFTER pruning,
        so explicit strokes are never filtered as noise.

        ``base_skeleton`` — pass the CURRENT result's skeleton when the mask
        did not change: skeletonization is not perfectly deterministic (the
        medial axis / graph pruning can shift pixels by ±1 between runs on
        the same mask), so re-deriving would make centerline erasures miss
        the shifted pixels. Using the exact skeleton the user saw keeps
        their edits pixel-accurate.

        ``protect`` — full-res bool of user-ADDED mask regions; skeleton
        pixels inside are exempt from branch pruning / min-fragment filters
        (hand-painted bones survive however small).
        """
        t0 = time.time()
        mask = (edited_mask01 > 0).astype(np.uint8)
        new_result = self._assemble(
            result.source_path, result.image, result.georef, result.prob,
            mask, settings, result.inference_seconds, time.time() - t0,
            cl_add=cl_add, cl_remove=cl_remove, base_skeleton=base_skeleton,
            protect=protect,
        )
        new_result.stats["edited"] = "paint"
        return new_result

    def delete_component(
        self,
        result: PipelineResult,
        row: int,
        col: int,
        settings: PostprocessSettings,
    ) -> tuple[PipelineResult, bool]:
        """Remove one connected mask component under (row, col) — click-to-delete.

        Returns ``(result, False)`` unchanged if the click missed any
        foreground pixel (nothing to delete).
        """
        new_mask, removed = remove_component_at(result.mask, row, col)
        if not removed:
            return result, False
        t0 = time.time()
        new_result = self._assemble(
            result.source_path, result.image, result.georef, result.prob,
            new_mask, settings, result.inference_seconds, time.time() - t0,
        )
        new_result.stats["edited"] = "delete"
        return new_result, True

    def _assemble(
        self,
        source_path: Path,
        image: np.ndarray,
        georef: GeoRef | None,
        prob: np.ndarray,
        mask: np.ndarray,
        settings: PostprocessSettings,
        inference_seconds: float,
        postprocess_seconds: float,
        cl_add: np.ndarray | None = None,
        cl_remove: np.ndarray | None = None,
        base_skeleton: np.ndarray | None = None,
        protect: np.ndarray | None = None,
    ) -> PipelineResult:
        """Skeleton + vectors from a FINAL binary mask — shared by postprocess()
        and the manual-edit paths so exported artifacts always match what's
        displayed, however the mask was produced. Optional ``cl_add`` /
        ``cl_remove`` layers (direct centerline edits) are merged into the
        pruned skeleton before vectorization. When ``base_skeleton`` is given
        it is used instead of re-deriving from the mask (see
        :meth:`apply_manual_mask` for why). ``protect`` marks user-added mask
        regions: medial-axis pixels inside are restored after pruning, so the
        filters never eat hand-painted content."""
        has_cl = (cl_add is not None and cl_add.any()) or (
            cl_remove is not None and cl_remove.any())

        if base_skeleton is not None and has_cl:
            skeleton, graph = apply_centerline_edits(base_skeleton, cl_add, cl_remove)
        else:
            graph = build_skeleton_graph(mask)
            unpruned = graph_to_image(graph, mask.shape)  # full medial axis
            if graph is not None:
                graph = prune_graph(
                    graph,
                    prune_branch_px=settings.prune_branch_px,
                    min_component_px=settings.min_skeleton_px,
                )
            skeleton = graph_to_image(graph, mask.shape)
            # Bring back what pruning removed inside user-added mask regions
            # (fed through the same merge path as explicit centerline edits).
            add_eff = cl_add
            if protect is not None and protect.any():
                extra = (unpruned > 0) & protect & ~(skeleton > 0)
                if cl_remove is not None:
                    extra &= ~(cl_remove > 0)
                if extra.any():
                    add_eff = extra if add_eff is None else ((add_eff > 0) | extra)
            if (add_eff is not None and add_eff.any()) or has_cl:
                skeleton, graph = apply_centerline_edits(skeleton, add_eff, cl_remove)

        polylines_px = graph_to_polylines_px(
            graph,
            downsample_k=settings.spline_downsample,
            spline_smooth=settings.spline_smooth,
            density=settings.spline_density,
        )
        polylines_out = polylines_px_to_output(polylines_px, georef)
        rings_out = mask_to_rings(mask, georef)

        result = PipelineResult(
            source_path=Path(source_path),
            image=image,
            georef=georef,
            prob=prob,
            mask=mask,
            skeleton=skeleton,
            polylines_px=polylines_px,
            polylines_out=polylines_out,
            rings_out=rings_out,
            n_components=count_components(mask),
            fg_pixels=int(mask.sum()),
            inference_seconds=inference_seconds,
            postprocess_seconds=postprocess_seconds,
        )
        logger.info(
            "Assembled result in %.1fs: %d components, %d centerlines, fg %.3f%%",
            postprocess_seconds, result.n_components, len(polylines_out),
            result.fg_fraction * 100,
        )
        return result

    def process(
        self,
        path: Path,
        inference: InferenceSettings,
        postprocess: PostprocessSettings,
        progress_cb: Callable[[float], None] | None = None,
    ) -> PipelineResult:
        """Convenience: both stages in one call (used by batch mode)."""
        img, georef, prob, dt = self.run_inference(path, inference, progress_cb)
        return self.postprocess(path, img, georef, prob, postprocess, inference_seconds=dt)

    # ------------------------------------------------------------------ #
    # Export                                                              #
    # ------------------------------------------------------------------ #
    def export(
        self,
        result: PipelineResult,
        export: ExportSettings,
        display: DisplaySettings,
        out_dir: Path | None = None,
    ) -> list[Path]:
        """Write the selected artifacts for one result; returns written paths.

        When ``export.append_master`` is set, the vectors are additionally
        merged into the site-wide master files in ``export.output_dir``
        (one GeoJSON pair + one DXF for ALL exported graves).
        """
        out_dir = Path(out_dir) if out_dir is not None else Path(export.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = result.source_path.stem
        written: list[Path] = []

        if export.save_mask:
            written.append(save_mask_raster(result.mask, out_dir / f"{stem}_mask", result.georef))
        if export.save_overlay:
            overlay = render_overlay(
                result.image, result.mask, display.overlay_opacity, display.overlay_color
            )
            written.append(save_overlay_png(overlay, out_dir / f"{stem}_overlay.png"))
        if export.save_skeleton:
            written.append(save_skeleton_png(result.skeleton, out_dir / f"{stem}_skeleton.png"))
        if export.save_geojson:
            written.append(
                save_geojson(result.polylines_out, out_dir / f"{stem}_centerlines.geojson", result.georef)
            )
            polygons_path = out_dir / f"{stem}_polygons.geojson"
            with open(polygons_path, "w", encoding="utf-8") as f:
                json.dump(rings_to_geojson_dict(result.rings_out, result.georef), f)
            written.append(polygons_path)
        if export.save_dxf:
            # The photo rides along as an IMAGE underlay (copied next to
            # the DXF) so any CAD opens it already under the vectors.
            dxf_path = save_dxf(result.rings_out, result.polylines_out,
                                out_dir / f"{stem}.dxf", result.georef,
                                image_path=result.source_path,
                                image_size=(result.width, result.height))
            if dxf_path is not None:
                written.append(dxf_path)
        if export.save_svg:
            written.append(
                save_svg(result.polylines_px, out_dir / f"{stem}_centerlines.svg",
                         width=result.width, height=result.height)
            )
        if export.save_geotiff:
            # Writes only when a georeference exists (GCPs or geo input);
            # otherwise logs + skips — a pixel-space GeoTIFF is meaningless.
            gt_path = save_photo_geotiff(
                result.image, out_dir / f"{stem}_photo.tif", result.georef)
            if gt_path is not None:
                written.append(gt_path)
        if export.save_plate:
            plate_path = save_plate_pdf(
                result.image, result.mask, result.polylines_px, result.georef,
                out_dir / f"{stem}_plate.pdf",
                label=stem, site=export.plate_site, note=export.plate_note,
                model_name=self.engine.spec.key,
                overlay_opacity=min(display.overlay_opacity, 0.4),
            )
            if plate_path is not None:
                written.append(plate_path)
        if export.append_master:
            written.extend(
                append_to_master(
                    Path(export.output_dir), stem,
                    result.rings_out, result.polylines_out, result.georef,
                    image_width=result.width, image_height=result.height,
                )
            )

        logger.info("Exported %d files for %s -> %s", len(written), stem, out_dir)
        return written

    # ------------------------------------------------------------------ #
    # Batch mode                                                          #
    # ------------------------------------------------------------------ #
    def process_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        inference: InferenceSettings,
        postprocess: PostprocessSettings,
        export: ExportSettings,
        display: DisplaySettings,
        file_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[dict]:
        """Process every supported image in ``input_dir``.

        Per-image failures are logged and recorded, never fatal — one
        corrupted file must not abort a night-long batch run.

        Returns a list of per-file summary dicts (also written as
        ``batch_summary.json`` in the output folder).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("Batch: %d files in %s -> %s", len(files), input_dir, output_dir)

        # One combined catalog (grave per page) accumulated alongside the
        # per-image plates; only when the plate export is selected.
        catalog = None
        catalog_pages = 0
        if export.save_plate:
            from boneseg.data.plate import HAS_MPL, render_plate_figure
            if HAS_MPL:
                from matplotlib import pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages
                output_dir.mkdir(parents=True, exist_ok=True)
                catalog = PdfPages(output_dir / "catalog.pdf")

        rows: list[dict] = []
        try:
            for i, path in enumerate(files):
                if file_cb is not None:
                    file_cb(i, len(files), path.name)
                try:
                    result = self.process(path, inference, postprocess)
                    self.export(result, export, display, out_dir=output_dir / path.stem)
                    if catalog is not None:
                        fig = render_plate_figure(
                            result.image, result.mask, result.polylines_px,
                            result.georef, label=path.stem,
                            site=export.plate_site, note=export.plate_note,
                            model_name=self.engine.spec.key,
                            overlay_opacity=min(display.overlay_opacity, 0.4))
                        if fig is not None:
                            catalog.savefig(fig)
                            plt.close(fig)
                            catalog_pages += 1
                    rows.append({
                        "file": path.name,
                        "path": str(path),
                        "out_dir": str(output_dir / path.stem),
                        "status": "ok",
                        "width": result.width,
                        "height": result.height,
                        "georeferenced": result.georef is not None,
                        "components": result.n_components,
                        "centerlines": len(result.polylines_out),
                        "fg_fraction": round(result.fg_fraction, 5),
                        "seconds": round(result.inference_seconds + result.postprocess_seconds, 1),
                    })
                except Exception as exc:
                    logger.exception("Batch item failed: %s", path.name)
                    rows.append({
                        "file": path.name,
                        "path": str(path),
                        "out_dir": str(output_dir / path.stem),
                        "status": f"failed: {exc}",
                        "width": 0, "height": 0, "georeferenced": False,
                        "components": 0, "centerlines": 0,
                        "fg_fraction": 0.0, "seconds": 0.0,
                    })
        finally:
            # Close the combined catalog even on cancel; drop it when empty.
            if catalog is not None:
                catalog.close()
                if catalog_pages:
                    logger.info("Catalog written: %s (%d pages)",
                                output_dir / "catalog.pdf", catalog_pages)
                else:
                    (output_dir / "catalog.pdf").unlink(missing_ok=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.engine.spec.key,
                "device": self.engine.device_info.label,
                "threshold": postprocess.threshold,
                "tta": inference.use_tta,
                "n_files": len(rows),
                "files": rows,
            }, f, indent=2)
        logger.info("Batch done: %d files, summary at %s", len(rows), summary_path)
        return rows
