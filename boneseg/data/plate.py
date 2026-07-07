"""Catalog plate PDF: one A4 page per grave, publication-style.

Layout (A4 portrait): header with site name + grave label, the photo with
the segmentation drawn on top, and a title block with a scale bar, north
arrow and metadata. When the image is georeferenced the plate is printed
at an EXACT standard cartographic scale (1:10, 1:20, ...): the image panel
is sized so that panel_mm * scale = ground metres, and the scale bar is
derived from the same number — what you measure on paper with a ruler is
what the bone measures in the ground.

Non-georeferenced images still get a plate (photo + vectors + metadata),
just without the scale bar / north arrow, and are labelled "scale unknown".

matplotlib is optional like ezdxf: without it the export is skipped with a
warning, never an error.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from boneseg.data.readers import GeoRef
from boneseg.data.writers import render_overlay
from boneseg.logging_setup import get_logger

try:
    import matplotlib

    matplotlib.use("Agg")  # headless; never touch a GUI backend
    from matplotlib import pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MPL = False

logger = get_logger(__name__)

Polyline = list[tuple[float, float]]

# A4 portrait in millimetres.
PAGE_W, PAGE_H = 210.0, 297.0
MARGIN = 15.0
HEADER_H = 14.0          # site + label line
FOOTER_H = 30.0          # scale bar / north arrow / metadata block
GAP = 5.0
PANEL_W = PAGE_W - 2 * MARGIN
PANEL_H = PAGE_H - 2 * MARGIN - HEADER_H - FOOTER_H - 2 * GAP

# Standard catalog scales, ascending.
STANDARD_SCALES = (5, 10, 20, 25, 50, 100, 200, 250, 500)
# Nice scale-bar lengths in metres.
BAR_LENGTHS_M = (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0)

MAX_EMBED_DIM = 2400     # cap embedded photo resolution to keep PDFs small


def _pixel_size_m(georef: GeoRef) -> tuple[float, float]:
    """Ground size of one pixel (width, height) from the affine transform."""
    t = georef.transform
    return math.hypot(t.a, t.d), math.hypot(t.b, t.e)


def _north_angle_page_deg(georef: GeoRef, width: int, height: int) -> float:
    """Direction of world north on the page, in degrees CCW from page-up.

    0 = north points straight up on the plate (typical north-up raster).
    """
    inv = ~georef.transform
    cx, cy = georef.transform * (width / 2.0, height / 2.0)
    c0 = inv * (cx, cy)
    c1 = inv * (cx, cy + 1.0)          # one unit toward world north
    dcol, drow = c1[0] - c0[0], c1[1] - c0[1]
    # Page coords: x right, y up; image rows increase downward.
    return math.degrees(math.atan2(dcol, -drow)) * -1.0


def _pick_scale(extent_w_m: float, extent_h_m: float) -> int:
    """Smallest standard scale at which the image fits the panel."""
    for s in STANDARD_SCALES:
        if (extent_w_m * 1000.0 / s) <= PANEL_W and (extent_h_m * 1000.0 / s) <= PANEL_H:
            return s
    return STANDARD_SCALES[-1]


def _pick_bar_m(panel_w_mm: float, scale: int) -> float:
    """Longest nice bar length that stays within ~40% of the panel width."""
    best = BAR_LENGTHS_M[0]
    for b in BAR_LENGTHS_M:
        if b * 1000.0 / scale <= panel_w_mm * 0.4:
            best = b
    return best


def _draw_scale_bar(ax, x_mm: float, y_mm: float, bar_m: float, scale: int) -> None:
    """Classic alternating black/white bar with end labels, in page mm."""
    bar_mm = bar_m * 1000.0 / scale
    n_seg = 4
    seg = bar_mm / n_seg
    for i in range(n_seg):
        ax.add_patch(plt.Rectangle(
            (x_mm + i * seg, y_mm), seg, 2.0,
            facecolor=("black" if i % 2 == 0 else "white"),
            edgecolor="black", linewidth=0.5))
    label = f"{bar_m:g} m" if bar_m >= 1 else f"{bar_m * 100:g} cm"
    ax.text(x_mm, y_mm + 3.2, "0", fontsize=7, ha="center")
    ax.text(x_mm + bar_mm, y_mm + 3.2, label, fontsize=7, ha="center")
    ax.text(x_mm + bar_mm / 2, y_mm - 4.0, f"M 1:{scale}", fontsize=9,
            ha="center", fontweight="bold")


def _draw_north_arrow(ax, x_mm: float, y_mm: float, angle_deg: float) -> None:
    """Simple survey-style north arrow rotated by ``angle_deg`` (CCW)."""
    a = math.radians(angle_deg)
    length = 10.0
    dx, dy = -math.sin(a) * length, math.cos(a) * length
    ax.annotate(
        "", xy=(x_mm + dx, y_mm + dy), xytext=(x_mm, y_mm),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="black",
                        mutation_scale=14))
    ax.text(x_mm + dx * 1.35, y_mm + dy * 1.35, "N", fontsize=10,
            ha="center", va="center", fontweight="bold")


def render_plate_figure(
    image_rgb: np.ndarray,
    mask01: np.ndarray,
    polylines_px: Sequence[Polyline],
    georef: GeoRef | None,
    label: str,
    site: str = "",
    note: str = "",
    model_name: str = "",
    overlay_opacity: float = 0.35,
):
    """Compose one plate as a matplotlib Figure (A4). Returns None if
    matplotlib is unavailable. Caller owns the figure (savefig + close);
    the batch catalog reuses this to append pages into one PdfPages."""
    if not HAS_MPL:
        logger.warning("matplotlib not installed — plate export skipped")
        return None

    h, w = image_rgb.shape[:2]

    # ---- panel geometry ------------------------------------------------ #
    scale: int | None = None
    if georef is not None:
        px_w, px_h = _pixel_size_m(georef)
        ext_w_m, ext_h_m = w * px_w, h * px_h
        scale = _pick_scale(ext_w_m, ext_h_m)
        panel_w = ext_w_m * 1000.0 / scale
        panel_h = ext_h_m * 1000.0 / scale
    else:
        # Fit the panel, aspect preserved — no true scale available.
        k = min(PANEL_W / w, PANEL_H / h)
        panel_w, panel_h = w * k, h * k

    # ---- compose the page (axes in page millimetres) ------------------- #
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    page = fig.add_axes([0, 0, 1, 1])
    page.set_xlim(0, PAGE_W)
    page.set_ylim(0, PAGE_H)
    page.axis("off")

    top = PAGE_H - MARGIN
    page.text(MARGIN, top - 4, site or "", fontsize=12, fontweight="bold")
    page.text(PAGE_W - MARGIN, top - 4, label, fontsize=14,
              fontweight="bold", ha="right")
    page.plot([MARGIN, PAGE_W - MARGIN], [top - HEADER_H] * 2,
              color="black", linewidth=0.8)

    # Image panel, centred in the area between header and footer.
    px0 = MARGIN + (PANEL_W - panel_w) / 2.0
    avail_top = top - HEADER_H - GAP
    avail_bot = MARGIN + FOOTER_H + GAP
    py0 = avail_bot + (avail_top - avail_bot - panel_h) / 2.0
    ax_img = fig.add_axes([px0 / PAGE_W, py0 / PAGE_H,
                           panel_w / PAGE_W, panel_h / PAGE_H])
    ax_img.axis("off")

    small = image_rgb
    if max(h, w) > MAX_EMBED_DIM:
        from PIL import Image as PILImage
        k = MAX_EMBED_DIM / max(h, w)
        small = np.asarray(PILImage.fromarray(image_rgb).resize(
            (max(1, round(w * k)), max(1, round(h * k))), PILImage.BILINEAR))
        small_mask = np.asarray(PILImage.fromarray(
            (mask01 * 255).astype(np.uint8)).resize(
            (small.shape[1], small.shape[0]), PILImage.NEAREST)) > 127
    else:
        small_mask = mask01 > 0

    shown = render_overlay(small, small_mask.astype(np.uint8),
                           overlay_opacity) if overlay_opacity > 0 else small
    # extent in FULL-RES pixel coords so vectors overlay exactly.
    ax_img.imshow(shown, extent=(0, w, h, 0), interpolation="bilinear")
    for line in polylines_px:
        if len(line) >= 2:
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            ax_img.plot(xs, ys, color="#b40000", linewidth=0.7,
                        solid_capstyle="round")
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.axis("on")

    # ---- footer / title block ------------------------------------------ #
    fy = MARGIN + FOOTER_H
    page.plot([MARGIN, PAGE_W - MARGIN], [fy] * 2, color="black", linewidth=0.8)
    if georef is not None and scale is not None:
        bar_m = _pick_bar_m(panel_w, scale)
        _draw_scale_bar(page, MARGIN + 2, MARGIN + 12, bar_m, scale)
        _draw_north_arrow(page, PAGE_W - MARGIN - 12, MARGIN + 8,
                          _north_angle_page_deg(georef, w, h))
        crs_txt = str(georef.crs) if georef.crs else "local grid"
        page.text(MARGIN + 2, MARGIN + 2.5, f"CRS: {crs_txt}", fontsize=7,
                  color="0.35")
    else:
        page.text(MARGIN + 2, MARGIN + 12,
                  "not georeferenced — scale unknown", fontsize=9,
                  style="italic", color="0.35")

    mid_x = PAGE_W / 2.0
    info = [date.today().isoformat()]
    if model_name:
        info.append(f"segmentation: {model_name}")
    page.text(mid_x, MARGIN + 12, "\n".join(info), fontsize=7.5,
              ha="center", va="top", color="0.25")
    if note:
        page.text(mid_x, MARGIN + 22, note, fontsize=8.5, ha="center",
                  va="top", wrap=True)

    fig._boneseg_scale = scale  # for the writer's log line
    return fig


def save_plate_pdf(
    image_rgb: np.ndarray,
    mask01: np.ndarray,
    polylines_px: Sequence[Polyline],
    georef: GeoRef | None,
    out_path: Path,
    label: str,
    site: str = "",
    note: str = "",
    model_name: str = "",
    overlay_opacity: float = 0.35,
) -> Path | None:
    """Render one catalog plate to ``out_path`` (PDF). Returns None if
    matplotlib is unavailable."""
    fig = render_plate_figure(image_rgb, mask01, polylines_px, georef,
                              label, site, note, model_name, overlay_opacity)
    if fig is None:
        return None
    fig.savefig(out_path, format="pdf", dpi=200)
    scale = fig._boneseg_scale
    plt.close(fig)
    logger.info("Plate written: %s (scale %s)", out_path,
                f"1:{scale}" if scale else "none")
    return out_path
