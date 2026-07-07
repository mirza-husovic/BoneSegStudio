"""Skeletonization with branch pruning.

Pipeline (extends predict.py, which used the raw medial axis):

  1. ``medial_axis`` of the cleaned binary mask (as in predict.py),
  2. build a topology graph with ``sknw`` (nodes = junctions/endpoints,
     edges = pixel chains between them),
  3. iteratively remove short *terminal* branches (spurs) — these are the
     medial-axis artifacts caused by small bumps on the outline,
  4. remove isolated fragments whose total length is below a minimum,

The pruned graph is the single source of truth: both the skeleton image
shown in the UI and the vector polylines are derived from it, so what you
see is exactly what gets exported.

The graph is built with ``multi=True`` (``nx.MultiGraph``): bone outlines
are closed loops, and a loop with exactly two junctions produces TWO pixel
chains between the SAME node pair. A plain ``nx.Graph`` silently keeps only
one of them, which used to drop whole outline sides from the vector output
(verified: a synthetic 2-junction ring lost 87 of 194 skeleton px).
"""

from __future__ import annotations

from typing import Iterator

import networkx as nx
import numpy as np
from skimage.morphology import medial_axis

from boneseg.logging_setup import get_logger

try:
    import sknw

    HAS_SKNW = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SKNW = False

logger = get_logger(__name__)


def iter_edges(graph: nx.Graph) -> Iterator[tuple]:
    """Yield ``(s, e, key, data)`` for Graph and MultiGraph alike.

    ``key`` is None for a plain Graph. Centralizing this lets the rest of
    the code stay agnostic about the graph flavor.
    """
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for s, e, d in graph.edges(data=True):
            yield s, e, None, d


def _remove_edge(graph: nx.Graph, s, e, key) -> None:
    if graph.is_multigraph():
        graph.remove_edge(s, e, key=key)
    else:
        graph.remove_edge(s, e)


def build_skeleton_graph(mask01: np.ndarray) -> nx.Graph | None:
    """Medial axis of the mask, converted to an sknw topology graph.

    Returns None when the mask is empty or sknw is unavailable (the caller
    then degrades gracefully to "no skeleton").
    """
    if not HAS_SKNW:
        logger.warning("sknw not installed — skeletonization disabled")
        return None
    if mask01.sum() == 0:
        return nx.MultiGraph()

    skel = medial_axis(mask01 > 0).astype(np.uint8)
    return sknw.build_sknw(skel, multi=True)


def prune_graph(
    graph: nx.Graph,
    prune_branch_px: int,
    min_component_px: int,
) -> nx.Graph:
    """Remove short terminal branches, then short isolated fragments.

    Parameters
    ----------
    prune_branch_px
        Terminal edges (one endpoint of degree 1) shorter than this are
        removed. Pruning iterates because removing a spur can expose a new
        one at the same junction.
    min_component_px
        Connected components whose total edge length is below this are
        dropped entirely (isolated speckle centerlines).

    Operates on a copy; the input graph is not modified.
    """
    g = graph.copy()

    # -- 1. iterative spur removal ------------------------------------------
    changed = True
    while changed:
        changed = False
        for s, e, k, d in list(iter_edges(g)):
            if len(d["pts"]) >= prune_branch_px:
                continue
            # terminal spur: at least one endpoint is a dead end, and the
            # edge is not the only thing connecting two dead ends of a
            # longer structure (that case is handled by the component filter)
            if g.degree(s) == 1 or g.degree(e) == 1:
                _remove_edge(g, s, e, k)
                changed = True
        # drop nodes that lost all their edges
        g.remove_nodes_from([n for n, deg in g.degree() if deg == 0])

    # -- 2. drop short isolated fragments ------------------------------------
    for comp in list(nx.connected_components(g)):
        sub = g.subgraph(comp)
        total = sum(len(d["pts"]) for _, _, _, d in iter_edges(sub))
        if total < min_component_px:
            g.remove_nodes_from(comp)

    return g


def apply_centerline_edits(
    skel01: np.ndarray,
    add01: np.ndarray | None,
    remove01: np.ndarray | None,
) -> tuple[np.ndarray, nx.Graph | None]:
    """Merge the user's direct centerline edits into a derived skeleton.

    ``add01`` marks pixels the user drew with the line pen (they become
    centerline outright, bypassing the mask -> medial-axis route), and
    ``remove01`` marks regions erased in the skeleton view. The merged
    image is re-thinned (pen strokes are wider than 1 px and stroke/skeleton
    overlaps double up) and a fresh topology graph is built from it WITHOUT
    pruning — the derived part was already pruned, and explicit user strokes
    must never be filtered away as noise.

    Returns ``(skeleton01, graph)``; graph is None when sknw is missing.
    """
    from skimage.morphology import skeletonize

    merged = skel01 > 0
    if remove01 is not None:
        merged &= ~(remove01 > 0)
    if add01 is not None:
        merged |= add01 > 0
    if not merged.any():
        return np.zeros_like(skel01, dtype=np.uint8), nx.MultiGraph()

    thin = skeletonize(merged).astype(np.uint8)
    if not HAS_SKNW:
        return thin, None
    return thin, sknw.build_sknw(thin, multi=True)


def graph_to_image(graph: nx.Graph | None, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize the (pruned) skeleton graph back to a binary uint8 image."""
    img = np.zeros(shape, dtype=np.uint8)
    if graph is None:
        return img
    for _, _, _, d in iter_edges(graph):
        pts = d["pts"]
        img[pts[:, 0], pts[:, 1]] = 1
    for n in graph.nodes():
        pts = graph.nodes[n]["pts"]
        img[pts[:, 0], pts[:, 1]] = 1
    return img
