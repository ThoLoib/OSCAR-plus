#!/usr/bin/env python3
"""Step 7 — Geometric check: dGeDi re-ranking of the fusion shortlist.

History: until 2026-09-18 this module was called ``step_b2_geometry_reranking.py``
("sub-step B2") and contained an in-process implementation against the old
GeDi service (gedi:5060), which docker-compose.yml no longer defines. The
REPORTED geometry results never went through it: Stage 1 (E2/O1c/O1e)
registers via ``_pair_scores_dgedi`` in experiment1, Stage 3 and Stage 4
call ``evaluation/dgedi_bridge.dgedi_rerank`` directly — all against the
dGeDi HTTP service (port 5061). Since the rework this module is pipeline
step 7 (it replaces the scale estimation, which no reported run used) and
is a client of that same dGeDi service:

  - ``geo_rerank(...)`` — the shortlist re-ordering rule, moved here
    VERBATIM from the Stage-3 driver; the driver
    (``experiments/stage3_bop.py``) now imports it from here
    (rankings unchanged).
  - ``GeometryReRanker.rerank(...)`` — dGeDi ``/rerank`` + ``geo_rerank``
    for the interactive pipeline (run_pipeline, step 7).
  - ``GeometryReRanker._load_cad_pointcloud`` — UNCHANGED: Stage 1 builds
    its CAD clouds on it (UnitSphereReRanker in experiment1), and the
    descriptor caches key off their fingerprint.

The old in-process path (GeDi descriptors, in-process RANSAC/ICP, signals
``fitness``/``chamfer_*``) was removed; ``rerank()`` with its argument
``all_aligned=`` aborts with a clear message. Historical implementation:
``git show bd2de45d:pipeline/step_b2_geometry_reranking.py``.
"""

import logging
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .step6_fusion import FusedCandidate

logger = logging.getLogger(__name__)

# Rangkriterium der Umsortierung (Stage-3-Vokabular): Env STAGE3_GEO_SIGNAL,
# Default "distance" — gesetzt von experiments/stage3_bop.py.
GEO_SIGNAL = os.environ.get("STAGE3_GEO_SIGNAL", "distance")   # distance | borda | fitness


def geo_rerank(fused_ranking, geo, top_k, signal: str = None):
    """Re-rank the fused top-K by the dGeDi geometry signal.

    ``signal`` (default ``STAGE3_GEO_SIGNAL`` = **distance**):
      * ``distance`` — rank by the trimmed surface distance after alignment.
        This is the Stage-1 C1 winner (0.6405 vs 0.6362 Borda vs 0.6251 fitness)
        and therefore the **cross-stage-consistent** criterion.
      * ``borda``    — mean-rank of fitness and distance (the pre-2026-08-27
        behaviour; kept so the earlier runs remain reproducible).
      * ``fitness``  — RANSAC inlier fraction only.
    Failed/uncached candidates sort to the back of the shortlist; the tail past
    top_k is untouched."""
    signal = signal or GEO_SIGNAL
    head = fused_ranking[:top_k]
    tail = fused_ranking[top_k:]
    ids = [oid for oid, _ in head]
    NEG = float("-inf")

    def _sig(o, key, sign):
        g = geo.get(o)
        if not g or not g.get("ok"):
            return NEG
        return sign * float(g[key])

    fit = [_sig(o, "ransac_fitness", 1.0) for o in ids]
    dst = [_sig(o, "d_ransac", -1.0) for o in ids]

    def _ranks(vals):
        return np.argsort(np.argsort(-np.asarray(vals), kind="stable"),
                          kind="stable").astype(float)

    if signal == "distance":
        key = _ranks(dst)
    elif signal == "fitness":
        key = _ranks(fit)
    elif signal == "borda":
        key = (_ranks(fit) + _ranks(dst)) / 2.0
    else:
        raise ValueError(f"unknown STAGE3_GEO_SIGNAL {signal!r}")
    order = list(np.argsort(key, kind="stable"))
    head_re = [(ids[i], -float(key[i])) for i in order]
    return head_re + tail


# ---------------------------------------------------------------------------
# Datenstrukturen (Felder unveraendert — Alt-Aufrufer lesen diese Attribute)
# ---------------------------------------------------------------------------

@dataclass
class GeometryCandidate:
    """Candidate after the geometric check (step 7).

    ``chamfer_score``/``d_ransac`` = trimmed one-sided surface distance
    after RANSAC(+ICP) alignment (mm in gallery scale; lower = better);
    ``ransac_fitness`` = inlier fraction. The remaining fields come from the
    fusion or are kept for legacy compatibility (the removed in-process path
    also populated ``d_icp``/``icp_*``)."""
    object_id: str
    fused_score: float = 0.0
    gedi_score: float = 0.0
    chamfer_score: float = float("inf")
    d_ransac: float = float("inf")
    d_icp: float = float("inf")
    geometry_score: float = 0.0
    ransac_transformation: Optional[np.ndarray] = None
    ransac_fitness: float = 0.0
    icp_transformation: Optional[np.ndarray] = None
    icp_fitness: float = 0.0
    icp_inlier_rmse: float = 0.0
    transformation: Optional[np.ndarray] = None
    registration_failed: bool = False
    cad_model_path: str = ""
    best_view_path: str = ""
    clip_score: float = 0.0
    dino_score: float = 0.0
    ulip_score: float = 0.0


@dataclass
class GeometryReRankingResult:
    """Result of step 7 (order = new ranking, best first)."""
    candidates: List[GeometryCandidate]
    signal: str
    best_candidate: Optional[GeometryCandidate] = None
    best_transformation: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Schritt-7-Modul
# ---------------------------------------------------------------------------

class GeometryReRanker:
    """Geometric check of the fusion shortlist via the dGeDi service.

    Usage (run_pipeline, step 7):
        >>> reranker = GeometryReRanker(config)
        >>> result = reranker.rerank(fused_candidates, observed_pcd)

    Prerequisite: a running dGeDi service with a matching gallery
    (``docker compose up -d dgedi``; gallery via ``DGEDI_CACHE_DIR``).
    The candidate IDs must be keys of the gallery manifest, and the query
    cloud must be in the units of the gallery (BOP gallery
    .dgedi_gallery: METRES — as backproject_masked in query_cloud.py returns).
    """

    # Konfig-Signale des Altbestands -> Stage-3-Rangkriterium
    _LEGACY_TO_STAGE3 = {
        "distance": "distance", "borda": "borda", "fitness": "fitness",
        "gedi": "fitness", "chamfer": "distance", "chamfer_unaligned": "distance",
        "chamfer_ransac": "distance", "chamfer_icp": "distance", "both": "borda",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config

    def rerank(
        self,
        fused_candidates: List[FusedCandidate],
        observed_pcd,
        signal: Optional[str] = None,
        query_id: Optional[str] = None,
        **legacy,
    ) -> GeometryReRankingResult:
        """dGeDi registration of the top-K + re-ordering via ``geo_rerank``.

        Parameters as in the reported Stage-3 runs: 6000 keypoints /
        10000 RANSAC iterations / +ICP.
        ``query_id`` is only accepted for logging purposes now.
        """
        if legacy:
            raise RuntimeError(
                "GeometryReRanker: the in-process GeDi path "
                f"({', '.join(sorted(legacy))}=...) was removed on 2026-09-18 — "
                "this module is now a client of the dGeDi service "
                "(docker compose up -d dgedi). Historical implementation: "
                "git show bd2de45d:pipeline/step_b2_geometry_reranking.py")
        sig = self._LEGACY_TO_STAGE3.get(
            signal or self.config.geometry_reranking_signal, "distance")
        top_k = int(self.config.geometry_reranking_top_k)
        if not fused_candidates:
            return GeometryReRankingResult(candidates=[], signal=sig)

        # dgedi_bridge liegt in evaluation/ (kein Paket) — Pfad ergaenzen.
        _eval = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "evaluation")
        if _eval not in sys.path:
            sys.path.insert(0, _eval)
        from dgedi_bridge import dgedi_rerank

        pts = np.asarray(observed_pcd.points, dtype=np.float32)
        ids = [c.object_id for c in fused_candidates[:top_k]]
        geo = dgedi_rerank(
            pts, ids,
            ransac_keypoints=getattr(self.config, "dgedi_ransac_keypoints", 6000),
            ransac_max_iter=getattr(self.config, "dgedi_ransac_max_iter", 10000),
            use_icp=True)
        if geo is None:
            logger.warning("dGeDi service unreachable — step 7 leaves "
                           "the fusion ranking unchanged "
                           "(docker compose up -d dgedi).")

        by_id = {c.object_id: c for c in fused_candidates}
        ranking: List[Tuple[str, float]] = [
            (c.object_id, float(c.fused_score)) for c in fused_candidates]
        n_ok = sum(1 for v in (geo or {}).values() if v.get("ok"))
        if n_ok:
            ranking = geo_rerank(ranking, geo, top_k, signal=sig)

        out: List[GeometryCandidate] = []
        for oid, score in ranking:
            fc = by_id[oid]
            g = (geo or {}).get(oid) or {}
            d = g.get("d_ransac")
            out.append(GeometryCandidate(
                object_id=oid,
                fused_score=float(fc.fused_score),
                geometry_score=float(score),
                ransac_fitness=float(g.get("ransac_fitness", 0.0)),
                chamfer_score=float(d) if d is not None else float("inf"),
                d_ransac=float(d) if d is not None else float("inf"),
                registration_failed=bool(g) and not bool(g.get("ok")),
                cad_model_path=getattr(fc, "cad_model_path", ""),
                best_view_path=getattr(fc, "best_view_path", ""),
                clip_score=getattr(fc, "clip_score", 0.0),
                dino_score=getattr(fc, "dino_score", 0.0),
                ulip_score=getattr(fc, "ulip_score", 0.0)))
        best = out[0] if n_ok else None
        logger.info("Step 7: %d/%d registrations ok (signal %s)%s",
                    n_ok, len(ids), sig,
                    f" — new rank 1: {best.object_id}" if best else "")
        return GeometryReRankingResult(candidates=out, signal=sig,
                                       best_candidate=best,
                                       best_transformation=None)

    @staticmethod
    def _load_cad_pointcloud(cad_path: str, n_points: int = 10000):
        """Load a CAD model and sample a point cloud."""
        import open3d as o3d

        if not cad_path or not os.path.isfile(cad_path):
            # Try common mesh extensions
            for ext in (".obj", ".ply", ".glb", ".stl"):
                alt = cad_path + ext if cad_path else ""
                if os.path.isfile(alt):
                    cad_path = alt
                    break
            else:
                return None

        try:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            if mesh.is_empty():
                return None
            mesh.compute_vertex_normals()
            # Deterministic sampling: sample_points_uniformly() draws from
            # Open3D's GLOBAL RNG and takes no seed argument (0.19), so
            # without this the same CAD yields a different cloud on every
            # call — irreproducible geometry scores, and a descriptor cache
            # that can never hit.  Seed from the CAD path so the cloud is a
            # pure function of the model, not of call order.
            # Masked into the non-negative int32 range — Open3D's seed() is
            # bound to a 32-bit signed int and rejects larger values.
            o3d.utility.random.seed(
                int(hashlib.sha1(os.path.basename(cad_path).encode()
                                 ).hexdigest()[:8], 16) % (2 ** 31 - 1))
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            return pcd
        except Exception as exc:
            logger.warning("CAD load failed (%s): %s", cad_path, exc)
            return None
