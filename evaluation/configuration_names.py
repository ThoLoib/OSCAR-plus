"""
configuration_names.py
======================
Internal ablation keys  <->  release configuration names.

Two naming systems exist in this repository and they must never be confused:

* **internal names** (``E1c_full_fusion``, ``3b_cross``, ``oscarplus_v2_tau037_
  dinomean_partialforce``) are the *ablation keys of the drivers*.  They are the
  identifiers the experiment scripts select a run by (``AblationSpec.name``, the
  ``--mode``/``--query``/``--gallery`` combination of Stage 3, the mode folder of
  Stage 2) and they encode the history of the grid, not its meaning.
* **release names** (``fusion``, ``pose_proxy_cross_partial``,
  ``fused_partial``) *name the configuration*.  They are the identifiers used in
  ``results/`` and in every output a reader sees: file names, folder names and
  the identifier columns of the CSVs (``configuration``,
  ``configuration_a``/``configuration_b``).

The numbers behind both names are the same; only the label changes.  A freshly
produced run therefore writes its artefacts under the **release** name, in the
same shape as the corresponding file in ``results/``, and the reading scripts
accept either name on their CLI.

API
---
``to_release(name)``  internal -> release;  unknown names pass through UNCHANGED
``to_internal(name)`` release  -> canonical internal;  unknown pass through
``release_names()`` / ``internal_names()``  the known vocabulary
Every function takes an optional ``stage`` (``"stage1"`` | ``"stage2"`` |
``"stage3"``) to restrict the lookup to one stage's table.

Pass-through instead of raising is deliberate: these helpers sit in the middle of
path construction in the drivers, so an unmapped identifier (a smoke arm, a
sweep point ``W_0.3_0.4_0.3``, a legacy run folder such as
``fusion_views16_shape_topk8``) must keep flowing untouched rather than break a
run or, worse, silently retarget it.

Collisions and the canonical internal name
------------------------------------------
The mapping is many-to-one: in Stage 3 both ``3a_cross`` and ``3a_cross_v2``
denote ``retrieval_cross_partial``, and both ``3a_pc`` and ``3a_pc_v2`` denote
``retrieval_pc_partial`` (the ``_v2`` re-runs are what ``results/`` holds; the
plain keys are the superseded first pass of the same configuration).
``to_internal`` therefore has to choose, and it returns the ``_v2`` key whenever
one exists, the plain key otherwise.  Consequence:

    to_release(to_internal(r)) == r                 for every release name r
    to_internal(to_release(i)) == i                 for every canonical i
    to_internal(to_release("3a_cross")) == "3a_cross_v2"   (non-canonical input)

Names deliberately NOT in the tables
------------------------------------
``STAGE1_DUPLICATES`` lists two internal Stage-1 keys that are *the same
configuration* as another arm (``--views 42`` is the default, so the V42 cells
duplicate the plain ones).  They have no release name of their own and no file
in ``results/``; they pass through ``to_release`` unchanged so that a reader
looking for them finds nothing instead of silently reading the neighbouring
arm's numbers.  Likewise the two Stage-1 geometry variants
``E2_chamfer_icp`` / ``E2_chamfer_unaligned`` are not part of the release at all
and have no entry here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stage 1 — SHREC'18 ablation arms (experiments/stage1_shrec18.py)
# ---------------------------------------------------------------------------
STAGE1: Dict[str, str] = {
    "E1a_text_only": "text_only",
    "E1_view_only": "appearance_only",
    "A2_view_only_V8": "appearance_only_views8",
    "A2_view_only_V16": "appearance_only_views16",
    "A2_view_only_V32": "appearance_only_views32",
    "E4_siglip_only": "appearance_only_siglip",
    "E1_shape_only": "shape_only",
    "A7_shape_only_V8": "shape_only_views8",
    "A7_shape_only_V16": "shape_only_views16",
    "A7_shape_only_V32": "shape_only_views32",
    "E7_uni3d_shape_only": "shape_only_uni3d",
    "O5_xyz_shape_only": "shape_only_xyz_checkpoint",
    "E7_ulip2_cross_shape_only": "shape_only_cross",
    "E2b_fullmesh_shape_only": "shape_only_fullmesh",
    "E7_ulip2_cross_fullmesh_shape_only": "shape_only_cross_fullmesh",
    "E1b_text_view": "text_appearance",
    "E1c_full_fusion": "fusion",
    "A7f_full_fusion_shape_V42": "fusion_shape_prefix_views42",
    "O4_V8": "fusion_views8",
    "O4_V16": "fusion_views16",
    "O4_V32": "fusion_views32",
    "E4_siglip": "fusion_siglip",
    "E7_uni3d": "fusion_uni3d",
    "O5_xyz_only": "fusion_xyz_checkpoint",
    "E7_ulip2_cross": "fusion_cross",
    "E2b_fullmesh": "fusion_fullmesh",
    "E7_ulip2_cross_fullmesh": "fusion_cross_fullmesh",
    "E6_rrf": "fusion_rrf",
    "E1d_clip_pruned": "fusion_clip_pruned",
    "E1_oscar_cascade": "oscar_cascade",
    "O2_clip_threshold": "fusion_clip_threshold",
    "O2_clip_threshold_cal": "fusion_clip_threshold_calibrated",
    "O2_visual_first": "fusion_visual_first",
    "E2_fitness": "fusion_geo_fitness",
    "E2_chamfer_ransac": "fusion_geo_trimmed_distance",
    "E2_both": "fusion_geo_mean_rank",
    "O1e_gedi_with_base": "fusion_geo_mean_rank_with_base",
    "O1c_gedi_post_fusion": "text_appearance_geo_post_fusion",
    "E2b_fullmesh_geo": "fusion_fullmesh_geo_trimmed_distance",
}

# Internal Stage-1 keys that duplicate another arm (V42 IS the default view
# count).  No release name, no file in results/ — see the module docstring.
STAGE1_DUPLICATES: Dict[str, str] = {
    "A2_view_only_V42": "E1_view_only",
    "A7_shape_only_V42": "E1_shape_only",
}

# ---------------------------------------------------------------------------
# Stage 2 — MI3DOR (experiments/stage2_mi3dor.py)
# ---------------------------------------------------------------------------
STAGE2: Dict[str, str] = {
    "oscarplus_v2_tau037_dinomean_partialforce": "fused_partial",
    "oscarplus_v2_tau037_dinomean_ulipfix": "fused_fullmesh",
    "oscar_legacy_v8": "legacy_oscar_views8",
}

# ---------------------------------------------------------------------------
# Stage 3 — BOP retrieval / pose (experiments/stage3_bop.py)
# ---------------------------------------------------------------------------
STAGE3: Dict[str, str] = {
    "3a_cross": "retrieval_cross_partial",
    "3a_cross_v2": "retrieval_cross_partial",
    "3a_cross_fullmesh_v2": "retrieval_cross_fullmesh",
    "3a_pc": "retrieval_pc_partial",
    "3a_pc_v2": "retrieval_pc_partial",
    "3a_pc_fullmesh_v2": "retrieval_pc_fullmesh",
    "3a_oscar": "retrieval_oscar_baseline",
    "3a_cross_geo_distance": "retrieval_cross_geo_trimmed_distance",
    "3a_cross_geo_fitness": "retrieval_cross_geo_fitness",
    "3a_pc_geo_distance": "retrieval_pc_geo_trimmed_distance",
    "3a_pc_geo_fitness": "retrieval_pc_geo_fitness",
    "gt": "pose_gt",
    "3b_cross": "pose_proxy_cross_partial",
    "3b_cross_fullmesh": "pose_proxy_cross_fullmesh",
    "3b_cross_geo": "pose_proxy_cross_geo",
    "3b_oscar": "pose_proxy_oscar",
    "3c_cross": "pose_decomposition_cross",
    "3c_cross_fullmesh": "pose_decomposition_cross_fullmesh",
}

BY_STAGE: Dict[str, Dict[str, str]] = {
    "stage1": STAGE1,
    "stage2": STAGE2,
    "stage3": STAGE3,
}
STAGES: Tuple[str, ...] = tuple(BY_STAGE)


def _canonical(new: str, old: str) -> bool:
    """Should ``new`` replace ``old`` as the canonical internal name?

    Stage 3 re-ran two configurations under a ``_v2`` key and ``results/`` holds
    those re-runs, so the ``_v2`` key is the one a reader should be pointed at.
    """
    return new.endswith("_v2") and not old.endswith("_v2")


def _reverse(mapping: Dict[str, str]) -> Dict[str, str]:
    rev: Dict[str, str] = {}
    for internal, release in mapping.items():
        if release not in rev or _canonical(internal, rev[release]):
            rev[release] = internal
    return rev


REVERSE_BY_STAGE: Dict[str, Dict[str, str]] = {
    stage: _reverse(table) for stage, table in BY_STAGE.items()
}
REVERSE_STAGE1: Dict[str, str] = REVERSE_BY_STAGE["stage1"]
REVERSE_STAGE2: Dict[str, str] = REVERSE_BY_STAGE["stage2"]
REVERSE_STAGE3: Dict[str, str] = REVERSE_BY_STAGE["stage3"]


def _tables(stage: Optional[str], reverse: bool) -> List[Dict[str, str]]:
    src = REVERSE_BY_STAGE if reverse else BY_STAGE
    if stage is None:
        return [src[s] for s in STAGES]
    if stage not in src:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return [src[stage]]


def to_release(name: str, stage: Optional[str] = None) -> str:
    """Internal ablation key -> release configuration name.

    Unknown names (including names that already ARE release names) are returned
    unchanged, so this is safe to wrap around any identifier and is idempotent.
    """
    for table in _tables(stage, reverse=False):
        if name in table:
            return table[name]
    return name


def to_internal(name: str, stage: Optional[str] = None) -> str:
    """Release configuration name -> canonical internal ablation key.

    Unknown names (including names that already ARE internal keys) are returned
    unchanged.  Where several internal keys share one release name the canonical
    one is returned — see "Collisions" in the module docstring.
    """
    for table in _tables(stage, reverse=True):
        if name in table:
            return table[name]
    return name


def release_names(stage: Optional[str] = None) -> List[str]:
    """Every known release name (de-duplicated, stage order then table order)."""
    out: List[str] = []
    for table in _tables(stage, reverse=False):
        for release in table.values():
            if release not in out:
                out.append(release)
    return out


def internal_names(stage: Optional[str] = None) -> List[str]:
    """Every known internal ablation key (stage order then table order)."""
    out: List[str] = []
    for table in _tables(stage, reverse=False):
        out.extend(table)
    return out


def stage_of(name: str) -> Optional[str]:
    """Stage a name belongs to (either direction), or ``None`` if unknown."""
    for stage in STAGES:
        if name in BY_STAGE[stage] or name in REVERSE_BY_STAGE[stage]:
            return stage
    return None
