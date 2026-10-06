#!/usr/bin/env python3
"""BOP instance annotations Stage 5 needs — visibility and the test split root.

A Stage-5 trial is a BOP instance (dataset, scene, image, gt_idx).  Two things
about such an instance have to be read straight from the downloaded BOP data:
where the test split lives, and how much of the target was visible in that frame
(``visib_fract`` from ``scene_gt_info.json``), which is the sampling gate of the
plan (``grasping/plan.py``, ``experiments/stage5_grasping.py``).

Kept deliberately pure Python (no numpy, no PyBullet): the plan is built on the
host, the trial runs in the container.  ``grasping/sim_scene.BOP_DATASETS`` is
the sim-side twin of the test-root table below.  All locations come from
``config/paths.yaml`` via ``pipeline.paths``.
"""
from __future__ import annotations

import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline import paths as _paths_mod                             # noqa: E402

_DATA = _paths_mod.resolve()["datasets_root"]

# BOP test roots. The first candidate that exists and is non-empty wins (same
# rule as sim_scene._first_dir).
_TEST_ROOTS = {
    "ycbv": [os.path.join(_DATA, "ycbv/test")],
    "tless": [os.path.join(_DATA, "tless/test_primesense")],
    "lmo": [os.path.join(_DATA, "lmo/test")],
}


def _test_root(ds: str) -> str:
    for c in _TEST_ROOTS[ds]:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return _TEST_ROOTS[ds][0]


_INFO_CACHE: dict = {}


def _visib(ds: str, scene: int, im: int, gt_idx: int):
    """visib_fract from BOP scene_gt_info.json (None if the file is absent)."""
    key = (ds, scene)
    if key not in _INFO_CACHE:
        p = os.path.join(_test_root(ds), f"{scene:06d}", "scene_gt_info.json")
        try:
            _INFO_CACHE[key] = json.load(open(p))
        except OSError:
            _INFO_CACHE[key] = None
    info = _INFO_CACHE[key]
    try:
        return round(float(info[str(im)][gt_idx]["visib_fract"]), 4)
    except (TypeError, KeyError, IndexError):
        return None
