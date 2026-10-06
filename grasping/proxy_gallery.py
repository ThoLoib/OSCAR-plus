#!/usr/bin/env python3
"""Proxy-CAD resolution for Stage 5 — the gallery G_proxy on disk.

Stage 5 plans its grasps on a CAD that Stage 3 retrieved, so it has to turn a
namespaced gallery id (``gso/...``, ``housecat6d/...``, ``itodd/...``) back into
a mesh file plus its unit convention.  This module is that lookup, and nothing
else: it mirrors ``evaluation/stage3_gallery.DATASET_LAYOUT`` /
``_pose_mesh_path`` (GSO + HouseCat6D in metres, ITODD in millimetres) so the
grasp study poses exactly the file Stage 3 scored.

``random_proxy`` is the predeclared chance baseline of the study (one
hash-drawn gallery CAD per target object).

Datasets (reference list: grasping/README.md): GSO [R10], HouseCat6D [R11],
ITODD [R9] form the proxy gallery G_proxy of Stage 3b [R5].

This module is pure Python (no numpy), so it also runs on a machine without the
sim stack.  All locations come from ``config/paths.yaml`` via ``pipeline.paths``.
"""
from __future__ import annotations

import glob
import os
import sys
import zlib
from typing import List, Optional, Tuple

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline import paths as _paths_mod                             # noqa: E402

_P = _paths_mod.resolve()
_CAD, _DATA = _P["cad_root"], _P["datasets_root"]


# ---------------------------------------------------------------------------
# Proxy CAD resolution — mirrors stage3_gallery.DATASET_LAYOUT / _pose_mesh_path
# (GSO + HouseCat6D in metres, ITODD in mm) with the one local fallback some
# machines need: ITODD CADs may live under <datasets_root>/itodd/models there.
# ---------------------------------------------------------------------------
def proxy_mesh(nsid: str) -> Tuple[Optional[str], bool]:
    """namespaced gallery id -> (mesh path or None, units_m)."""
    ds, oid = nsid.split("/", 1)
    if ds == "gso":
        return _exists(os.path.join(_CAD, "gso", oid, "meshes/model.obj")), True
    if ds == "housecat6d":
        hits = glob.glob(os.path.join(_CAD, "housecat6d", "*", oid + ".obj"))
        return (hits[0] if hits else None), True
    if ds == "itodd":
        p = os.path.join(_CAD, "itodd", oid, "model.ply")
        if not os.path.isfile(p):
            p = os.path.join(_DATA, "itodd/models", oid + ".ply")
        return _exists(p), False
    raise ValueError(f"not a proxy dataset: {nsid}")


def _exists(p: str) -> Optional[str]:
    return p if os.path.isfile(p) else None


def proxy_pool() -> List[str]:
    """G_proxy as used in Stage 3b — GSO (1030) ∪ HouseCat6D (199) ∪ ITODD (28)
    = 1257 gallery ids, enumerated from the CAD files present. Sorted, so the
    random-proxy draw below is stable across machines with the same data."""
    ids = []
    for d in sorted(glob.glob(os.path.join(_CAD, "gso/*/meshes/model.obj"))):
        oid = os.path.basename(os.path.dirname(os.path.dirname(d)))
        if oid != "models_orig":
            ids.append("gso/" + oid)
    for f in sorted(glob.glob(os.path.join(_CAD, "housecat6d/*/*.obj"))):
        ids.append("housecat6d/" + os.path.splitext(os.path.basename(f))[0])
    itodd = sorted(glob.glob(os.path.join(_CAD, "itodd/*/model.ply")))
    if itodd:
        ids += ["itodd/" + os.path.basename(os.path.dirname(f)) for f in itodd]
    else:
        ids += ["itodd/" + os.path.splitext(os.path.basename(f))[0] for f in
                sorted(glob.glob(os.path.join(_DATA, "itodd/models/obj_*.ply")))]
    return sorted(set(ids))


def random_proxy(dataset: str, obj_id: int, pool: List[str], salt: str = "random-proxy") -> str:
    """Predeclared baseline: ONE uniformly random gallery CAD per target object,
    drawn by a hash of the object's identity — no test outcome, no category
    label (none is available consistently across the three proxy sources), no
    RNG state to get wrong. The same object always gets the same proxy."""
    h = zlib.crc32(f"{salt}/{dataset}/{obj_id}".encode("utf-8"))
    return pool[h % len(pool)]
