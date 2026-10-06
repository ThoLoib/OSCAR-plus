#!/usr/bin/env python3
"""The draw rule of the Stage-5 plan — how instances are picked per object.

Per object the plan draws up to ``PER_OBJECT`` instances from the Stage-3 records
in which exactly the planned proxy was rank 1 (the link to the real retrieval is
thereby kept; in the solo scenario the instance only supplies the camera and the
initial pose). Drawing is round-robin over the scenes (sorted) and, within one
scene, evenly spaced over the sorted frames; instances with
``visib >= MIN_VISIB`` are preferred, measured on the BOP annotation
(``grasping/instances._visib``).

The caller is ``experiments/stage5_grasping.py`` (plan construction). This module
holds only the rule and its two parameters — no object list, no paths.
"""
from __future__ import annotations

import collections

PER_OBJECT = 10          # 2026-09-11 von 6 auf 10 erhoeht; Faelle mit kleinerem
                         # Rang-1-Pool werden beim Pool-Maximum gekappt (geloggt)
MIN_VISIB = 0.5


def draw(insts: list, k: int) -> list:
    """Round-robin over scenes (sorted), within a scene evenly spaced over
    the sorted frames — the draw rule of the Stage-5 protocol."""
    by_scene = collections.defaultdict(list)
    for r in insts:
        by_scene[r["scene"]].append(r)
    for s in by_scene:
        by_scene[s].sort(key=lambda r: r["im"])
        n = len(by_scene[s])
        order = sorted(range(n), key=lambda i: (i * n) % n)  # stabil
        # gleichmaessig: nimm Indizes in Reihenfolge maximaler Spreizung
        spread = []
        step = max(1, n // max(1, min(k, n)))
        seen = set()
        for start in range(step):
            for i in range(start, n, step):
                if i not in seen:
                    seen.add(i)
                    spread.append(by_scene[s][i])
        by_scene[s] = spread
    out, scenes = [], sorted(by_scene)
    while len(out) < k and any(by_scene[s] for s in scenes):
        for s in scenes:
            if by_scene[s] and len(out) < k:
                out.append(by_scene[s].pop(0))
    return out
