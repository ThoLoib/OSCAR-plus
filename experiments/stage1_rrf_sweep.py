#!/usr/bin/env python3
"""
stage1_rrf_sweep.py
===================
RRF sensitivity over the Cormack factor c (Stage-1 robustness check E6).

Recomputes the RRF arm on the cached BASE score vectors for several c and pits
each c per-query paired against the weighted sum (E1c_full_fusion).
Reported as a SENSITIVITY, not a selection procedure: c = 60
[cormackReciprocalRankFusion2009] stays the frozen reported value; the sweep
only shows whether "weighted sum > RRF" depends on c.

This script computes NOTHING itself. It is a thin driver invocation: the sweep
logic sits unchanged in ``experiments/stage1_shrec18.py``
(``run_rrf_c_sweep``, reachable via its internal flag ``--rrf-c-sweep``), and
exactly that path is executed here via ``stage1_shrec18.main([...])`` —
including the preparation (``validate_inputs`` -> ``load_official_gt`` ->
``prepare_queries`` -> ``run_pass``) that matches the arms of the grid. The
driver puts its CSV next to the score stores (``<source>/rrf_c_sweep.csv``);
it is then copied to ``--out`` under the report name ``rrf_sweep.csv``.
Columns (unchanged from the driver, identical to the frozen
record ``results/stage1_shrec18/rrf_sweep.csv``)::

    c,n,nDCG_rrf,mAP_rrf,delta_weighted_minus_rrf,wins_weighted,wins_rrf,ties

``delta_weighted_minus_rrf`` > 0 means the weighted sum is ahead;
``wins_weighted``/``wins_rrf``/``ties`` is the per-query record against it.

Tier 2, no GPU run: the computation runs on the persisted score stores
(``<source>/_cache/scores_base.pt``, ``scores_ulip_pc_rgb.pt``) and the
per-query reference of the BASE arm.

How to run
----------
    python3 experiments/stage1_rrf_sweep.py
    python3 experiments/stage1_rrf_sweep.py --c-values 1,60,300

Produce the input (``--source``, default ``<runs_root>/stage1_shrec18``) with:

    python3 experiments/stage1_shrec18.py      # BASE arm + both score stores

Output (``--out``, default ``<runs_root>/stage1_rrf_sweep/rrf_sweep.csv``) plus
``run_config.json`` next to it. Outside the container the script wraps itself
automatically into ``docker compose run`` (the score stores are read with
torch).
"""
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

# ===========================================================================
# 0. CLI prelude
# ===========================================================================
# Laeuft VOR dem Import des Treibers (numpy) und vor run_pass (torch), damit
# ``--help`` und die Eingabepruefung ohne Container-Abhaengigkeiten laufen.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "evaluation") not in sys.path:
    sys.path.insert(1, os.path.join(_REPO, "evaluation"))

from configuration_names import to_release  # noqa: E402

_C_DEFAULT = "1,10,30,60,100,300"     # Berichtsliste der Arbeit
_BASE_ARM = "E1c_full_fusion"         # per-Query-Referenz der gewichteten Summe
_DRIVER_CSV = "rrf_c_sweep.csv"       # Dateiname, den der Treiber schreibt
_REPORT_CSV = "rrf_sweep.csv"         # Dateiname des eingefrorenen Berichts

_PATHS = None
_SOURCE = ""
_OUT_CSV = ""
_C_VALUES = []


def _base_per_query(source: str) -> str:
    """Per-query file of the BASE fusion — internal or release name.

    The driver computes the arm under its internal ablation key and renames the
    folder to the release name at the end (see
    ``evaluation/configuration_names.py``), so both are accepted; the release
    path is the one reported when nothing is there.
    """
    internal = os.path.join(source, _BASE_ARM, "results_per_query.json")
    if os.path.isfile(internal):
        return internal
    return os.path.join(source, to_release(_BASE_ARM, "stage1"),
                        "results_per_query.json")


def _require_source(source: str) -> None:
    """Abort with the generating command if a score store or the BASE reference is missing."""
    needed = [os.path.join(source, "_cache", "scores_base.pt"),
              os.path.join(source, "_cache", "scores_ulip_pc_rgb.pt"),
              _base_per_query(source)]
    missing = [p for p in needed if not os.path.isfile(p)]
    if not missing:
        return
    sys.exit(f"[rrf-sweep] inputs missing:\n  " + "\n  ".join(missing)
             + f"\nThe sweep computes on the persisted scores of the "
               f"BASE run and compares against its per-query nDCG. "
               f"Run Stage 1 first:\n"
               f"  python3 experiments/stage1_shrec18.py\n"
               f"(or point --source at a run that holds both).")


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    _parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _parser.add_argument("--source", default=None, metavar="RUN_DIR",
                         help="Stage-1 run with _cache/scores_*.pt and the "
                              f"arm {_BASE_ARM} (default: "
                              "<runs_root>/stage1_shrec18)")
    _parser.add_argument("--c-values", default=_C_DEFAULT, metavar="C_LIST",
                         help=f"Cormack factors, comma list "
                              f"(default: {_C_DEFAULT})")
    _parser.add_argument("--out", default=None,
                         help="target CSV (default: <runs_root>/"
                              f"stage1_rrf_sweep/{_REPORT_CSV})")
    _parser.add_argument("--no-docker", action="store_true",
                         help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(_parser)
    _args = _parser.parse_args()

    try:
        _C_VALUES = [int(x) for x in _args.c_values.split(",") if x.strip()]
    except ValueError:
        _C_VALUES = []
    if not _C_VALUES:
        sys.exit(f"[rrf-sweep] --c-values {_args.c_values}: comma list of "
                 f"integers expected (e.g. {_C_DEFAULT}).")

    # Gleiche Env wie der Stage-1-Treiber (die Score-Stores sind damit
    # entstanden), damit ein Neu-Lauf gar nicht abweichen koennte.
    os.environ.update({"PYTHONHASHSEED": "0", "SHREC_DINO_POOLING": "mean"})
    _PATHS = _paths_mod.from_args(_args)
    _SOURCE = _args.source or os.path.join(_PATHS["runs_root"],
                                           "stage1_shrec18")
    _OUT_CSV = _args.out or os.path.join(_PATHS["runs_root"],
                                         "stage1_rrf_sweep", _REPORT_CSV)
    _require_source(_SOURCE)             # schon auf dem Host, vor dem Wrappen

    if not os.path.exists("/.dockerenv") and not _args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[rrf-sweep] {_REPO}/docker-compose.yml missing — "
                     "repo incomplete?")
        print("[rrf-sweep] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage1_rrf_sweep.py"]
                  + sys.argv[1:])

if _PATHS is None:   # als Modul importiert: reine paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()
    _SOURCE = os.path.join(_PATHS["runs_root"], "stage1_shrec18")
    _OUT_CSV = os.path.join(_PATHS["runs_root"], "stage1_rrf_sweep", _REPORT_CSV)
    _C_VALUES = [int(x) for x in _C_DEFAULT.split(",")]

import stage1_shrec18 as E                                          # noqa: E402

# Der Treiber loest seine Modulkonstanten beim Import aus paths.yaml auf (die
# CLI-Overrides dieses Skripts sieht er nicht) — die beiden Eingaben, die er
# nicht per Flag annimmt, deshalb explizit nachziehen.
E.OFFICIAL_DIR = os.path.join(_PATHS["datasets_root"], "shrec18",
                              "shrec18_official")
E.DEFAULTS["stage1_root"] = os.path.join(_PATHS["datasets_root"], "shrec18",
                                         "stage1")

DRIVER_ARGV = [
    "--rrf-c-sweep", ",".join(str(c) for c in _C_VALUES),
    "--data-root", os.path.join(_PATHS["datasets_root"], "shrec18",
                                "shrec18_full"),
    "--images-dir", os.path.join(_PATHS["gallery_root"], "shrec18"),
    "--desc-file", os.path.join(_PATHS["cad_root"], "shrec18",
                                "descriptions_attributes.json"),
    "--results-root", _SOURCE,
]


def main():
    print(f"[rrf-sweep] c = {_C_VALUES}", flush=True)
    print(f"[rrf-sweep] driver call: stage1_shrec18.main("
          f"{DRIVER_ARGV})", flush=True)
    E.main(DRIVER_ARGV)

    produced = os.path.join(_SOURCE, _DRIVER_CSV)
    if not os.path.isfile(produced):
        sys.exit(f"[rrf-sweep] {produced} missing — the driver wrote no CSV "
                 f"(run incomplete).")
    os.makedirs(os.path.dirname(os.path.abspath(_OUT_CSV)), exist_ok=True)
    shutil.copyfile(produced, _OUT_CSV)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=_REPO).stdout.strip()
    except Exception:                                      # noqa: BLE001
        git = ""
    with open(os.path.join(os.path.dirname(os.path.abspath(_OUT_CSV)),
                           "run_config.json"), "w") as fh:
        json.dump({"argv": sys.argv[1:], "source": _SOURCE,
                   "c_values": _C_VALUES, "driver_argv": DRIVER_ARGV,
                   "driver_csv": produced, "out_csv": _OUT_CSV,
                   "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
                           "SHREC_DINO_POOLING": os.environ.get("SHREC_DINO_POOLING", "")},
                   "git": git,
                   "time": _dt.datetime.now().isoformat(timespec="seconds")},
                  fh, indent=1)
    print(f"[rrf-sweep] {produced} -> {_OUT_CSV}", flush=True)
    print(f"[rrf-sweep] reference of the thesis: "
          f"results/stage1_shrec18/{_REPORT_CSV}", flush=True)


if __name__ == "__main__":
    main()
