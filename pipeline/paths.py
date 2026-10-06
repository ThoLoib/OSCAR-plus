"""Central path registry.

Reads config/paths.yaml once and resolves relative entries against the
repository root, so no script needs a hard-coded path.  Override either by
editing the YAML, pointing OSCAR_PATHS_FILE at another file, or per CLI flag
(the experiment drivers pass overrides into :func:`resolve`).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FILE = os.path.join(REPO_ROOT, "config", "paths.yaml")

_KEYS = {
    "datasets_root": ("data", "datasets_root"),
    "cad_root": ("data", "cad_root"),
    "gallery_root": ("generated", "gallery_root"),
    "caches_root": ("generated", "caches_root"),
    "runs_root": ("outputs", "runs_root"),
    "checkpoint_ulip2": ("checkpoints", "ulip2"),
    "checkpoint_ulip2_xyz": ("checkpoints", "ulip2_xyz"),
    "checkpoint_uni3d": ("checkpoints", "uni3d"),
    "blender": ("tools", "blender"),
}

# Keys that name executables or container-absolute files — never resolved
# against the repository root.
_NO_RESOLVE = {"blender", "checkpoint_ulip2", "checkpoint_ulip2_xyz",
               "checkpoint_uni3d"}


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Process-wide overrides, set once by the entry point from its CLI flags (see
# from_args).  Library modules such as evaluation/stage3_gallery.py and
# grasping/instances.py resolve their own paths when they are imported, which
# happens AFTER the entry point parsed its arguments — without this they would
# silently fall back to the YAML defaults and, for example, look for BOP scenes
# in an empty checkout.  So: a --datasets-root on the command line reaches every
# module, which is what "every path can be overridden per CLI" has to mean.
_PROCESS_OVERRIDES: Dict[str, str] = {}
_PROCESS_PATHS_FILE: Optional[str] = None


def set_process_overrides(overrides: Optional[Dict[str, str]] = None,
                          paths_file: Optional[str] = None) -> None:
    """Make CLI path overrides visible to every later resolve() in this process."""
    global _PROCESS_PATHS_FILE
    if overrides:
        _PROCESS_OVERRIDES.update({k: v for k, v in overrides.items()
                                   if v is not None})
    if paths_file:
        _PROCESS_PATHS_FILE = paths_file


def resolve(overrides: Optional[Dict[str, str]] = None,
            paths_file: Optional[str] = None) -> Dict[str, str]:
    """Return the flat path map {key: absolute path}.

    ``overrides`` maps flat keys (e.g. ``datasets_root``) to values that win
    over the YAML; ``None`` values are ignored so argparse defaults can be
    passed straight through.  Overrides registered by the entry point via
    ``set_process_overrides`` apply as well (the explicit argument wins).
    """
    # Empty counts as unset: docker-compose.yml forwards the variable as
    # ${OSCAR_PATHS_FILE:-}, which sets it to "" on hosts that never use it.
    fname = (paths_file or _PROCESS_PATHS_FILE
             or os.environ.get("OSCAR_PATHS_FILE") or _DEFAULT_FILE)
    raw = _load_yaml(fname)
    flat: Dict[str, str] = {}
    for key, (sect, name) in _KEYS.items():
        try:
            value = raw[sect][name]
        except (KeyError, TypeError):
            raise KeyError(f"{fname}: missing entry {sect}.{name}")
        flat[key] = value
    merged = dict(_PROCESS_OVERRIDES)
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key, value in merged.items():
        if key not in _KEYS:
            raise KeyError(f"unknown paths override: {key}")
        flat[key] = value
    for key, value in flat.items():
        if key not in _NO_RESOLVE and not os.path.isabs(value):
            flat[key] = os.path.join(REPO_ROOT, value)
    return flat


def add_path_args(parser) -> None:
    """Attach the standard --*-root/--checkpoint-* overrides to an argparse parser."""
    grp = parser.add_argument_group("paths (defaults from config/paths.yaml)")
    grp.add_argument("--paths-file", default=None,
                     help="alternative paths.yaml")
    for key in _KEYS:
        grp.add_argument("--" + key.replace("_", "-"), default=None,
                         help=f"override paths.yaml entry '{key}'")


def from_args(args) -> Dict[str, str]:
    """Build the path map from parsed argparse args (see add_path_args).

    Also registers the overrides process-wide, so modules that resolve their
    own paths on import see them too.
    """
    overrides = {key: getattr(args, key, None) for key in _KEYS}
    paths_file = getattr(args, "paths_file", None)
    set_process_overrides(overrides, paths_file)
    return resolve(overrides, paths_file=paths_file)
