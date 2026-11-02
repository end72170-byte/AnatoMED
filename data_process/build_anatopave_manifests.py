"""Legacy import compatibility for data_process.build_anatomed_manifests; use the AnatoMED module for new work.

Usage: python data_process/build_anatopave_manifests.py --help (legacy CLI entry points only).
"""
from importlib import import_module as _import_module
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

_module = _import_module("data_process.build_anatomed_manifests")
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

if __name__ == "__main__":
    _module.main()
