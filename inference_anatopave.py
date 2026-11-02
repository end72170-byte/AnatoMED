"""Legacy import compatibility for inference_anatomed; use the AnatoMED module for new work.

Usage: python inference_anatopave.py --help (legacy CLI entry points only).
"""
from importlib import import_module as _import_module

_module = _import_module("inference_anatomed")
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

if __name__ == "__main__":
    _module.main()
