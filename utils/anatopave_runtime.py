"""Legacy import compatibility for utils.anatomed_runtime; use the AnatoMED module for new work.

This module preserves historical imports; canonical code uses the renamed module.
"""
from importlib import import_module as _import_module

_module = _import_module("utils.anatomed_runtime")
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})
