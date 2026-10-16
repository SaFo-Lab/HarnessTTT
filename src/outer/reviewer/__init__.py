"""Compatibility shim package — moved to ``outer.safety.reviewer``.

RESTRUCTURE-001: the submodule shims (reviewer.py, selftest.py) alias the
canonical modules; this __init__ stays a plain package so submodule imports
resolve through those shims to the same module objects.
"""
