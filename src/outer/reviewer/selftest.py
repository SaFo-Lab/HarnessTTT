"""Compatibility shim — this module moved to ``outer.safety.reviewer.selftest``.

RESTRUCTURE-001: src/ was regrouped into themed subpackages; every historic
import path keeps working through this alias (including private names and
monkeypatching — both paths resolve to the same module object).
"""
if __name__ == "__main__":
    import runpy
    runpy.run_module("outer.safety.reviewer.selftest", run_name="__main__")
else:
    import sys as _sys
    import outer.safety.reviewer.selftest as _module
    _sys.modules[__name__] = _module
