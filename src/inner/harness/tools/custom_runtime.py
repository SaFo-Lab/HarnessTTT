"""Runtime dispatcher for M_phi-generated custom tools (h2spec/1.0).

A materialized candidate carries its generated tools under
``custom_tools/<name>.py`` (each defining ``def run(ctx, args): ...``, already
static-gated and self-tested by ``validate_harness``). agent.yaml binds every such
tool to this single dispatcher, passing the source path through NexAU
``extra_kwargs`` (the compiler also adds a const-valued schema property so
NexAU validates the merged internal argument even when a tool uses
``additionalProperties: false``):

    tools:
      - name: sample_hit_probe
        yaml_path: ./tools/sample_hit_probe.tool.yaml
        binding: inner.harness.tools.custom_runtime:custom_tool
        extra_kwargs: {_sah_py_path: ./custom_tools/sample_hit_probe.py}

The dispatcher builds a fresh :class:`ToolContext` per call over the active
session (contextvar bridge), so generated code reaches only the audited
capability surface. Return values are stringified and every exception is
trapped — a crashing custom tool must never kill the rollout.
"""
from __future__ import annotations

import builtins as _builtins
import importlib
import json
import hashlib
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Dict

from inner.harness.tools.code_policy import FORBIDDEN_ATTRIBUTES, FORBIDDEN_MODULES, IMPORT_WHITELIST
from inner.harness.tools.runtime import get_session
from inner.runtime.harness_sdk import ToolContext

_CACHE: Dict[str, Callable] = {}


# ----------------------------------------------------------------------------------------------------------------------
# restricted execution namespace: the static gate (outer.safety.static_gates) is a denylist over the AST and cannot see
# types, so a tool that passed it may still reach `dataclasses.sys` or `json.codecs`; here those chains do not resolve
# ----------------------------------------------------------------------------------------------------------------------
class GuardedModule:
    """A whitelisted module as generated code sees it: no private attributes, no I/O-capable names, and any module
    reached through an attribute must itself be whitelisted (``numpy.linalg`` yes, ``dataclasses.sys`` no)."""
    __slots__ = ("_mod",)

    def __init__(self, mod: types.ModuleType):
        object.__setattr__(self, "_mod", mod)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in FORBIDDEN_ATTRIBUTES:
            raise AttributeError(f"attribute {name!r} is not available to generated tools")
        value = getattr(object.__getattribute__(self, "_mod"), name)
        if isinstance(value, types.ModuleType):
            root = (value.__name__ or "").split(".")[0]
            if root in FORBIDDEN_MODULES or root not in IMPORT_WHITELIST:
                raise AttributeError(f"module {value.__name__!r} is not available to generated tools")
            return GuardedModule(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("generated tools cannot modify modules")

    def __repr__(self) -> str:
        return f"<guarded module {object.__getattribute__(self, '_mod').__name__}>"


def _guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root in FORBIDDEN_MODULES or root not in IMPORT_WHITELIST:
        raise ImportError(f"import of {name!r} is not allowed in generated tools")
    leaf = importlib.import_module(name)
    return GuardedModule(leaf if fromlist else importlib.import_module(root))


_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "bytes", "bytearray", "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "classmethod", "staticmethod", "property", "object", "__build_class__",
    "BaseException", "Exception", "ArithmeticError", "AssertionError", "AttributeError", "FloatingPointError", "ImportError",
    "IndexError", "KeyError", "LookupError", "NameError", "NotImplementedError", "OverflowError", "RecursionError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)
SAFE_BUILTINS: Dict[str, Any] = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}
SAFE_BUILTINS["__import__"] = _guarded_import


def exec_generated(source: bytes, filename: str) -> Dict[str, Any]:
    """Execute generated tool source in the restricted namespace and return its globals."""
    ns: Dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "__name__": "custom_tool"}
    exec(compile(source, filename, "exec"), ns)  # gated + self-tested upstream; restricted here
    return ns


def _load_run(py_path: Path) -> Callable:
    source = py_path.read_bytes()
    key = f"{py_path}:{hashlib.sha256(source).hexdigest()}"
    fn = _CACHE.get(key)
    if fn is not None:
        return fn
    ns = exec_generated(source, str(py_path))
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError(f"custom tool {py_path} defines no run()")
    _CACHE[key] = fn
    return fn


def _resolve_source(session, declared: str) -> Path:
    if not session.harness_dir:
        raise ValueError("active session has no candidate H2 root")
    raw = Path(declared)
    if raw.is_absolute():
        raise ValueError("absolute custom-tool paths are forbidden")
    root = Path(session.harness_dir).resolve()
    path = (root / raw).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("custom-tool path escapes candidate H2") from exc
    if len(relative.parts) != 2 or relative.parts[0] != "custom_tools" \
            or path.suffix != ".py" or not path.is_file():
        raise ValueError("custom-tool path must be custom_tools/<name>.py")
    return path


def custom_tool(
    _sah_py_path: str = "", py_path: str = "", **kwargs: Any,
) -> str:
    """Generic binding for every generated tool.

    ``_sah_py_path`` comes from trusted compiler-owned ``extra_kwargs``; the
    legacy ``py_path`` alias keeps old materialized packages readable. Model
    arguments arrive in ``kwargs``.
    """
    declared_path = _sah_py_path or py_path
    if not declared_path:
        return "custom tool error: no py_path bound"
    try:
        session = get_session()
    except Exception:
        return "custom tool error: no active session"
    name = Path(declared_path).stem
    session.record_tool_event(name, "invoked")
    try:
        source = _resolve_source(session, declared_path)
        ctx = ToolContext(session, session.custom_tool_scratch())
        out = _load_run(source)(ctx, dict(kwargs))
    except Exception as exc:
        session.record_tool_event(name, "error", str(exc))
        return "custom tool raised (ignored):\n" + traceback.format_exc(limit=3)[-600:]
    session.record_tool_event(name, "completed")
    if isinstance(out, str):
        session.record_tool_result(name, out)
        return out[:8000]
    try:
        rendered = json.dumps(out, allow_nan=False)
    except Exception as exc:
        session.record_tool_event(
            name, "error", f"non-JSON-safe generated-tool result: {exc}"
        )
        return "custom tool returned a non-JSON-safe result (ignored)"
    session.record_tool_result(name, out)
    return rendered[:8000]
