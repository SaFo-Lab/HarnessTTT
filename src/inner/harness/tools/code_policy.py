"""What proposer-generated code (custom tools, middleware hooks) may use.

Shared by the proposer-side static gate (``outer.safety.static_gates``) and the executor-side runtime
(``inner.harness.tools.custom_runtime``), which executes generated tools in a namespace restricted to exactly this policy.
"""
from __future__ import annotations

# modules generated code may import (top-level package names)
IMPORT_WHITELIST = frozenset({
    "math", "re", "json", "itertools", "functools", "collections", "heapq",
    "bisect", "random", "statistics", "string", "typing", "dataclasses",
    "numpy", "pandas",
    # numerical optimisation + wall-clock budgeting: scipy is pure numerics (its I/O entry points are blocked by
    # FORBIDDEN_ATTRIBUTES), `time` has no I/O surface
    "scipy", "time",
})
# builtins generated code may not name
FORBIDDEN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview", "exit", "quit",
    "getattr", "setattr", "delattr", "type", "object", "super",
})
# import roots that are always refused (even if a future whitelist entry pulled them in)
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "multiprocessing", "threading", "signal", "resource", "pickle",
    "requests", "urllib", "http", "ftplib", "telnetlib", "builtins", "io",
})
# attribute names that reach the filesystem, processes or other modules from otherwise useful libraries
# (whitelisted modules re-export the modules they import: dataclasses.sys, json.codecs, typing.collections ...)
FORBIDDEN_ATTRIBUTES = frozenset({
    # file / network capable entry points of numpy, pandas, scipy
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "memmap",
    "savetxt", "loadtxt", "genfromtxt", "savemat", "loadmat", "mmread", "mmwrite", "wavfile", "netcdf_file",
    "read_csv", "read_json", "read_pickle", "read_parquet", "read_excel", "read_feather", "read_hdf", "read_sql",
    "read_table", "read_fwf", "read_html", "read_xml", "read_clipboard",
    "to_csv", "to_json", "to_pickle", "to_parquet", "to_excel", "to_feather", "to_hdf", "to_sql", "to_clipboard",
    "to_string", "to_html", "to_latex", "to_markdown", "to_xml",
    # modules and process/filesystem primitives reachable through attribute chains
    "sys", "os", "modules", "builtins", "codecs", "ctypes", "ctypeslib", "io", "subprocess", "importlib", "shutil",
    "pathlib", "socket", "signal", "threading", "multiprocessing", "resource", "pickle", "marshal", "shelve",
    "tempfile", "glob", "inspect", "gc", "weakref", "traceback", "runpy", "pkgutil", "sysconfig", "site", "platform",
    "environ", "system", "popen", "open", "spawn", "fork", "execv", "execl", "Popen", "check_output", "check_call",
    "import_module", "exec_module", "load_module", "run_path", "run_module", "urlopen", "urlretrieve", "request",
    # more numpy / pandas / scipy file-I/O and code-generation entry points reachable as plain (non-module) attributes.
    # Deliberately narrow so common safe names (json.dumps, re.compile, numpy.frombuffer) keep working: ndarray.dump writes
    # a file (json.dumps returns a string, so "dumps" is NOT forbidden), np.fromregex/np.DataSource open a path, np.f2py /
    # np.distutils / scipy.datasets run code or fetch over the network, pandas HDFStore/ExcelWriter/ExcelFile write files.
    "fromregex", "dump", "DataSource", "f2py", "distutils", "datasets", "HDFStore", "ExcelWriter", "ExcelFile",
    # string-driven attribute resolution: "{0.__self__}".format(len) / string.Formatter().get_field(...) resolve dotted
    # names that live inside string literals, invisible to an AST gate (use f-strings, whose expressions are AST nodes)
    "format", "format_map", "Formatter", "get_field", "get_value", "vformat", "convert_field", "check_unused_args",
    "get_type_hints", "ForwardRef", "evaluate_forward_ref", "attrgetter", "methodcaller", "query", "eval",
})
# NOTE: a bare string literal (e.g. a code template with "__ROW_COUNTS__" placeholders) is harmless because every way to
# turn a string into an attribute lookup is blocked above -- getattr/eval/exec/compile (FORBIDDEN_NAMES), .format/.format_map
# and string.Formatter().get_field (FORBIDDEN_ATTRIBUTES), pandas .eval/.query -- and f-string expressions are real AST nodes
# the gate inspects.  So generated tools may build program strings for ctx.stage_edit without tripping the gate.
MAX_CODE_CHARS = 10000
