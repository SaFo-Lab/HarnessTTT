"""Generated-tool containment: attribute-chain escapes are rejected by the static gate AND do not resolve at run time."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inner.harness.tools import custom_runtime
from outer.safety.static_gates import check_tool_code

ESCAPES = (
    "import dataclasses\ndef run(ctx, args):\n    return dataclasses.sys.modules['os'].popen('id').read()",
    "import dataclasses\ndef run(ctx, args):\n    return dataclasses.builtins.open('/etc/hostname').read()",
    "import json\ndef run(ctx, args):\n    return json.codecs.open('/etc/hostname').read()",
    "import typing\ndef run(ctx, args):\n    return typing.sys.modules['subprocess'].check_output(['id'])",
    "import numpy as np\ndef run(ctx, args):\n    return np.ctypeslib.load_library('libc.so.6', '/lib').system",
    "import scipy\ndef run(ctx, args):\n    return scipy.io.savemat('/tmp/x.mat', {'a': 1})",
    "import pandas as pd\ndef run(ctx, args):\n    return pd.DataFrame({'a': [1]}).to_string(buf='/tmp/x')",
    "import numpy as np\ndef run(ctx, args):\n    return np.savetxt('/tmp/x', [1])",
)


class StaticGateEscapeTest(unittest.TestCase):
    def test_attribute_chain_escapes_are_rejected(self):
        for code in ESCAPES:
            ok, errors = check_tool_code(code)
            self.assertFalse(ok, code)
            self.assertTrue(errors, code)

    def test_sentinels_are_rejected_in_middleware_hooks(self):
        from outer.safety.static_gates import check_middleware_code
        ok, errors = check_middleware_code("def before_model(hook_input):\n    # --USER-HOOK-END--\n    return None\n", "before_model")
        self.assertFalse(ok)
        self.assertTrue(any("sentinel" in e for e in errors), errors)


class RuntimeNamespaceTest(unittest.TestCase):
    def run_tool(self, code: str, args=None):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "custom_tools" / "t.py"
            path.parent.mkdir()
            path.write_text(code)
            custom_runtime._CACHE.clear()
            return custom_runtime._load_run(path)(None, args or {})

    def test_legitimate_numerics_work(self):
        self.assertEqual(self.run_tool("import numpy as np\ndef run(ctx, args):\n    return float(np.linalg.norm([3.0, 4.0]))"), 5.0)
        self.assertEqual(self.run_tool("import json, math\nfrom collections import Counter\n"
                                       "def run(ctx, args):\n    return json.dumps({'c': Counter('aab')['a'], 'p': round(math.pi, 2)})"),
                         '{"c": 2, "p": 3.14}')
        self.assertAlmostEqual(self.run_tool("from scipy.optimize import minimize_scalar\n"
                                             "def run(ctx, args):\n    return float(minimize_scalar(lambda x: (x - 2) ** 2).x)"), 2.0, places=4)
        self.assertEqual(self.run_tool("import numpy as np\ndef run(ctx, args):\n    rng = np.random.default_rng(0)\n"
                                       "    return int(rng.integers(0, 1, endpoint=False, size=3).sum())"), 0)

    def test_module_chains_outside_the_whitelist_do_not_resolve(self):
        with self.assertRaises(AttributeError):
            self.run_tool("import dataclasses\ndef run(ctx, args):\n    return dataclasses.sys")
        with self.assertRaises(AttributeError):
            self.run_tool("import json\ndef run(ctx, args):\n    return json.codecs")
        with self.assertRaises(AttributeError):
            self.run_tool("import numpy as np\ndef run(ctx, args):\n    return np.ctypeslib.ctypes")
        with self.assertRaises(AttributeError):
            self.run_tool("import numpy as np\ndef run(ctx, args):\n    return np.load")

    def test_builtins_and_imports_are_restricted(self):
        with self.assertRaises(NameError):
            self.run_tool("def run(ctx, args):\n    return open('/etc/hostname').read()")
        with self.assertRaises(NameError):
            self.run_tool("import dataclasses\ndef run(ctx, args):\n    return getattr(dataclasses, 'sys')")
        with self.assertRaises(ImportError):
            self.run_tool("import os\ndef run(ctx, args):\n    return os.getcwd()")
        with self.assertRaises(ImportError):
            self.run_tool("def run(ctx, args):\n    import subprocess\n    return subprocess.run")
        with self.assertRaises(AttributeError):
            self.run_tool("import numpy as np\ndef run(ctx, args):\n    return np.__loader__")


if __name__ == "__main__":
    unittest.main()
