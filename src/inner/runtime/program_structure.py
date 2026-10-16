"""Trusted structural identity for candidate Python programs.

Literal values and bound-identifier spelling are erased before hashing, and
obvious inert statements are removed.  Coordinate/constant tuning, variable
renames, ``pass``, and ``x = x`` therefore stay in one family while genuine
control-flow or data-structure rewrites normally move to a new family.  This is
an operational search-control signal, not a claim of semantic equivalence.
"""
from __future__ import annotations

import ast
import hashlib
import json
import operator
from collections import Counter


STRUCTURAL_CANONICAL_SCHEMA = "sah-python-ast-structure/3.0"
_VOLATILE_AST_FIELDS = {"type_comment", "type_ignores", "type_params"}
_NOT_LITERAL = object()


def _literal_expression_value(node: ast.AST):
    """Evaluate a small, side-effect-free literal expression conservatively."""

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, bytes)) and len(value) > 4096:
            return _NOT_LITERAL
        if isinstance(value, int) and not isinstance(value, bool) \
                and abs(value) > 10 ** 18:
            return _NOT_LITERAL
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _literal_expression_value(node.operand)
        if operand is _NOT_LITERAL:
            return _NOT_LITERAL
        operation = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
            ast.Not: operator.not_,
            ast.Invert: operator.invert,
        }.get(type(node.op))
        if operation is None:
            return _NOT_LITERAL
        try:
            return operation(operand)
        except Exception:
            return _NOT_LITERAL
    if isinstance(node, ast.BinOp):
        left = _literal_expression_value(node.left)
        right = _literal_expression_value(node.right)
        if left is _NOT_LITERAL or right is _NOT_LITERAL:
            return _NOT_LITERAL
        operation = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.BitAnd: operator.and_,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
        }.get(type(node.op))
        if operation is None:
            return _NOT_LITERAL
        if isinstance(node.op, (ast.LShift, ast.RShift)) \
                and (not isinstance(right, int) or right < 0 or right > 64):
            return _NOT_LITERAL
        try:
            value = operation(left, right)
        except Exception:
            return _NOT_LITERAL
        if isinstance(value, (str, bytes)) and len(value) > 4096:
            return _NOT_LITERAL
        if isinstance(value, int) and not isinstance(value, bool) \
                and abs(value) > 10 ** 18:
            return _NOT_LITERAL
        return value
    if isinstance(node, ast.IfExp) and isinstance(node.test, ast.Constant):
        return _literal_expression_value(
            node.body if bool(node.test.value) else node.orelse
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_literal_expression_value(child) for child in node.elts]
        if any(value is _NOT_LITERAL for value in values):
            return _NOT_LITERAL
        return tuple(values)
    if isinstance(node, ast.Dict):
        keys = [_literal_expression_value(child) for child in node.keys]
        values = [_literal_expression_value(child) for child in node.values]
        if any(value is _NOT_LITERAL for value in [*keys, *values]):
            return _NOT_LITERAL
        return tuple(zip(keys, values))
    return _NOT_LITERAL


def _canonical_ast_value(value):
    """Serialize an AST without Python-minor-version-only empty fields.

    ``ast.dump`` changed when Python 3.12 added ``type_params`` to several
    nodes.  Artifacts produced under 3.12 therefore could not be replayed on
    3.9 even when the source and normalized behavior were identical.  This
    deliberately small wire schema includes semantic node fields and ignores
    annotations/locations whose presence varies across supported runtimes.
    """

    if isinstance(value, ast.AST):
        fields = []
        for name, child in ast.iter_fields(value):
            if name in _VOLATILE_AST_FIELDS:
                continue
            fields.append([name, _canonical_ast_value(child)])
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [_canonical_ast_value(child) for child in value]
    if isinstance(value, tuple):
        return [_canonical_ast_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_order_key(node: ast.AST) -> str:
    return json.dumps(
        _canonical_ast_value(node), sort_keys=True, separators=(",", ":")
    )


def _numeric_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _commutation_safe(node: ast.AST) -> bool:
    """Conservatively identify operands whose evaluation order has no effects."""

    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _commutation_safe(node.operand)
    if isinstance(node, ast.BinOp):
        return _commutation_safe(node.left) and _commutation_safe(node.right)
    return False


class _BoundIdentifiers(ast.NodeVisitor):
    """Collect names introduced by this program, including repeated binds.

    Free names such as ``sum`` and ``max`` remain meaningful operations and
    must not be alpha-normalized.  Store targets, parameters, definitions, and
    import aliases are local spellings and may be canonicalized safely enough
    for the operational family signal.
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.loads: Counter[str] = Counter()

    def _bind(self, name: str | None) -> None:
        if isinstance(name, str) and name:
            self.counts[name] += 1

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.loads[node.id] += 1

    def visit_arg(self, node: ast.arg) -> None:  # noqa: N802 - ast API
        self._bind(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._bind(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(  # noqa: N802 - ast API
        self, node: ast.AsyncFunctionDef,
    ) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._bind(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            self._bind(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        self._bind(node.name)
        self.generic_visit(node)


class _CanonicalStructure(ast.NodeTransformer):
    """Erase literals, alpha-normalize bound names, and drop clear no-ops."""

    def __init__(
        self, bindings: Counter[str], loads: Counter[str],
    ) -> None:
        self.bindings = bindings
        self.loads = loads
        self.identifiers: dict[str, str] = {}

    def _identifier(self, name: str) -> str:
        if name not in self.bindings:
            return name
        if name not in self.identifiers:
            self.identifiers[name] = f"bound_{len(self.identifiers):04d}"
        return self.identifiers[name]

    def visit_Constant(self, node: ast.Constant):  # noqa: N802 - ast API
        marker = type(node.value).__name__
        return ast.copy_location(ast.Constant(value=marker), node)

    def visit_Name(self, node: ast.Name):  # noqa: N802 - ast API
        node.id = self._identifier(node.id)
        return node

    def visit_arg(self, node: ast.arg):  # noqa: N802 - ast API
        node.arg = self._identifier(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # noqa: N802
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):  # noqa: N802
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias):  # noqa: N802 - ast API
        if node.asname:
            node.asname = self._identifier(node.asname)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler):  # noqa: N802
        if node.name:
            node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_Global(self, node: ast.Global):  # noqa: N802 - ast API
        node.names = [self._identifier(name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal):  # noqa: N802 - ast API
        node.names = [self._identifier(name) for name in node.names]
        return node

    def visit_MatchAs(self, node: ast.MatchAs):  # noqa: N802 - ast API
        if node.name:
            node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar):  # noqa: N802 - ast API
        if node.name:
            node.name = self._identifier(node.name)
        return node

    def visit_Pass(self, node: ast.Pass):  # noqa: N802 - ast API
        return None

    def visit_Expr(self, node: ast.Expr):  # noqa: N802 - ast API
        # Module/function docstrings and standalone literal expressions do not
        # change the candidate computation and cannot satisfy a structure gate.
        if _literal_expression_value(node.value) is not _NOT_LITERAL:
            return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If):  # noqa: N802 - ast API
        # A dead constant branch must not satisfy a structure-change gate.
        if isinstance(node.test, ast.Constant):
            selected = node.body if bool(node.test.value) else node.orelse
            rows = []
            for statement in selected:
                visited = self.visit(statement)
                if isinstance(visited, list):
                    rows.extend(visited)
                elif visited is not None:
                    rows.append(visited)
            return rows
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp):  # noqa: N802 - ast API
        if isinstance(node.test, ast.Constant):
            return self.visit(
                node.body if bool(node.test.value) else node.orelse
            )
        return self.generic_visit(node)

    def visit_While(self, node: ast.While):  # noqa: N802 - ast API
        # ``while False`` executes only its ``else`` suite. Retaining the loop
        # node would let unreachable control flow satisfy a structure gate.
        if isinstance(node.test, ast.Constant) and not bool(node.test.value):
            rows = []
            for statement in node.orelse:
                visited = self.visit(statement)
                if isinstance(visited, list):
                    rows.extend(visited)
                elif visited is not None:
                    rows.append(visited)
            return rows
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802 - ast API
        node = self.generic_visit(node)
        # Treat a purely syntactic operand swap as one operational family.
        # This is intentionally conservative: the signal is not a proof of
        # semantic equivalence and should not reward a cosmetic commutation.
        if (
            isinstance(node.op, (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor))
            and _commutation_safe(node.left)
            and _commutation_safe(node.right)
        ):
            if _canonical_order_key(node.right) < _canonical_order_key(node.left):
                node.left, node.right = node.right, node.left
        return node

    def visit_AugAssign(self, node: ast.AugAssign):  # noqa: N802 - ast API
        if (
            isinstance(node.target, ast.Name)
            and self.bindings[node.target.id] >= 2
            and isinstance(node.op, (ast.Add, ast.Sub))
            and _numeric_zero(node.value)
        ):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):  # noqa: N802 - ast API
        # Drop only the common, genuinely inert bound-name spelling ``x = x``.
        # A lone top-level ``x = x`` is retained because it can raise NameError.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self.bindings[node.targets[0].id] >= 2
        ):
            target = node.targets[0].id
            if isinstance(node.value, ast.Name) and target == node.value.id:
                return None
            if isinstance(node.value, ast.BinOp):
                left, right, op = node.value.left, node.value.right, node.value.op
                same_plus_zero = (
                    isinstance(op, ast.Add)
                    and (
                        isinstance(left, ast.Name) and left.id == target
                        and _numeric_zero(right)
                        or isinstance(right, ast.Name) and right.id == target
                        and _numeric_zero(left)
                    )
                )
                same_minus_zero = (
                    isinstance(op, ast.Sub)
                    and isinstance(left, ast.Name) and left.id == target
                    and _numeric_zero(right)
                )
                if same_plus_zero or same_minus_zero:
                    return None
        # Drop only side-effect-free constant assignments whose value is never
        # read. Calls, indexing, operators, and destructuring remain because
        # evaluating them can raise or mutate state.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self.loads[node.targets[0].id] == 0
            and isinstance(node.value, (
                ast.Constant, ast.UnaryOp, ast.BinOp, ast.IfExp,
            ))
            and _literal_expression_value(node.value) is not _NOT_LITERAL
        ):
            return None
        return self.generic_visit(node)


def normalized_ast_payload(program: str) -> str | None:
    """Return the versioned canonical AST payload, or ``None`` if invalid."""

    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, TypeError):
        return None

    collector = _BoundIdentifiers()
    collector.visit(tree)
    tree = _CanonicalStructure(collector.counts, collector.loads).visit(tree)
    ast.fix_missing_locations(tree)
    payload = json.dumps(
        {
            "schema": STRUCTURAL_CANONICAL_SCHEMA,
            "tree": _canonical_ast_value(tree),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload


def normalized_ast_digest(program: str) -> str:
    """Return a replay-stable 16-hex structural digest or ``invalid-python``."""

    payload = normalized_ast_payload(program)
    if payload is None:
        return "invalid-python"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
