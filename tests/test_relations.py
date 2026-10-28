"""I3: `t.relation` is a member of the enum built from relations.yaml. No
relation string literal may exist anywhere in the c2k package outside the
enum construction itself."""

from __future__ import annotations

import ast
from pathlib import Path

from c2k.config import RELATIONS_PATH
from c2k.schema import RELATION_DEFS

C2K_ROOT = Path(__file__).resolve().parent.parent / "c2k"


def _string_constants(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def test_relation_enum_matches_yaml() -> None:
    from c2k.schema import Relation

    assert {r.value for r in Relation} == set(RELATION_DEFS)


def test_no_relation_string_literals_outside_yaml() -> None:
    relation_names = set(RELATION_DEFS)
    offenders: list[str] = []
    for py_file in C2K_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for value in _string_constants(tree):
            if value in relation_names:
                offenders.append(f"{py_file.relative_to(C2K_ROOT.parent)}: {value!r}")
    assert not offenders, f"relation string literal(s) found outside relations.yaml: {offenders}"


def test_relations_yaml_shape() -> None:
    raw_names = list(RELATION_DEFS)
    assert 12 <= len(raw_names) <= 15, "relations.yaml must hold 12-15 relations (§12)"
    for name, entry in RELATION_DEFS.items():
        assert entry.get("definition"), f"{name} missing a definition"
        assert len(entry.get("exemplars", [])) == 2, f"{name} must have exactly two exemplars"
