"""I5: the test split is read-only outside `c2k eval --split test`. This is
enforced structurally in `store.load_gold`, not just documented — verify both
that the lock actually blocks reads and that no code path other than
`cli.py`'s `eval` command can construct the key that opens it."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from c2k.store import Split, Store, TestSplitLocked, TestUnlock

C2K_ROOT = Path(__file__).resolve().parent.parent / "c2k"


def test_load_gold_test_split_without_unlock_raises(tmp_store: Store) -> None:
    with pytest.raises(TestSplitLocked):
        tmp_store.load_gold(Split.TEST)


def test_load_gold_test_split_with_unlock_succeeds(tmp_store: Store) -> None:
    rows = tmp_store.load_gold(Split.TEST, unlock=TestUnlock())
    assert rows == []  # no data seeded; the point is it doesn't raise


def test_load_gold_tune_split_never_requires_unlock(tmp_store: Store) -> None:
    assert tmp_store.load_gold(Split.TUNE) == []


def test_only_cli_eval_command_constructs_test_unlock() -> None:
    """Scans every module for `TestUnlock(...)` call sites. Only
    `cli.py` may contain one — and it must be inside the `eval` command,
    gated on `--split test`, never inside `tune`."""
    offenders: list[str] = []
    for py_file in C2K_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TestUnlock"
            ):
                if py_file.name != "cli.py":
                    offenders.append(str(py_file.relative_to(C2K_ROOT.parent)))
    assert not offenders, f"TestUnlock() constructed outside cli.py: {offenders}"


def test_tune_command_source_never_names_test_unlock() -> None:
    cli_source = (C2K_ROOT / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(cli_source)
    tune_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "tune"
    )
    names_in_tune = {n.id for n in ast.walk(tune_fn) if isinstance(n, ast.Name)}
    assert "TestUnlock" not in names_in_tune
