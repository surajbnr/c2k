"""I4: `c2k/llm.py` is the only module that may import `anthropic` or
`ollama`. Everything else must go through `llm.call()`."""

from __future__ import annotations

import ast
from pathlib import Path

C2K_ROOT = Path(__file__).resolve().parent.parent / "c2k"
ALLOWED = {C2K_ROOT / "llm.py"}
_SDK_MODULES = {"anthropic", "ollama"}


def _imported_top_level_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


def test_only_llm_module_imports_sdks() -> None:
    offenders: list[str] = []
    for py_file in C2K_ROOT.rglob("*.py"):
        if py_file in ALLOWED:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        modules = set(_imported_top_level_modules(tree))
        hit = modules & _SDK_MODULES
        if hit:
            offenders.append(f"{py_file.relative_to(C2K_ROOT.parent)}: {sorted(hit)}")
    assert not offenders, f"direct SDK import(s) found outside llm.py: {offenders}"


def test_llm_module_imports_are_lazy() -> None:
    """anthropic/ollama must not be imported at module load time in llm.py
    either — only inside the provider closures — so `import c2k.llm` never
    requires either SDK to be installed."""
    tree = ast.parse((C2K_ROOT / "llm.py").read_text(encoding="utf-8"))
    top_level_modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_modules.add(node.module.split(".")[0])
    assert not (top_level_modules & _SDK_MODULES)
