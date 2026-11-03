# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Import structure: no cycles, independent settings, thin entry points, no stale imports."""

import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
PACKAGE = "polynomial_cl"
CIL = "polynomial_cl.continual"


def _modules():
    """dotted name -> file, for every module of the package."""
    found = {}
    for path in (SRC / PACKAGE).rglob("*.py"):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


def _imports(path, module):
    """Package-internal modules imported by one file (any depth, including lazy imports)."""
    is_package = path.name == "__init__.py"
    base = module.split(".") if is_package else module.split(".")[:-1]
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names if a.name.startswith(PACKAGE)}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = base[: len(base) - node.level + 1]
                target = ".".join(prefix + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith(PACKAGE):
                names.add(target)
                names |= {f"{target}.{a.name}" for a in node.names}
    return names


def _graph():
    modules = _modules()
    return {m: {i for i in _imports(p, m) if i in modules} for m, p in modules.items()}


def test_module_import_graph_has_no_cycles():
    graph = _graph()
    state = {}

    def visit(node, trail):
        if state.get(node) == "done":
            return
        assert state.get(node) != "open", f"import cycle: {' -> '.join([*trail, node])}"
        state[node] = "open"
        for dep in sorted(graph[node]):
            if dep != node and not (
                node.startswith(dep + ".") or dep.startswith(node + ".")
            ):
                visit(dep, [*trail, node])
        state[node] = "done"

    for module in graph:
        visit(module, [])


def _closure(graph, start):
    seen, todo = set(), [start]
    while todo:
        for dep in graph[todo.pop()]:
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def test_class_incremental_code_does_not_depend_on_binary_code():
    graph = _graph()
    forbidden = (
        "polynomial_cl.binary",
        "polynomial_cl.models",
        "polynomial_cl.training.binary",
    )
    for module in graph:
        if (
            module == CIL
            or module.startswith(CIL + ".")
            or module.endswith(("training.multiclass", "plotting.continual"))
        ):
            reached = _closure(graph, module) | {module}
            bad = [r for r in reached if r.startswith(forbidden)]
            assert not bad, f"{module} reaches binary-only code: {bad}"


def test_binary_code_does_not_depend_on_class_incremental_code():
    graph = _graph()
    for module in graph:
        if module.startswith(("polynomial_cl.binary", "polynomial_cl.models")):
            reached = _closure(graph, module)
            assert not [r for r in reached if r.startswith(CIL)], module


def test_every_module_imports_in_a_fresh_interpreter_order():
    import subprocess
    import sys

    for module in sorted(_modules()):
        done = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True, check=False, cwd=ROOT,
        )  # fmt: skip
        assert done.returncode == 0, f"{module}: {done.stderr[-300:]}"


def _imported_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith(PACKAGE):
                yield node.module, [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(PACKAGE):
                    yield a.name, []


def _sources():
    for folder in ("src", "tests", "experiments"):
        for path in (ROOT / folder).rglob("*.py"):
            yield path.name, ast.parse(path.read_text())
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for i, cell in enumerate(json.loads(nb.read_text())["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(
                    ln
                    for ln in cell["source"]
                    if not ln.lstrip().startswith(("%", "!"))
                )
                yield f"{nb.name}:{i}", ast.parse(code)


def test_no_stale_package_imports_anywhere():
    for where, tree in _sources():
        for module, names in _imported_names(tree):
            imported = importlib.import_module(module)
            for name in names:
                if name == "*" or hasattr(imported, name):
                    continue
                importlib.import_module(f"{module}.{name}")  # a submodule import


def test_experiment_entry_points_do_not_import_each_other():
    scripts = {p.stem for p in (ROOT / "experiments").glob("*.py")} - {
        "__init__",
        "common",
    }
    for path in (ROOT / "experiments").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            clash = {m.split(".")[0] for m in modules} & scripts
            assert not clash, f"{path.name} imports entry point(s) {clash}"


def test_top_level_package_exports_resolve():
    package = importlib.import_module(PACKAGE)
    for name in package.__all__:
        assert hasattr(package, name), name


@pytest.mark.parametrize("subpackage", ["binary", "continual"])
def test_subpackage_exports_resolve(subpackage):
    module = importlib.import_module(f"{PACKAGE}.{subpackage}")
    missing = [n for n in module.__all__ if not hasattr(module, n)]
    assert not missing
