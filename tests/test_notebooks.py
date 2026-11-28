# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import ast
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_compile_and_are_committed_without_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), (
            f"{path.name} cell {index} has committed outputs"
        )
        # IPython magics are not Python syntax; the rest of the cell is.
        source = "".join(
            line for line in cell["source"] if not line.lstrip().startswith(("%", "!"))
        )
        compile(source, f"{path.name}:cell-{index}", "exec")


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_top_to_bottom_in_a_clean_kernel(path):
    """Run every cell in order in a fresh kernel; any exception fails the test.

    The notebook file is not modified. Split MNIST cells run if ``data/mnist`` exists and
    print a skip message otherwise, so this works on a fresh checkout.
    """
    nbformat = pytest.importorskip("nbformat")
    preprocessors = pytest.importorskip("nbconvert.preprocessors")
    notebook = nbformat.read(path, as_version=4)
    executor = preprocessors.ExecutePreprocessor(
        timeout=3600, kernel_name="python3", allow_errors=False
    )
    executor.preprocess(notebook, {"metadata": {"path": str(path.parent)}})
    # Part I of the main notebook must have produced its tables, not silently skipped.
    if path.name.startswith("02_"):
        printed = "".join(
            output.get("text", "")
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.outputs
        )
        assert "fixed_capacity" in printed and "relaxed" in printed


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_report_sample_standard_deviations(path):
    """Results are documented as sample std (ddof=1); NumPy's default is the population std."""
    import re

    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for call in re.finditer(r"\bstd\(([^()]*(?:\([^()]*\)[^()]*)*)\)", source):
            assert "ddof=1" in call.group(1), (path.name, index, call.group(0))


def test_exactly_the_two_research_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == [
        "01_joint_training.ipynb",
        "02_continual_learning.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cell_ids_are_valid_and_unique(path):
    import re

    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert (notebook["nbformat"], notebook["nbformat_minor"] >= 5) == (4, True)
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(
        isinstance(i, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids
    )
    assert len(set(ids)) == len(ids), "duplicate cell ids"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_run_their_own_experiments_and_read_no_result_files(path):
    """Tables and figures come from runs inside the notebook, never from saved result JSON."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for forbidden in ("results/", "load_study_cells", "json.load", "read_text"):
            assert forbidden not in source, (path.name, index, forbidden)


def test_joint_training_notebook_does_not_report_forgetting():
    """Notebook 01 is a reference without a sequence of experiences."""
    path = next(p for p in NOTEBOOKS if p.name.startswith("01_"))
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert "forgetting" not in "".join(cell["source"]).lower(), index


# Strategy names of the pre-C4 class-incremental API. The current methods are ``ordinary`` and
# ``relaxed`` (with ``beta``), and replay is the protocol option ``replay``. The names may be
# mentioned only in a markdown cell that is explicitly a "Migration note". The binary module of
# Part II has presets of its own that share some of these names; Part II is therefore excluded.
RETIRED_STRATEGY_NAMES = re.compile(
    r"strategy_b\w*|frozen_old|small_old_lr|ordinary_no_replay"
)


def _source(cell):
    return "".join(cell["source"])


def _class_incremental_cells(path):
    """Every cell of notebook 01; the cells of notebook 02 before its Part II heading."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and _source(cell).startswith("# Part II"):
            return cells[:index]
    return cells


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_class_incremental_cells_use_no_retired_strategy_names(path):
    for cell in _class_incremental_cells(path):
        source = _source(cell)
        found = RETIRED_STRATEGY_NAMES.findall(source)
        if cell["cell_type"] == "code":
            assert not found, (path.name, found, source[:80])
        elif found:
            assert "Migration note" in source, (path.name, found, source[:80])


def _package_callables(tree):
    """``{local name: object}`` for what a cell imports from the package (modules included)."""
    objects = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("polynomial_cl"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    name = alias.asname or alias.name
                    try:
                        objects[name] = getattr(module, alias.name)
                    except AttributeError:  # a submodule import
                        objects[name] = importlib.import_module(
                            f"{node.module}.{alias.name}"
                        )
    return objects


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_calls_pass_only_arguments_the_current_api_accepts(path):
    """Supplementary static guard: unknown keywords and too many positional arguments.

    Covered: calls whose function name is imported by the notebook itself from
    ``polynomial_cl`` (``from polynomial_cl... import f``) or reached through a module alias
    imported the same way (``P.plot_cil_tradeoff``). For those calls it checks that every
    explicit keyword exists in the current signature and that the positional arguments are
    not more than it takes.

    Not covered: calls on instances (``model.expanded(4)``), ``**kwargs`` unpacking (the
    keywords inside it are not seen), starred positional arguments, whether every *required*
    argument is supplied, argument types and values, and names the notebook defines or
    gets from other libraries. It does not replace executing the notebooks.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}  # imports accumulate over the cells, as in the kernel
    problems = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(
            line for line in cell["source"] if not line.lstrip().startswith(("%", "!"))
        )
        tree = ast.parse(code)
        namespace.update(_package_callables(tree))
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            func = call.func
            if isinstance(func, ast.Name):
                target = namespace.get(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                owner = namespace.get(func.value.id)
                target = (
                    getattr(owner, func.attr, None) if inspect.ismodule(owner) else None
                )
            else:
                continue
            if not (inspect.isfunction(target) or inspect.isclass(target)):
                continue
            try:
                parameters = inspect.signature(target).parameters
            except (TypeError, ValueError):
                continue
            takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
            takes_args = any(p.kind is p.VAR_POSITIONAL for p in parameters.values())
            positional = [
                p
                for p in parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            where = f"{path.name} cell {index}: {ast.unparse(func)}"
            for keyword in call.keywords:
                if keyword.arg is not None and not takes_kwargs:
                    if keyword.arg not in parameters:
                        problems.append(f"{where}: unexpected keyword {keyword.arg!r}")
            if not takes_args and not any(
                isinstance(a, ast.Starred) for a in call.args
            ):
                if len(call.args) > len(positional):
                    problems.append(f"{where}: too many positional arguments")
    assert not problems, "\n".join(problems)
