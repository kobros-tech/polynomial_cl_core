# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""The tests must exercise this checkout, not an older copy installed in the environment."""

from pathlib import Path

import pytest
from script_runner import SRC, run_python, run_script

import polynomial_cl

REINSTALL = "reinstall this checkout with: python -m pip install -e '.[dev]'"


def test_the_tests_import_this_checkout():
    imported = Path(polynomial_cl.__file__).resolve()
    assert SRC.resolve() in imported.parents, (
        f"polynomial_cl was imported from {imported}, not from {SRC}; {REINSTALL}"
    )


def test_child_processes_import_this_checkout():
    out = run_python("-c", "import polynomial_cl; print(polynomial_cl.__file__)")
    imported = Path(out.stdout.strip()).resolve()
    assert SRC.resolve() in imported.parents, f"child imported {imported}"


def test_a_failing_script_reports_what_the_child_printed():
    with pytest.raises(AssertionError) as caught:
        run_script("class_incremental.py", "--dataset", "nope")
    message = str(caught.value)
    assert "exit status 2" in message and "--- stderr ---" in message
    assert "invalid choice" in message  # the argparse error, not just the exit code


def test_expected_failures_can_be_inspected_without_raising():
    result = run_script("class_incremental.py", "--dataset", "nope", check=False)
    assert result.returncode == 2 and "invalid choice" in result.stderr
