# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Run the experiment scripts the way the tests need: from the checkout, with readable failures.

Two properties matter for diagnosing a failing script test:

* the child process imports ``polynomial_cl`` from this checkout's ``src`` (via PYTHONPATH),
  never from whatever copy happens to be installed in the environment, and runs from the
  repository root like a user would;
* when the child fails, the test fails with its command, exit status, stdout and stderr, instead
  of a bare ``CalledProcessError`` that hides the traceback the child printed.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"


def child_environment():
    """The parent's environment with this checkout's ``src`` first on PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH")
    parts = [str(SRC)] + ([existing] if existing else [])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}


def run_python(*args, check=True):
    """``python *args`` from the repository root; returns the CompletedProcess.

    With ``check`` (the default) a non-zero exit raises AssertionError carrying everything the
    child printed. Pass ``check=False`` for a command that is expected to fail.
    """
    command = [sys.executable, *map(str, args)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=child_environment(),
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed with exit status {result.returncode}: {' '.join(command)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def run_script(name, *args, check=True):
    """``python experiments/<name> *args``; see ``run_python``."""
    return run_python(ROOT / "experiments" / name, *args, check=check)
