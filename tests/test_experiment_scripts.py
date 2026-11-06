# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / name), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_class_incremental_script_writes_a_self_describing_json_result(tmp_path):
    out = tmp_path / "result.json"
    run_script(
        "class_incremental.py",
        "--dataset",
        "synthetic_2d",
        "--seeds",
        0,
        1,
        "--epochs",
        1,
        "--memory-per-class",
        2,
        "--train-per-class",
        6,
        "--test-per-class",
        4,
        "--output",
        out,
    )
    result = json.loads(out.read_text())
    assert result["dataset"] == "synthetic_2d"
    assert result["protocol"]["seeds"] == [0, 1]
    assert result["protocol"]["degrees_by_condition"]["expanding"] == [1, 2, 3, 3, 3]
    assert len(result["runs"]) == 6
    assert set(result["summary"]) == {"fixed_linear", "expanding", "fixed_capacity"}
    first = result["runs"][0]["accuracy_matrix"]
    assert (
        first[0][1] is None and first[1][1] is not None
    )  # unseen experiences are null


def test_class_incremental_script_fails_cleanly_without_mnist_files(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "class_incremental.py"),
            "--dataset",
            "split_mnist",
            "--no-download",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "x.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and "download is disabled" in proc.stderr
    assert not (tmp_path / "x.json").exists()


def test_degree_ablation_script_reports_all_three_conditions(tmp_path):
    out = tmp_path / "ablation.json"
    run_script(
        "synthetic_degree_ablation.py",
        "--seeds",
        0,
        "--n-experiences",
        3,
        "--epochs",
        5,
        "--output",
        out,
    )
    result = json.loads(out.read_text())
    assert result["protocol"]["fixed_degree"] == 3
    assert set(result["summary"]) == {
        "expanding",
        "fixed_initial_degree",
        "fixed_capacity",
    }
    assert len(result["runs"]) == 3


def test_class_incremental_script_strategies_mode(tmp_path):
    out = tmp_path / "strategies.json"
    run_script(
        "class_incremental.py",
        "--dataset",
        "synthetic_2d",
        "--experiment",
        "strategies",
        "--seeds",
        0,
        "--epochs",
        1,
        "--memory-per-class",
        2,
        "--train-per-class",
        6,
        "--test-per-class",
        4,
        "--n-ref-per-class",
        1,
        "--output",
        out,
    )
    result = json.loads(out.read_text())
    assert result["protocol"]["experiment"] == "strategies"
    assert result["protocol"]["n_ref_per_class"] == 1
    assert set(result["summary"]) == {
        "ordinary",
        "ordinary_no_replay",
        "small_old_lr",
        "frozen_old",
        "strategy_b",
        "strategy_b_new",
    }


def test_rate_selection_script_selects_on_validation_and_reports_test(tmp_path):
    out = tmp_path / "selection.json"
    run_script(
        "rate_selection.py",
        "--dataset",
        "synthetic_2d",
        "--powers",
        0,
        1,
        "--lrs",
        0.05,
        0.2,
        "--select-seeds",
        0,
        "--test-seeds",
        1,
        2,
        "--epochs",
        1,
        "--memory-per-class",
        2,
        "--train-per-class",
        60,
        "--test-per-class",
        8,
        "--output",
        out,
    )
    result = json.loads(out.read_text())
    assert not set(result["protocol"]["select_seeds"]) & set(
        result["protocol"]["test_seeds"]
    )
    assert len(result["validation_grid"]) == 4
    assert {row["name"] for row in result["selected"]} == {
        "fixed_linear",
        "expanding",
        "fixed_capacity",
    }
    for row in result["selected"]:
        cell = (row["feature_power"], row["learning_rate"])
        best = max(
            g["final_mean_accuracy"][row["name"]] for g in result["validation_grid"]
        )
        assert row["validation_accuracy"] == best  # chosen by validation accuracy only
        assert (cell[0], cell[1]) in {
            (g["feature_power"], g["learning_rate"]) for g in result["validation_grid"]
        }
        assert 0 <= row["final_mean_accuracy"]["mean"] <= 1


def test_rate_selection_rejects_overlapping_selection_and_test_seeds(tmp_path):
    out = tmp_path / "selection.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "rate_selection.py"),
            "--dataset",
            "synthetic_2d",
            "--select-seeds",
            "0",
            "1",
            "--test-seeds",
            "1",
            "2",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and not out.exists()
    assert "overlap" in proc.stderr and "[1]" in proc.stderr


def test_rate_selection_defaults_use_disjoint_seed_sets(tmp_path):
    proc = run_script("rate_selection.py", "--help")
    assert "must be disjoint" in proc.stdout.replace("\n", " ")


def test_per_degree_script_runs_study_then_epoch_robustness(tmp_path):
    common = (
        "--dataset",
        "synthetic_2d",
        "--train-per-class",
        12,
        "--test-per-class",
        6,
        "--memory-per-class",
        4,
        "--n-ref-per-class",
        1,
        "--batch-size",
        16,
        "--strategies",
        "strategy_b_new",
        "--samplings",
        "class_balanced",
        "--validation-fraction",
        0.5,
    )
    cell = tmp_path / "cell_strategy_b_new_class_balanced.json"
    run_script(
        "per_degree_lr.py",
        *common,
        "--experiment",
        "per-degree",
        "--epochs",
        1,
        "--select-seeds",
        0,
        "--test-seeds",
        1,
        2,
        "--output",
        cell,
    )
    study = json.loads(cell.read_text())
    assert study["selection_seeds"] == [0] and study["test_seeds"] == [1, 2]
    (c,) = study["cells"]
    assert c["strategy"] == "strategy_b_new" and c["selected"] in c["pareto"]
    longer = tmp_path / "epochs.json"
    run_script(
        "per_degree_lr.py",
        *common,
        "--experiment",
        "epochs",
        "--epochs",
        2,
        "--cells-dir",
        tmp_path,
        "--output",
        longer,
    )
    result = json.loads(longer.read_text())
    assert result["epochs"] == 2 and result["cells"][0]["selected"] == c["selected"]
    with pytest.raises(
        subprocess.CalledProcessError
    ) as error:  # overlap rejected up front
        run_script(
            "per_degree_lr.py",
            *common,
            "--experiment",
            "per-degree",
            "--select-seeds",
            1,
            "--test-seeds",
            1,
            "--output",
            tmp_path / "bad.json",
        )
    assert "overlap" in error.value.stderr and not (tmp_path / "bad.json").exists()


def test_joint_training_script_binary_target_is_selectable_and_writes_scalars(tmp_path):
    results = {}
    for target in ("sinusoidal", "circular"):
        out = tmp_path / f"{target}.json"
        run_script(
            "joint_training.py",
            "--target",
            target,
            "--binary-degrees",
            1,
            2,
            "--binary-epochs",
            60,
            "--n-train",
            30,
            "--n-test",
            30,
            "--output",
            out,
        )
        results[target] = json.loads(out.read_text())
    rows = results["circular"]["rows"]
    assert [(r["degree"], r["epochs"]) for r in rows] == [(1, 60), (2, 60)]
    assert all("loss_history" not in r and "mean_forgetting" not in r for r in rows)
    assert results["circular"]["protocol"]["target"] == "circular"
    assert (
        rows != results["sinusoidal"]["rows"]
    )  # the target really changes the problem


def test_joint_training_script_ten_class_reference(tmp_path):
    out = tmp_path / "joint.json"
    run_script(
        "joint_training.py",
        "--experiment",
        "ten_class",
        "--dataset",
        "synthetic_2d",
        "--degrees",
        1,
        "--epochs",
        1,
        "--seeds",
        0,
        1,
        "--train-per-class",
        6,
        "--test-per-class",
        4,
        "--output",
        out,
    )
    row = json.loads(out.read_text())["rows"][0]
    assert row["seeds"] == [0, 1] and len(row["per_class_accuracy_mean"]) == 10
    assert "forgetting" not in json.dumps(row)


def test_joint_training_ten_class_requires_a_dataset():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "joint_training.py"),
            "--experiment",
            "ten_class",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and "--dataset" in proc.stderr
