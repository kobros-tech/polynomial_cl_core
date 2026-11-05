# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import gzip
import struct

import numpy as np
import pytest

from polynomial_cl.continual import make_feature_data, run_comparison
from polynomial_cl.data import load_projected_mnist, mnist


def write_tiny_mnist(root, per_class=2):
    """Write all four MNIST archives with random 28x28 images (no network needed)."""
    rng = np.random.default_rng(71)
    root.mkdir(parents=True, exist_ok=True)

    def write(name, header, payload):
        with gzip.open(root / name, "wb") as stream:
            stream.write(header + payload)

    for prefix, labels in (
        ("train", np.repeat(np.arange(10, dtype=np.uint8), per_class)),
        ("t10k", np.repeat(np.arange(10, dtype=np.uint8), per_class)),
    ):
        images = rng.integers(0, 256, size=(len(labels), 28, 28), dtype=np.uint8)
        header = struct.pack(">IIII", 2051, len(labels), 28, 28)
        write(f"{prefix}-images-idx3-ubyte.gz", header, images.tobytes())
        write(
            f"{prefix}-labels-idx1-ubyte.gz",
            struct.pack(">II", 2049, len(labels)),
            labels.tobytes(),
        )


def test_read_idx_images_and_labels(tmp_path):
    images = tmp_path / "images.gz"
    with gzip.open(images, "wb") as f:
        f.write(struct.pack(">IIII", 2051, 2, 2, 2) + bytes(range(8)))
    got = mnist.read_idx(images, image=True)
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got[1], [4, 5, 6, 7])

    labels = tmp_path / "labels.gz"
    with gzip.open(labels, "wb") as f:
        f.write(struct.pack(">II", 2049, 3) + bytes([1, 4, 9]))
    np.testing.assert_array_equal(mnist.read_idx(labels, image=False), [1, 4, 9])


@pytest.mark.parametrize(
    ("image", "payload", "message"),
    [
        (
            True,
            struct.pack(">IIII", 2051, 2, 2, 2) + bytes([0, 1, 2]),
            "invalid image IDX length",
        ),
        (
            True,
            struct.pack(">IIII", 1234, 1, 1, 1) + bytes([0]),
            "invalid image IDX magic",
        ),
        (True, b"\x00" * 5, "truncated image IDX header"),
        (False, struct.pack(">II", 1234, 1) + bytes([0]), "invalid label IDX magic"),
        (False, struct.pack(">II", 2049, 3) + bytes([0]), "invalid label IDX length"),
        (False, b"\x00" * 3, "truncated label IDX header"),
    ],
)
def test_read_idx_rejects_malformed_files(tmp_path, image, payload, message):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(payload)
    with pytest.raises(ValueError, match=message):
        mnist.read_idx(path, image=image)


def test_no_download_mode_reports_missing_files_without_touching_the_network(
    tmp_path, monkeypatch
):
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("fetch must not be called when downloads are disabled")

    monkeypatch.setattr(mnist, "fetch", unexpected_fetch)
    with pytest.raises(FileNotFoundError, match="download is disabled") as error:
        mnist.load_mnist(tmp_path, download=False)
    assert all(name in str(error.value) for name in mnist.FILES.values())


def test_load_mnist_returns_scaled_images_and_labels(tmp_path):
    write_tiny_mnist(tmp_path)
    x_train, y_train, x_test, y_test = mnist.load_mnist(tmp_path, download=False)
    assert x_train.shape == (20, 784) and x_test.shape == (20, 784)
    assert y_train.shape == y_test.shape == (20,)
    assert x_train.dtype == np.float32 and 0.0 <= x_train.min() and x_train.max() <= 1.0


def test_split_mnist_runs_end_to_end_offline(tmp_path):
    write_tiny_mnist(tmp_path)
    x_train, y_train, x_test, y_test = load_projected_mnist(
        tmp_path, projection_seed=4, download=False
    )
    assert x_train.shape == (20, 16)
    data = make_feature_data(x_train, y_train, x_test, y_test)
    result = run_comparison(
        data, [0], epochs=1, learning_rate=0.01, batch_size=8, memory_per_class=1
    )
    expanding = next(r for r in result["runs"] if r["condition"] == "expanding")
    assert expanding["feature_count_by_task"] == [17, 153, 969, 969, 969]
