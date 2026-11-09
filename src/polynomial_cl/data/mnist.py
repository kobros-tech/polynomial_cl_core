# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""MNIST download, parsing and fixed random projection (NumPy only).

``load_projected_mnist`` normalises MNIST and maps it to ``N_COMPONENTS`` dimensions by a fixed
random orthonormal projection that is never fitted to data. The class partition into
experiences is *not* decided here; that belongs to ``polynomial_cl.continual``.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

N_COMPONENTS = 16
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def fetch(root: Path, name: str) -> Path:
    """Download one archive atomically so an interrupted transfer is not reused."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.exists():
        try:
            with gzip.open(path, "rb") as stream:
                stream.read(1)
            return path
        except (OSError, EOFError):
            path.unlink()

    request = urllib.request.Request(
        BASE_URL + name, headers={"User-Agent": "polynomial_cl_core/0.1"}
    )
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        with gzip.open(temporary, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_idx(path: Path, image: bool) -> np.ndarray:
    """Read and validate a compressed MNIST IDX image or label file."""
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if image:
        if len(raw) < 16:
            raise ValueError(f"truncated image IDX header in {path}")
        magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
        if magic != 2051:
            raise ValueError(f"invalid image IDX magic in {path}")
        expected = 16 + n * rows * cols
        if len(raw) != expected:
            raise ValueError(
                f"invalid image IDX length in {path}: expected {expected}, got {len(raw)}"
            )
        return (
            np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols).copy()
        )
    if len(raw) < 8:
        raise ValueError(f"truncated label IDX header in {path}")
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"invalid label IDX magic in {path}")
    expected = 8 + n
    if len(raw) != expected:
        raise ValueError(
            f"invalid label IDX length in {path}: expected {expected}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8, offset=8, count=n).copy()


def load_mnist(
    root: Path, *, download: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST images as float32 in [0, 1] and labels as uint8.

    When `download` is false, require all four archives to exist locally and
    never attempt network access, including when a local archive is corrupt.
    """
    if download:
        paths = {key: fetch(root, filename) for key, filename in FILES.items()}
    else:
        missing = [
            filename for filename in FILES.values() if not (root / filename).is_file()
        ]
        if missing:
            missing_list = "\n".join(f"  - {name}" for name in missing)
            raise FileNotFoundError(
                f"MNIST download is disabled; missing files under {root}:\n{missing_list}"
            )
        paths = {key: root / filename for key, filename in FILES.items()}
    return (
        read_idx(paths["train_images"], True).astype(np.float32) / 255,
        read_idx(paths["train_labels"], False),
        read_idx(paths["test_images"], True).astype(np.float32) / 255,
        read_idx(paths["test_labels"], False),
    )


def make_projection(seed: int, input_dim: int = 784, n_components: int = N_COMPONENTS):
    """Deterministic orthonormal random projection matrix (input_dim, n_components)."""
    if not 1 <= n_components <= input_dim:
        raise ValueError("n_components must be between 1 and input_dim")
    matrix = np.random.default_rng(seed).standard_normal((input_dim, n_components))
    projection, _ = np.linalg.qr(matrix, mode="reduced")
    return projection.astype(np.float32)


def project_images(images: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Apply the fixed MNIST normalisation, then the shared projection."""
    normalized = (np.asarray(images, dtype=np.float32) - 0.1307) / 0.3081
    if normalized.ndim != 2 or normalized.shape[1] != projection.shape[0]:
        raise ValueError(
            "images must be a 2D array matching projection input dimension"
        )
    return normalized @ projection


def load_projected_mnist(root, projection_seed=0, download=True):
    """Load MNIST from ``root`` and project it. Returns (train_x, train_y, test_x, test_y)."""
    train_x, train_y, test_x, test_y = load_mnist(Path(root), download=download)
    projection = make_projection(projection_seed)
    return (
        project_images(train_x, projection),
        train_y,
        project_images(test_x, projection),
        test_y,
    )
