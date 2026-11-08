#!/usr/bin/env python3
"""Download the canonical CVDF MNIST IDX data for CI without torch/torchvision.

SHA-256 checksums pin the compressed inputs. Never commit raw MNIST assets.
Source: https://github.com/cvdfoundation/mnist and the checksums published in
the MNIST dataset metadata (https://huggingface.co/datasets/severo/mnist).
"""
import argparse
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# (SHA-256 of compressed .gz, expected samples, expected uncompressed size, image?)
FILES = {
    "train-images-idx3-ubyte": (
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        60000, 16 + 60000 * 28 * 28, True,
    ),
    "train-labels-idx1-ubyte": (
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        60000, 8 + 60000, False,
    ),
    "t10k-images-idx3-ubyte": (
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        10000, 16 + 10000 * 28 * 28, True,
    ),
    "t10k-labels-idx1-ubyte": (
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        10000, 8 + 10000, False,
    ),
}


def validate_idx(data: bytes, count: int, expected_size: int, is_image: bool) -> None:
    assert len(data) == expected_size, f"IDX length {len(data)} != {expected_size}"
    if is_image:
        assert struct.unpack(">IIII", data[:16]) == (2051, count, 28, 28), "bad image IDX header"
    else:
        assert struct.unpack(">II", data[:8]) == (2049, count), "bad label IDX header"
        assert all(label <= 9 for label in data[8:]), "MNIST contains invalid labels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--train-only", action="store_true",
                        help="Download/verify only the official 60k train partition for validation-only runs")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, (digest, count, size, is_image) in FILES.items():
        if args.train_only and name.startswith("t10k-"):
            continue
        target = args.data_dir / name
        if target.exists():
            existing = target.read_bytes()
            validate_idx(existing, count, size, is_image)
            print(f"Existing IDX validated: {target}")
            continue
        url = MIRROR + name + ".gz"
        print(f"Fetching {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            compressed = response.read()
        actual = hashlib.sha256(compressed).hexdigest()
        if actual != digest:
            raise ValueError(f"Unexpected SHA-256 for {url}: {actual} != {digest}")
        data = gzip.decompress(compressed)
        validate_idx(data, count, size, is_image)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"Verified and extracted {name} ({count} rows, SHA-256 compressed {digest})", flush=True)


if __name__ == "__main__":
    main()
