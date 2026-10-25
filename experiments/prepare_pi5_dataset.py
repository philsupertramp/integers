#!/usr/bin/env python3
"""Build the shared, immutable MNIST i8 input bundle from pinned canonical IDX files.

The i8 pixels are the EXACT shared input representation already used by held_out:
round((raw / 255) * 254 - 127), interpreted as Dyadic(mantissa, shift=7).
FP32 decodes the same Dyadic inputs. Only the pixel normalization is precomputed;
no backend-specific training transforms, weights or statistics are baked.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from fetch_mnist import FILES, MIRROR, validate_idx

# SHA256 of uncompressed canonical IDX; preserved from the locked #15 report.
RAW_SHA256 = {
    "train-images-idx3-ubyte": "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    "train-labels-idx1-ubyte": "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    "t10k-images-idx3-ubyte": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "t10k-labels-idx1-ubyte": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
}
SPLIT_SEED = 81985529216486895
PIXEL_SHIFT = 7


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_to_signed(raw: int) -> int:
    # Nearest integer; ties never occur for p in [0,255] with this mapping.
    return math.floor((raw * 254 / 255 - 127) + 0.5)


def normalized_bytes(images: bytes) -> bytes:
    # Signed i8 two's-complement mantissas in original observation order.
    lut = bytes(pixel_to_signed(p) & 255 for p in range(256))
    return images.translate(lut)


def read_source(path: Path, name: str) -> bytes:
    content = path.read_bytes()
    compressed_sha, count, size, is_image = FILES[name]
    validate_idx(content, count, size, is_image)
    actual = sha256(content)
    if actual != RAW_SHA256[name]:
        raise ValueError(f"{name}: uncompressed SHA256 mismatch: {actual}")
    return content


def prepare(idx_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    artifacts = {}
    for split, count, prefix in (("train", 60000, "train"), ("test", 10000, "t10k")):
        image_name = f"{prefix}-images-idx3-ubyte"
        label_name = f"{prefix}-labels-idx1-ubyte"
        images = read_source(idx_dir / image_name, image_name)
        labels = read_source(idx_dir / label_name, label_name)
        for name in (image_name, label_name):
            sources[name] = {
                "url": MIRROR + name + ".gz",
                "compressed_sha256": FILES[name][0],
                "idx_sha256": RAW_SHA256[name],
            }
        assert struct.unpack(">IIII", images[:16]) == (2051, count, 28, 28)
        assert struct.unpack(">II", labels[:8]) == (2049, count)
        outputs = {
            f"{split}-inputs.i8": normalized_bytes(images[16:]),
            f"{split}-labels.u8": labels[8:],
        }
        for name, content in outputs.items():
            assert len(content) == (count * 784 if name.endswith(".i8") else count)
            (out_dir / name).write_bytes(content)
            artifacts[name] = {"sha256": sha256(content), "bytes": len(content)}
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "MNIST",
            "source_version": "CVDF canonical IDX, SHA256-pinned",
            "source": "https://github.com/cvdfoundation/mnist",
            "license": "CC-BY-SA-3.0",
            "license_information": "https://keras.io/api/datasets/mnist/",
            "attribution": "Yann LeCun and Corinna Cortes (MNIST; derived from NIST)",
            "redistribution": "CC BY-SA 3.0 attribution, license link, modification notice; see DATASET_NOTICE.txt",
        },
        "sources": sources,
        "preprocessing": {
            "version": 1,
            "format": "row-major signed i8 pixel mantissas and u8 labels",
            "shape": [28, 28],
            "shift": PIXEL_SHIFT,
            "pixel_mapping": "round((pixel / 255.0) * 254.0 - 127.0)",
            "order": "original canonical IDX order; no shuffling",
            "targets": "label -> 127/128 one-hot, computed by the runner",
            "backend_specific_preparation": False,
        },
        "splits": {
            "algorithm": "experiments/protocol.rs::stratified_validation",
            "seed": SPLIT_SEED,
            "validation_per_class": 500,
            "train": {"source": "train IDX", "count": 55000, "identifier": f"mnist-train-stratified-{SPLIT_SEED}"},
            "validation": {"source": "train IDX", "count": 5000, "identifier": f"mnist-validation-stratified-{SPLIT_SEED}"},
            "test": {"source": "official t10k IDX", "count": 10000, "identifier": "mnist-official-test-canonical"},
            "test_reuse_disclosure": "Already used by the project's #15/#35 benchmarks; not an untouched confirmatory holdout.",
        },
        "artifacts": artifacts,
    }
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (out_dir / "dataset-manifest.json").write_bytes(raw)
    (out_dir / "dataset-manifest.sha256").write_text(sha256(raw) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.idx_dir, args.output_dir)
    print("Prepared MNIST:", manifest["artifacts"])
    print("Manifest SHA256:", (args.output_dir / "dataset-manifest.sha256").read_text().strip())


if __name__ == "__main__":
    main()
