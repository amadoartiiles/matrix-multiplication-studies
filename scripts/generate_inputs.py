"""Generate the shared input matrices used by all three implementations.

For every size n in the configuration, A_n.bin and B_n.bin are written to the
data directory as raw little-endian float64 arrays in row-major order. A
manifest with SHA-256 hashes is written so the inputs can be verified later.

    python scripts/generate_inputs.py [--config config/benchmark_config.json] [--out data]
"""
import argparse
import hashlib
import json
import os

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(n, seed, low, high):
    rng = np.random.default_rng([seed, n])
    A = rng.uniform(low, high, size=(n, n)).astype("<f8")
    B = rng.uniform(low, high, size=(n, n)).astype("<f8")
    return A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--out", default="data")
    parser.add_argument("--sizes", type=int, nargs="*", help="override the configured sizes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    inp = cfg["input"]
    sizes = args.sizes or cfg["sizes"]
    os.makedirs(args.out, exist_ok=True)

    manifest = {"numpy_version": np.__version__, "input": inp, "files": {}}
    for n in sizes:
        A, B = generate(n, inp["seed"], inp["low"], inp["high"])
        for name, M in (("A", A), ("B", B)):
            path = os.path.join(args.out, f"{name}_{n}.bin")
            M.tofile(path)
            manifest["files"][f"{name}_{n}.bin"] = {"n": n, "bytes": os.path.getsize(path),
                                                     "sha256": sha256(path)}
        print(f"n={n}: wrote A_{n}.bin and B_{n}.bin")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
