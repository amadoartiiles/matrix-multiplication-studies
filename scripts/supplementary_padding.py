"""Supplementary experiment: row padding in C (tests the cache-conflict hypothesis).

Builds c/supplementary/matmul_padded.c with the main-study flags and runs it for
each size; pad = 0 and pad = 2 doubles are measured alternately in one process.
Writes results/supplementary/c_padding.csv.

    python scripts/supplementary_padding.py [--sizes 128 256 384 512 768 1024]
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[128, 256, 384, 512, 768, 1024])
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/supplementary")
    args = parser.parse_args()
    with open("config/benchmark_config.json") as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    exe = os.path.join("build", "matmul_padded" + (".exe" if sys.platform == "win32" else ""))
    subprocess.run([os.environ.get("CC", "gcc"), *cfg["c"]["flags"], "-o", exe,
                    "c/supplementary/matmul_padded.c", "-lm"], check=True)
    lines = []
    for n in args.sizes:
        res = subprocess.run([exe, "--n", str(n), "--data", args.data, "--pad", str(args.pad),
                              "--reps", "5"], capture_output=True, text=True, check=True)
        out = res.stdout.strip().splitlines()
        lines.extend(out[1:] if lines else out)
        for pad in (0, args.pad):
            t = sorted(float(l.split(",")[4]) for l in out[1:] if l.split(",")[1] == str(pad))
            print(f"n={n:5d} pad={pad}: median {t[len(t) // 2]:.4g} s", flush=True)
    with open(os.path.join(args.out, "c_padding.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
