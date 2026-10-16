"""Supplementary experiment (separate from the controlled comparison): effect of
GCC optimisation flags on the *same* C source.

Builds c/matmul.c with several flag sets and runs each with the same protocol as
the main study (1 warm-up, 5 measured repetitions, batching to >= 0.1 s).
Writes results/supplementary/c_flags.csv.

    python scripts/supplementary_c_flags.py [--sizes 256 512 1024]
"""
import argparse
import csv
import os
import subprocess
import sys

VARIANTS = {
    "O2 (main study)": ["-O2"],
    "O2 -funroll-loops": ["-O2", "-funroll-loops"],
    "O3": ["-O3"],
    "O0": ["-O0"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[256, 512, 1024])
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/supplementary")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cc = os.environ.get("CC", "gcc")
    rows = []
    for idx, (name, flags) in enumerate(VARIANTS.items()):
        exe = os.path.join("build", f"matmul_c_variant{idx}" + (".exe" if sys.platform == "win32" else ""))
        subprocess.run([cc, *flags, "-std=c11", "-o", exe, "c/matmul.c", "-lm"], check=True)
        for n in args.sizes:
            res = subprocess.run([exe, "bench", "--n", str(n), "--data", args.data, "--warmup", "1",
                                  "--reps", "5", "--min-batch-time", "0.1"],
                                 capture_output=True, text=True, check=True)
            for line in res.stdout.splitlines()[1:]:
                rows.append([name, " ".join(flags), *line.split(",")])
            med = sorted(float(r[8]) for r in rows if r[0] == name and r[3] == str(n)
                         and r[4] == "measure")[2]
            print(f"{name:20s} n={n:5d} median {med:.4g} s", flush=True)
    with open(os.path.join(args.out, "c_flags.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "flags", "language", "n", "phase", "rep", "inner", "batch_time_s",
                    "time_per_mult_s", "checksum"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
