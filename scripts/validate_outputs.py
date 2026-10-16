"""Check every dumped result matrix against a trusted reference (NumPy A @ B).

Tolerance: the standard componentwise forward-error bound for a dot product of
length n evaluated in binary64 is |C_hat - C| <= gamma_n * (|A| |B|), with
gamma_n = n*u / (1 - n*u) and u = 2^-53 [Higham 2002, Sec. 3.5]. Both the tested
implementation and the reference satisfy that bound with respect to the exact
product, so their difference must satisfy

    |C_impl - C_ref| <= 2 * gamma_n * (|A| |B|)      (element-wise)

The script also reports whether the three implementations are bitwise identical
to each other (expected, because they evaluate the same sums in the same order
with IEEE 754 binary64 and no fast-math reassociation), and whether the
checksums of all repetitions agree.

    python scripts/validate_outputs.py [--data data] [--results results]
"""
import argparse
import csv
import os

import numpy as np

U = 2.0 ** -53


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    out_dir = os.path.join(args.results, "outputs")
    meas = list(csv.DictReader(open(os.path.join(args.results, "raw", "measurements.csv"))))
    sizes = sorted({int(r["n"]) for r in meas})
    langs = sorted({r["language"] for r in meas})

    rows = []
    for n in sizes:
        A = np.fromfile(os.path.join(args.data, f"A_{n}.bin"), dtype="<f8").reshape(n, n)
        B = np.fromfile(os.path.join(args.data, f"B_{n}.bin"), dtype="<f8").reshape(n, n)
        ref = A @ B
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(A) @ np.abs(B))
        results = {}
        for lang in langs:
            path = os.path.join(out_dir, f"{lang}_{n}.bin")
            if not os.path.exists(path) or os.path.getsize(path) != 8 * n * n:
                continue
            C = np.fromfile(path, dtype="<f8").reshape(n, n)
            results[lang] = C
            err = np.abs(C - ref)
            checks = {r["checksum"] for r in meas
                      if r["language"] == lang and int(r["n"]) == n}
            checks = {float(c) for c in checks}
            rows.append({
                "n": n, "language": lang,
                "max_abs_error": f"{err.max():.3e}",
                "max_rel_error": f"{(err / np.maximum(np.abs(ref), 1e-300)).max():.3e}",
                "max_error_over_bound": f"{(err / bound).max():.3e}",
                "elements_over_bound": int((err > bound).sum()),
                "within_tolerance": bool((err <= bound).all()),
                "checksums_consistent_across_reps": len(checks) == 1,
            })
        # Bitwise agreement between languages.
        names = sorted(results)
        for r in rows:
            if r["n"] == n:
                r["bitwise_equal_to"] = " ".join(
                    o for o in names if o != r["language"]
                    and np.array_equal(results[o], results[r["language"]]))
        print(f"n={n}: " + ", ".join(
            f"{r['language']} ok={r['within_tolerance']} maxabs={r['max_abs_error']}"
            for r in rows if r["n"] == n))

    path = os.path.join(args.results, "validation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    if not all(r["within_tolerance"] for r in rows):
        raise SystemExit("some results exceed the tolerance")


if __name__ == "__main__":
    main()
