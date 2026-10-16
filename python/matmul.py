"""Basic dense matrix multiplication (triple loop, i-j-k order) in pure Python.

Storage: a list of n row lists; each row holds n references to boxed Python
float objects (IEEE 754 binary64). No NumPy is used in this implementation.

Usage:
    python matmul.py selftest
    python matmul.py bench --n N --data DIR --warmup W --reps R --min-batch-time T [--dump FILE]

Bench output (stdout, CSV, one row per warm-up or measured repetition):
    language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum
"""
import argparse
import math
import os
import sys
import time
from array import array


def multiply(A, B, n):
    """Kernel: the timed region is exactly this function (allocation + triple loop)."""
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def checksum(C, n):
    """Row-major sequential sum (explicit loop, so the summation order matches C and Java)."""
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += C[i][j]
    return s


def read_matrix(path, n):
    flat = array("d")
    with open(path, "rb") as f:
        flat.fromfile(f, n * n)
    if sys.byteorder != "little":
        flat.byteswap()
    return [flat[i * n:(i + 1) * n].tolist() for i in range(n)]


def write_matrix(path, C, n):
    flat = array("d", (C[i][j] for i in range(n) for j in range(n)))
    if sys.byteorder != "little":
        flat.byteswap()
    with open(path, "wb") as f:
        flat.tofile(f)


def check(name, ok):
    print(f"{name:<28} {'PASS' if ok else 'FAIL'}")
    return ok


def selftest():
    ok = True
    A3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    B3 = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    E3 = [[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]
    ok &= check("known 3x3 product", multiply(A3, B3, 3) == E3)

    A4 = [[float(idx * 3 - 7) for idx in range(r * 4, r * 4 + 4)] for r in range(4)]
    I4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    Z4 = [[0.0] * 4 for _ in range(4)]
    ok &= check("A * I == A", multiply(A4, I4, 4) == A4)
    ok &= check("I * A == A", multiply(I4, A4, 4) == A4)
    ok &= check("A * 0 == 0", multiply(A4, Z4, 4) == Z4)

    A2 = [[1.0, 2.0], [0.0, 1.0]]
    B2 = [[1.0, 0.0], [3.0, 1.0]]
    ok &= check("non-commuting 2x2", multiply(A2, B2, 2) == [[7.0, 2.0], [3.0, 1.0]])
    ok &= check("1x1", multiply([[2.5]], [[-4.0]], 1) == [[-10.0]])

    print(f"selftest {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def bench(n, data_dir, warmup, reps, min_batch_time, dump_path):
    A = read_matrix(os.path.join(data_dir, f"A_{n}.bin"), n)
    B = read_matrix(os.path.join(data_dir, f"B_{n}.bin"), n)
    out = sys.stdout
    out.write("language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum\n")
    out.flush()

    last_single = 0.0
    for w in range(warmup):
        t0 = time.perf_counter_ns()
        C = multiply(A, B, n)
        t1 = time.perf_counter_ns()
        last_single = (t1 - t0) * 1e-9
        out.write(f"python,{n},warmup,{w},1,{last_single:.9e},{last_single:.9e},"
                  f"{checksum(C, n):.17g}\n")
        out.flush()

    # Batch very short multiplications so each sample lasts >= min_batch_time.
    inner = 1
    if 0.0 < last_single < min_batch_time:
        inner = math.ceil(min_batch_time / last_single)

    C = None
    for r in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            C = multiply(A, B, n)
        t1 = time.perf_counter_ns()
        batch = (t1 - t0) * 1e-9
        out.write(f"python,{n},measure,{r},{inner},{batch:.9e},{batch / inner:.9e},"
                  f"{checksum(C, n):.17g}\n")
        out.flush()

    if dump_path is not None and C is not None:
        write_matrix(dump_path, C, n)
    return 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        return selftest()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["bench"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--min-batch-time", type=float, default=0.1)
    parser.add_argument("--dump", default=None)
    args = parser.parse_args()
    return bench(args.n, args.data, args.warmup, args.reps, args.min_batch_time, args.dump)


if __name__ == "__main__":
    sys.exit(main())
