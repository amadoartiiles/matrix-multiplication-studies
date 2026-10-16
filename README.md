# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027

Research question: **how does the matrix size n affect the runtime and memory use of
the same straightforward triple-loop dense matrix multiplication implemented in
C, Java and Python, on one fixed machine?**

The report is in `report/` (`report.tex` → `report.pdf`).

## Repository layout

```
c/matmul.c                 C implementation (+ self test + benchmark harness)
java/src/MatMul.java       Java implementation (+ self test + benchmark harness)
python/matmul.py           Python implementation (+ self test + benchmark harness)
config/benchmark_config.json  sizes, repetitions, warm-ups, budgets, seed, compiler/JVM flags
scripts/generate_inputs.py shared input matrices (data/A_n.bin, data/B_n.bin + manifest)
scripts/build.py           compiles C and Java into build/
scripts/run_benchmarks.py  driver: self tests, one process per (n, language), memory sampling, limits
scripts/validate_outputs.py correctness check of every result against NumPy with an error bound
scripts/collect_env.py     hardware / software description -> results/environment.json
scripts/analyze.py         summary tables, scaling fit and figures
scripts/supplementary_c_flags.py  supplementary: same C source with other GCC flags
scripts/supplementary_padding.py  supplementary: C kernel with rows padded by 16 bytes
c/supplementary/matmul_padded.c   kernel used by the padding experiment
results/raw/               raw measurements (measurements.csv, runs.csv)
results/replication/       second session for mid sizes (same protocol)
results/supplementary/     c_flags.csv, c_padding.csv
results/summary/           summary.csv, scaling_fit.csv, limits.csv, tables.tex
results/figures/           PNG figures used in the report
results/validation.csv     correctness results; results/selftest.txt known-answer tests
report/                    LaTeX source and PDF of the report
```

## Requirements (versions used for the reported results)

| Component | Version |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| C compiler | GCC 16.2.0 (MinGW-w64 x86_64-ucrt-posix-seh, WinLibs build) |
| Java | Oracle JDK 24.0.1 (HotSpot 64-bit Server VM) |
| Python | CPython 3.11.9 |
| Python packages (scripts only, not the benchmarked code) | numpy 2.2.5, matplotlib 3.10.1, psutil 7.2.2 |

The Python *implementation* (`python/matmul.py`) uses only the standard library.
NumPy is used only to generate inputs and as the correctness reference.

## Reproduce everything

Run from the repository root (Git Bash shown; PowerShell works the same with `python`):

```bash
pip install numpy==2.2.5 matplotlib==3.10.1 psutil==7.2.2
export JAVA_HOME="/c/Program Files/Java/jdk-24"   # use the real JDK, not the javapath launcher stub

python scripts/generate_inputs.py      # 1. shared inputs -> data/ (seed 20261012, uniform [-1, 1))
python scripts/build.py                # 2. gcc -O2 -std=c11 ... ; javac
python scripts/run_benchmarks.py       # 3. self tests + full benchmark (~1-1.5 h)
python scripts/validate_outputs.py     # 4. correctness vs NumPy
python scripts/collect_env.py          # 5. environment description
# optional, separate from the controlled comparison:
python scripts/run_benchmarks.py --languages python --sizes 512 768 --out results/replication
python scripts/run_benchmarks.py --languages c java --sizes 128 256 384 512 768 1024 --out results/replication
python scripts/supplementary_c_flags.py
python scripts/supplementary_padding.py
python scripts/analyze.py              # 6. tables + figures (includes the optional parts if present)
cd report && pdflatex report.tex && pdflatex report.tex   # 7. report
```

Individual programs can also be run directly:

```bash
build/matmul_c.exe selftest
build/matmul_c.exe bench --n 256 --data data --warmup 1 --reps 5 --min-batch-time 0.1
java -Xmx4g -cp build/java MatMul bench --n 256 --data data --warmup 3 --reps 5
python python/matmul.py bench --n 256 --data data --warmup 1 --reps 5
```

A quick pilot run: `python scripts/run_benchmarks.py --sizes 64 256 --reps 3 --out pilot`.

## Fixed experimental settings

* Algorithm: `C[i][j] = Σ_k A[i][k]·B[k][j]`, loop order i-j-k, local accumulator, same
  in all three languages; IEEE 754 binary64 everywhere.
* Timed region: allocation + zero initialisation of C and the triple loop (inputs are loaded
  before timing). Monotonic timers: `QueryPerformanceCounter` (C), `System.nanoTime` (Java),
  `time.perf_counter_ns` (Python).
* Repetitions: 5 measured repetitions per configuration; warm-ups 1 (C), 3 (Java), 1 (Python),
  excluded from statistics but kept in `measurements.csv`.
* Short runs are batched so that each measured sample lasts ≥ 0.1 s; the reported value is
  time per multiplication (`inner` column = multiplications per sample).
* Budgets: 300 s per single multiplication, 4 GB peak working set per process. A language
  that exceeds a budget is not attempted at larger sizes (recorded as `skipped`).
* Memory metric: peak working set of the process (Windows `PeakWorkingSetSize`), sampled every
  20 ms by the driver via psutil. Java additionally reports managed heap use (MemoryMXBean).

## Raw data format

`results/raw/measurements.csv`: `language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum`
(`phase` is `warmup` or `measure`; checksum = row-major sum of the result matrix).

`results/raw/runs.csv`: one row per (language, n) with status (`ok`, `timeout`, `memory_limit`,
`error`, `skipped`), wall time, peak working set, peak private bytes and Java heap values.

## Use of generative AI

Claude (Anthropic) was used to help write the implementations, scripts and a draft
of the report. All measurements in `results/` come from actually running the code on
the machine described in `results/environment.json`.
