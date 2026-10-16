"""Benchmark driver: runs every (size, language) configuration as a separate process.

For each configuration the driver
  * launches the implementation in "bench" mode with the shared input files,
  * reads its CSV rows (warm-up and measured repetitions) as they are produced,
  * samples the process' peak memory with psutil while it runs,
  * stops it if one multiplication exceeds the time limit or the process exceeds
    the memory budget, and then skips larger sizes for that language.

Outputs (in --out, default results/):
  raw/measurements.csv  every warm-up and measured row printed by the implementations
  raw/runs.csv          one row per configuration: status, memory, wall time, ...
  selftest.txt          output of the known-answer self tests
  outputs/<lang>_<n>.bin  result matrix of the last repetition (for validate_outputs.py)

    python scripts/run_benchmarks.py [--config ...] [--sizes 32 64] [--languages c java]
"""
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

RUN_FIELDS = ["timestamp", "language", "n", "status", "exit_code", "warmup_runs",
              "measured_rows", "inner", "wall_time_s", "peak_working_set_bytes",
              "peak_private_bytes", "java_heap_used_bytes", "java_heap_committed_bytes",
              "note"]
MEAS_FIELDS = ["language", "n", "phase", "rep", "inner", "batch_time_s", "time_per_mult_s",
               "checksum"]


def java_executable():
    home = os.environ.get("JAVA_HOME")
    if home and os.path.exists(os.path.join(home, "bin", "java.exe" if os.name == "nt" else "java")):
        return os.path.join(home, "bin", "java")
    return shutil.which("java")


def command(lang, cfg, n, warmup, reps, data_dir, dump):
    common = ["bench", "--n", str(n), "--data", data_dir, "--warmup", str(warmup),
              "--reps", str(reps), "--min-batch-time", str(cfg["min_batch_time_s"]),
              "--dump", dump]
    if lang == "c":
        exe = os.path.join("build", "matmul_c.exe" if sys.platform == "win32" else "matmul_c")
        return [exe, *common]
    if lang == "java":
        return [java_executable(), *cfg["java"]["jvm_flags"], "-cp", os.path.join("build", "java"),
                "MatMul", *common]
    if lang == "python":
        return [sys.executable, *cfg["python"]["flags"], os.path.join("python", "matmul.py"), *common]
    raise ValueError(lang)


def tree_peak_memory(proc):
    """Peak working set and peak private bytes of the process and any children (bytes)."""
    peak_ws = peak_priv = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0, 0
    for p in procs:
        try:
            mi = p.memory_info()
        except psutil.Error:
            continue
        peak_ws += getattr(mi, "peak_wset", mi.rss)
        peak_priv += getattr(mi, "peak_pagefile", getattr(mi, "vms", 0))
    return peak_ws, peak_priv


def run_config(lang, n, cfg, warmup, reps, data_dir, out_dir):
    dump = os.path.join(out_dir, "outputs", f"{lang}_{n}.bin")
    cmd = command(lang, cfg, n, warmup, reps, data_dir, dump)
    limit = cfg["time_limit_per_multiplication_s"]
    budget = cfg["memory_budget_bytes"]
    interval = cfg["memory_sampling_interval_s"]

    start = time.perf_counter()
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc = psutil.Process(popen.pid)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in popen.stdout], daemon=True).start()

    rows, comments = [], []
    peak_ws = peak_priv = 0
    last_progress = time.perf_counter()
    status, note = "ok", ""
    while True:
        ws, priv = tree_peak_memory(proc)
        peak_ws, peak_priv = max(peak_ws, ws), max(peak_priv, priv)
        try:
            while True:
                line = lines.get_nowait().strip()
                last_progress = time.perf_counter()
                if line.startswith("#"):
                    comments.append(line[1:].strip())
                elif line and not line.startswith("language,"):
                    rows.append(line.split(","))
        except queue.Empty:
            pass
        if popen.poll() is not None:
            break
        if peak_ws > budget:
            status, note = "memory_limit", f"peak working set exceeded {budget} bytes"
            popen.kill()
            break
        if time.perf_counter() - last_progress > limit:
            status, note = "timeout", f"no multiplication finished within {limit} s"
            popen.kill()
            break
        time.sleep(interval)
    popen.wait()
    wall = time.perf_counter() - start
    # Drain anything printed just before exit.
    time.sleep(0.2)
    while not lines.empty():
        line = lines.get_nowait().strip()
        if line.startswith("#"):
            comments.append(line[1:].strip())
        elif line and not line.startswith("language,"):
            rows.append(line.split(","))
    stderr = popen.stderr.read()
    if status == "ok" and popen.returncode != 0:
        status, note = "error", stderr.strip().replace("\n", " ")[:300]

    extra = {}
    for c in comments:
        parts = c.split(",")
        extra.update(dict(zip(parts[0::2], parts[1::2])))
    measured = [r for r in rows if r[2] == "measure"]
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": lang, "n": n, "status": status, "exit_code": popen.returncode,
        "warmup_runs": warmup, "measured_rows": len(measured),
        "inner": measured[0][4] if measured else "",
        "wall_time_s": f"{wall:.3f}", "peak_working_set_bytes": peak_ws,
        "peak_private_bytes": peak_priv,
        "java_heap_used_bytes": extra.get("java_heap_used_bytes", ""),
        "java_heap_committed_bytes": extra.get("java_heap_committed_bytes", ""),
        "note": note,
    }
    return run, rows


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--sizes", type=int, nargs="*")
    parser.add_argument("--languages", nargs="*")
    parser.add_argument("--reps", type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    sizes = args.sizes or cfg["sizes"]
    languages = args.languages or cfg["languages"]
    reps = args.reps or cfg["measured_repetitions"]
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "outputs"), exist_ok=True)

    # Known-answer self tests must pass before any benchmark is run.
    with open(os.path.join(args.out, "selftest.txt"), "w") as log:
        for lang in languages:
            cmd = command(lang, cfg, 1, 0, 0, args.data, "")
            base = cmd[:cmd.index("bench")]
            res = subprocess.run([*base, "selftest"], capture_output=True, text=True)
            log.write(f"== {lang} ==\n{res.stdout}\n")
            print(f"[selftest] {lang}: {'PASSED' if res.returncode == 0 else 'FAILED'}")
            if res.returncode != 0:
                sys.exit(f"self test failed for {lang}:\n{res.stdout}{res.stderr}")

    stopped = {}
    meas_path = os.path.join(args.out, "raw", "measurements.csv")
    runs_path = os.path.join(args.out, "raw", "runs.csv")
    for n in sizes:
        for lang in languages:
            if lang in stopped:
                append_csv(runs_path, RUN_FIELDS, [[
                    datetime.now(timezone.utc).isoformat(timespec="seconds"), lang, n,
                    "skipped", "", "", 0, "", "", "", "", "", "",
                    f"not attempted: {stopped[lang]} at a smaller size"]])
                print(f"[skip] {lang} n={n}")
                continue
            warmup = cfg["warmup_runs"][lang]
            print(f"[run ] {lang} n={n} ...", flush=True)
            run, rows = run_config(lang, n, cfg, warmup, reps, args.data, args.out)
            append_csv(meas_path, MEAS_FIELDS, rows)
            append_csv(runs_path, RUN_FIELDS, [[run[k] for k in RUN_FIELDS]])
            med = sorted(float(r[6]) for r in rows if r[2] == "measure")
            med_s = f"median {med[len(med) // 2]:.4g} s" if med else ""
            print(f"       {run['status']} {med_s} peakWS {run['peak_working_set_bytes'] / 1e6:.1f} MB",
                  flush=True)
            if run["status"] != "ok":
                stopped[lang] = run["status"]
            time.sleep(cfg["pause_between_runs_s"])


if __name__ == "__main__":
    main()
