"""Summarise the raw measurements and draw the figures used in the report.

Inputs : results/raw/measurements.csv, results/raw/runs.csv
Outputs: results/summary/summary.csv        median, IQR, ... per (language, n)
         results/summary/scaling_fit.csv    fitted exponent of t = c * n^b
         results/summary/limits.csv         status of every attempted configuration
         results/summary/tables.tex         LaTeX tables included by the report
         results/figures/*.png              figures

    python scripts/analyze.py [--results results]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LANGS = ["c", "java", "python"]
NAME = {"c": "C", "java": "Java", "python": "Python"}
LABEL = {"c": "C (gcc -O2)", "java": "Java (HotSpot JIT)", "python": "Python (CPython)"}
# First three slots of the validated categorical palette (pass all-pairs CVD checks).
COLOR = {"c": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARKER = {"c": "o", "java": "s", "python": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIT_MIN_N = 128


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


LABEL_DY = {"c": 7, "java": -7, "python": 0}  # C and Java end at almost the same point


def direct_label(ax, x, y, lang, offset=None, ha="left"):
    ax.annotate(LABEL[lang].split(" ")[0], (x, y), xytext=offset or (7, LABEL_DY[lang]),
                textcoords="offset points", va="center", ha=ha, fontsize=9.5, color=INK)


def storage_model_bytes(lang, n):
    """Analytical storage of A, B and C (elements and per-row structure only)."""
    if lang == "c":
        return 3 * 8 * n * n                              # contiguous double[]
    if lang == "java":
        return 3 * (n * (16 + 8 * n) + (16 + 4 * n))      # n row arrays + reference array
    return 3 * (n * (56 + 8 * n) + 24 * n * n)        # row lists of pointers + float objects


def median_by(rows, key_fields, value_field):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key_fields), []).append(float(r[value_field]))
    return {k: float(np.median(v)) for k, v in groups.items()}


def supplementary(results, summary, fig_dir, fmt_t):
    """Tables/figure for the replication run and the two supplementary C experiments."""
    lines = []
    main_med = {(s["language"], s["n"]): s["median_s"] for s in summary}

    rep_path = os.path.join(results, "replication", "raw", "measurements.csv")
    if os.path.exists(rep_path):
        rep = [r for r in csv.DictReader(open(rep_path)) if r["phase"] == "measure"]
        rep_med = median_by(rep, ("language", "n"), "time_per_mult_s")
        lines.append("\\newcommand{\\ReplicationTable}{")
        lines.append("\\begin{tabular}{l r r r r}\\toprule Language & $n$ & Main run & Replication "
                     "& Change \\\\\\midrule")
        for (lang, n), med in sorted(rep_med.items(), key=lambda x: (x[0][0], int(x[0][1]))):
            m = main_med.get((lang, int(n)))
            if m:
                lines.append(f"{NAME[lang]} & {n} & {fmt_t(m)} & {fmt_t(med)} & "
                             f"{100 * (med - m) / m:+.0f}\\% \\\\")
        lines.append("\\bottomrule\\end{tabular}}")

    flags_path = os.path.join(results, "supplementary", "c_flags.csv")
    if os.path.exists(flags_path):
        fl = [r for r in csv.DictReader(open(flags_path)) if r["phase"] == "measure"]
        fmed = median_by(fl, ("variant", "n"), "time_per_mult_s")
        variants = list(dict.fromkeys(r["variant"] for r in fl))
        sizes = sorted({int(r["n"]) for r in fl})
        lines.append("\\newcommand{\\FlagsTable}{")
        lines.append("\\begin{tabular}{l " + "r " * len(sizes) + "}\\toprule GCC flags & "
                     + " & ".join(f"$n={n}$" for n in sizes) + " \\\\\\midrule")
        for v in variants:
            lines.append(v.replace("-", "{-}") + " & " + " & ".join(
                fmt_t(fmed[(v, str(n))]) for n in sizes) + " \\\\")
        lines.append("\\bottomrule\\end{tabular}}")

    pad_path = os.path.join(results, "supplementary", "c_padding.csv")
    if os.path.exists(pad_path):
        pr = list(csv.DictReader(open(pad_path)))
        pmed = median_by(pr, ("n", "pad"), "time_per_mult_s")
        pads = sorted({r["pad"] for r in pr}, key=int)
        sizes = sorted({int(r["n"]) for r in pr})
        same = all(len({r["checksum"] for r in pr if r["n"] == str(n)}) == 1 for n in sizes)
        lines.append("\\newcommand{\\PaddingTable}{")
        lines.append("\\begin{tabular}{r r r r r}\\toprule $n$ & C, pad 0 (row stride $8n$ B) & "
                     "C, pad 2 ($8n+16$ B) & Speed-up & Java (main run) \\\\\\midrule")
        for n in sizes:
            a, b = pmed[(str(n), pads[0])], pmed[(str(n), pads[1])]
            j = main_med.get(("java", n))
            lines.append(f"{n} & {fmt_t(a)} & {fmt_t(b)} & {a / b:.1f}$\\times$ & "
                         f"{fmt_t(j) if j else '--'} \\\\")
        lines.append("\\bottomrule\\end{tabular}}")
        lines.append(f"\\newcommand{{\\PaddingChecksumsEqual}}{{{'yes' if same else 'no'}}}")

        fig, ax = plt.subplots(figsize=(5.4, 3.24))
        ns = np.array(sizes, dtype=float)
        for pad, color, marker, lab in ((pads[0], COLOR["c"], "o", "C, contiguous rows (pad 0)"),
                                        (pads[1], "#4a3aa7", "D", "C, rows padded by 16 B (pad 2)")):
            v = np.array([pmed[(str(n), pad)] for n in sizes]) / ns ** 3 * 1e9
            ax.plot(ns, v, color=color, marker=marker, markersize=5, linewidth=2, label=lab)
        jn = [n for n in sizes if ("java", n) in main_med]
        ax.plot(jn, [main_med[("java", n)] / n ** 3 * 1e9 for n in jn], color=COLOR["java"],
                marker=MARKER["java"], markersize=5, linewidth=2, linestyle="--",
                label="Java (main run, for reference)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, "matrix size n (n × n)", "median time / n³ (ns)",
              "Supplementary: effect of row padding on the same C kernel")
        ax.legend(frameon=False, fontsize=8.5)
        fig.savefig(os.path.join(fig_dir, "fig_supp_padding.png"))
        plt.close(fig)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    raw = os.path.join(args.results, "raw")
    sum_dir = os.path.join(args.results, "summary")
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    meas = list(csv.DictReader(open(os.path.join(raw, "measurements.csv"))))
    runs = list(csv.DictReader(open(os.path.join(raw, "runs.csv"))))
    run_by = {(r["language"], int(r["n"])): r for r in runs}

    # ---------- per-configuration summary ----------
    summary = []
    for lang in LANGS:
        sizes = sorted({int(r["n"]) for r in meas if r["language"] == lang})
        for n in sizes:
            t = np.array([float(r["time_per_mult_s"]) for r in meas
                          if r["language"] == lang and int(r["n"]) == n and r["phase"] == "measure"])
            if len(t) == 0 or run_by[(lang, n)]["status"] != "ok":
                continue
            q1, med, q3 = np.percentile(t, [25, 50, 75])
            run = run_by[(lang, n)]
            summary.append({
                "language": lang, "n": n, "reps": len(t), "inner": int(run["inner"]),
                "warmup_runs": int(run["warmup_runs"]),
                "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                "rel_iqr_pct": 100 * (q3 - q1) / med, "min_s": t.min(), "max_s": t.max(),
                "ns_per_inner_iteration": med / n ** 3 * 1e9,
                "gflops": 2 * n ** 3 / med / 1e9,
                "peak_working_set_mb": int(run["peak_working_set_bytes"]) / 1e6,
                "peak_private_mb": int(run["peak_private_bytes"]) / 1e6,
                "java_heap_used_mb": (int(run["java_heap_used_bytes"]) / 1e6
                                      if run["java_heap_used_bytes"] else ""),
                "storage_model_mb": storage_model_bytes(lang, n) / 1e6,
            })
    with open(os.path.join(sum_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})

    def series(lang, key):
        rows = [s for s in summary if s["language"] == lang]
        return np.array([s["n"] for s in rows]), np.array([s[key] for s in rows], dtype=float)

    # ---------- scaling fit: log t = log c + b log n ----------
    fits = []
    for lang in LANGS:
        n, med = series(lang, "median_s")
        mask = n >= FIT_MIN_N
        if mask.sum() >= 2:
            b, loga = np.polyfit(np.log(n[mask]), np.log(med[mask]), 1)
            fits.append({"language": lang, "n_min": int(n[mask].min()), "n_max": int(n[mask].max()),
                         "points": int(mask.sum()), "exponent_b": round(b, 3)})
    with open(os.path.join(sum_dir, "scaling_fit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
        w.writeheader()
        w.writerows(fits)

    # ---------- limits ----------
    with open(os.path.join(sum_dir, "limits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "largest_ok_n", "largest_attempted_n", "failures"])
        for lang in LANGS:
            mine = [r for r in runs if r["language"] == lang]
            ok = [int(r["n"]) for r in mine if r["status"] == "ok"]
            att = [int(r["n"]) for r in mine if r["status"] != "skipped"]
            fail = "; ".join(f"n={r['n']}: {r['status']} ({r['note']})" for r in mine
                             if r["status"] not in ("ok",))
            w.writerow([lang, max(ok) if ok else "", max(att) if att else "", fail or "none"])

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

    # 1. Median time vs n (log-log) with IQR bars and an n^3 guide.
    fig, ax = plt.subplots(figsize=(5.4, 3.78))
    for lang in LANGS:
        n, med = series(lang, "median_s")
        _, q1 = series(lang, "q1_s")
        _, q3 = series(lang, "q3_s")
        ax.errorbar(n, med, yerr=[med - q1, q3 - med], color=COLOR[lang], marker=MARKER[lang],
                    markersize=5, linewidth=2, capsize=3, label=LABEL[lang])
        direct_label(ax, n[-1], med[-1], lang)
    n_c, med_c = series("c", "median_s")
    ref_n = np.array([n_c.min(), n_c.max()], dtype=float)
    anchor = med_c[list(n_c).index(256)] if 256 in n_c else med_c[0]
    ax.plot(ref_n, anchor * (ref_n / 256) ** 3, linestyle="--", color=INK2, linewidth=1,
            label=r"$\propto n^3$ guide (anchored at C, n=256)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "time per multiplication (s, median)",
          "Runtime vs matrix size (error bars: IQR, 5 repetitions)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_time_vs_n.png"))
    plt.close(fig)

    # 2. Time per inner iteration (t / n^3).
    fig, ax = plt.subplots(figsize=(5.4, 3.42))
    for lang in LANGS:
        n, v = series(lang, "ns_per_inner_iteration")
        ax.plot(n, v, color=COLOR[lang], marker=MARKER[lang], markersize=5, linewidth=2,
                label=LABEL[lang])
        direct_label(ax, n[-1], v[-1], lang)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "median time / n³ (ns)",
          "Cost per multiply-add (flat line = pure cubic growth)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_time_per_n3.png"))
    plt.close(fig)

    # 3. Peak working set vs n with analytical storage of A, B, C.
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for lang in LANGS:
        n, ws = series(lang, "peak_working_set_mb")
        _, model = series(lang, "storage_model_mb")
        ax.plot(n, ws, color=COLOR[lang], marker=MARKER[lang], markersize=5, linewidth=2,
                label=f"{LABEL[lang]} – measured peak working set")
        ax.plot(n, model, color=COLOR[lang], linestyle=":", linewidth=1.5,
                label=f"{LABEL[lang].split(' ')[0]} – storage model (A, B, C)")
        if lang == "python":
            direct_label(ax, n[-1], ws[-1], lang, offset=(-4, 7), ha="right")
        else:
            direct_label(ax, n[-1], ws[-1], lang)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "memory (MB, 10⁶ bytes)",
          "Peak process memory vs matrix size")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(fig_dir, "fig_memory_vs_n.png"))
    plt.close(fig)

    # 4. Slowdown relative to C at each common size.
    fig, ax = plt.subplots(figsize=(5.4, 3.24))
    c_med = dict(zip(*series("c", "median_s")))
    for lang in ("java", "python"):
        n, med = series(lang, "median_s")
        common = [i for i, x in enumerate(n) if x in c_med]
        ratio = np.array([med[i] / c_med[n[i]] for i in common])
        ax.plot(n[common], ratio, color=COLOR[lang], marker=MARKER[lang], markersize=5,
                linewidth=2, label=f"{LABEL[lang]} / C")
        direct_label(ax, n[common][-1], ratio[-1], lang)
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, "matrix size n (n × n)", "median time ratio vs C (×)",
          "Relative runtime (values > 1: slower than C)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig_ratio_vs_c.png"))
    plt.close(fig)

    # ---------- LaTeX tables ----------
    def fmt_t(x):
        if x >= 1:
            return f"{x:.2f} s"
        if x >= 1e-3:
            return f"{x * 1e3:.2f} ms"
        return f"{x * 1e6:.1f} $\\mu$s"

    lines = []
    lines.append("% Generated by scripts/analyze.py -- do not edit by hand.")
    lines.append("\\newcommand{\\TimeTable}{")
    lines.append("\\begin{tabular}{r l r r r r r}\\toprule")
    lines.append("$n$ & Language & Median & IQR (\\%) & ns/iter & GFLOP/s & Batch \\\\\\midrule")
    for n in sorted({s["n"] for s in summary}):
        for s in [x for x in summary if x["n"] == n]:
            lines.append(f"{n} & {NAME[s['language']]} & {fmt_t(s['median_s'])} & "
                         f"{s['rel_iqr_pct']:.1f} & {s['ns_per_inner_iteration']:.2f} & "
                         f"{s['gflops']:.3f} & {s['inner']} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines.append("\\bottomrule\\end{tabular}}")

    lines.append("\\newcommand{\\MemoryTable}{")
    lines.append("\\begin{tabular}{r r r r r r r}\\toprule")
    lines.append("& \\multicolumn{3}{c}{Peak working set (MB)} & \\multicolumn{3}{c}{Storage model (MB)}"
                 " \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    lines.append("$n$ & C & Java & Python & C & Java & Python \\\\\\midrule")
    for n in sorted({s["n"] for s in summary}):
        cells_ws, cells_model = [], []
        for lang in LANGS:
            s = next((x for x in summary if x["n"] == n and x["language"] == lang), None)
            cells_ws.append(f"{s['peak_working_set_mb']:.1f}" if s else "--")
            cells_model.append(f"{s['storage_model_mb']:.1f}" if s else "--")
        lines.append(f"{n} & " + " & ".join(cells_ws + cells_model) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}}")

    lines.append("\\newcommand{\\FitTable}{")
    lines.append("\\begin{tabular}{l r r}\\toprule Language & Sizes used & Fitted exponent $b$ "
                 "\\\\\\midrule")
    for fdesc in fits:
        lines.append(f"{NAME[fdesc['language']]} & {fdesc['n_min']}--{fdesc['n_max']} "
                     f"({fdesc['points']} sizes) & {fdesc['exponent_b']:.2f} \\\\")
    lines.append("\\bottomrule\\end{tabular}}")
    lines += supplementary(args.results, summary, fig_dir, fmt_t)
    with open(os.path.join(sum_dir, "tables.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"summary rows: {len(summary)}")
    for fdesc in fits:
        print(f"fit {fdesc['language']}: b = {fdesc['exponent_b']} on n in "
              f"[{fdesc['n_min']}, {fdesc['n_max']}]")


if __name__ == "__main__":
    main()
