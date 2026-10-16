"""Compile the C and Java implementations into build/ using the configured flags.

    python scripts/build.py [--config config/benchmark_config.json]

The C compiler is taken from the CC environment variable, or "gcc" on PATH.
"""
import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs("build/java", exist_ok=True)
    cc = os.environ.get("CC", cfg["c"]["compiler"])
    exe = "build/matmul_c.exe" if sys.platform == "win32" else "build/matmul_c"
    run([cc, *cfg["c"]["flags"], "-o", exe, "c/matmul.c", "-lm"])
    run(["javac", "-d", "build/java", "java/src/MatMul.java"])


if __name__ == "__main__":
    main()
