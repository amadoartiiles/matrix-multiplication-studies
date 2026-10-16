"""Record the hardware and software environment of a benchmark run.

    python scripts/collect_env.py [--out results/environment.json]
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

import psutil


def cmd_output(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (res.stdout + res.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def powershell(expr):
    return cmd_output(["powershell", "-NoProfile", "-Command", expr])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/environment.json")
    parser.add_argument("--config", default="config/benchmark_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    java = os.path.join(os.environ["JAVA_HOME"], "bin", "java") if "JAVA_HOME" in os.environ \
        else shutil.which("java")
    env = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "psutil": psutil.__version__,
        "java_version": cmd_output([java, "-version"]),
        "java_executable": java,
        "javac_version": cmd_output(["javac", "-version"]),
        "gcc_version": cmd_output([os.environ.get("CC", cfg["c"]["compiler"]), "--version"]).splitlines()[0],
        "c_flags": cfg["c"]["flags"],
        "jvm_flags": cfg["java"]["jvm_flags"],
    }
    try:
        import numpy
        import matplotlib
        env["numpy"] = numpy.__version__
        env["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    if sys.platform == "win32":
        env["cpu_name"] = powershell("(Get-CimInstance Win32_Processor).Name")
        env["cpu_max_clock_mhz"] = powershell("(Get-CimInstance Win32_Processor).MaxClockSpeed")
        env["l2_cache_kb"] = powershell("(Get-CimInstance Win32_Processor).L2CacheSize")
        env["l3_cache_kb"] = powershell("(Get-CimInstance Win32_Processor).L3CacheSize")
        env["computer_model"] = powershell("(Get-CimInstance Win32_ComputerSystem).Model")
        env["power_plan"] = cmd_output(["powercfg", "/getactivescheme"])
        env["battery_status"] = powershell(
            "(Get-CimInstance Win32_Battery).BatteryStatus  # 2 = on AC power")
        env["jvm_gc_and_heap"] = cmd_output(
            [java, *cfg["java"]["jvm_flags"], "-XX:+PrintCommandLineFlags", "-version"]).splitlines()[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
