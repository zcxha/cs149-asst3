#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

SCENE_NAMES = [
    "rgb",
    "rand10k",
    "rand100k",
    "pattern",
    "snowsingle",
    "biglittle",
    "rand1M",
    "micro2M",
]

DEFAULT_RUNS = 3
LOG_DIR = "logs_selfcheck"
OUTPUT_PREFIX = os.path.join(LOG_DIR, "output")
CSV_PATH = os.path.join(LOG_DIR, "summary.csv")


def ensure_clean_log_dir(log_dir):
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)


def correctness_log_file(scene):
    return os.path.join(LOG_DIR, "correctness_%s.log" % scene)


def time_log_file(scene, run_idx):
    return os.path.join(LOG_DIR, "time_%s_run%d.log" % (scene, run_idx))


def run_shell(command, capture_output=False):
    kwargs = {"shell": True}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        # Python 3.6 compatibility
        kwargs["universal_newlines"] = True
    return subprocess.run(command, **kwargs)


def safe_write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def build_correctness_command(render_cmd, scene, output_prefix):
    seed = random.randint(0, 100000)
    return "./%s -c %s -s 1024 -S %d -f %s" % (
        render_cmd,
        scene,
        seed,
        shlex.quote(output_prefix),
    )


def build_time_command(render_cmd, scene, output_prefix):
    return "./%s -r cuda -b 0:4 %s -s 1024 -f %s" % (
        render_cmd,
        scene,
        shlex.quote(output_prefix),
    )


def check_correctness(render_cmd, scene):
    command = build_correctness_command(render_cmd, scene, OUTPUT_PREFIX)
    start = time.time()
    result = run_shell(command, capture_output=True)
    end = time.time()

    log_text = []
    log_text.append("$ %s\n" % command)
    log_text.append("returncode: %s\n" % result.returncode)
    log_text.append("wall_time_sec: %.6f\n\n" % (end - start))
    log_text.append("[stdout]\n%s\n\n" % (result.stdout or ""))
    log_text.append("[stderr]\n%s\n" % (result.stderr or ""))
    safe_write_text(correctness_log_file(scene), "".join(log_text))

    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "wall_time_sec": end - start,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "command": command,
    }


def parse_total_time(text):
    if not text:
        return None

    patterns = [
        r"Total:\s*([0-9]+(?:\.[0-9]+)?)",
        r"total:\s*([0-9]+(?:\.[0-9]+)?)",
        r"TOTAL:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # fallback: extract the first float on a line containing "Total"
    for line in text.splitlines():
        if "Total" in line or "total" in line or "TOTAL" in line:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass

    return None


def measure_time(render_cmd, scene, run_idx):
    command = build_time_command(render_cmd, scene, OUTPUT_PREFIX)
    start = time.time()
    result = run_shell(command, capture_output=True)
    end = time.time()

    combined = ""
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        if combined:
            combined += "\n"
        combined += result.stderr

    parsed_total = parse_total_time(combined)

    log_text = []
    log_text.append("$ %s\n" % command)
    log_text.append("returncode: %s\n" % result.returncode)
    log_text.append("wall_time_sec: %.6f\n" % (end - start))
    log_text.append("parsed_total: %s\n\n" % ("None" if parsed_total is None else parsed_total))
    log_text.append("[stdout]\n%s\n\n" % (result.stdout or ""))
    log_text.append("[stderr]\n%s\n" % (result.stderr or ""))
    safe_write_text(time_log_file(scene, run_idx), "".join(log_text))

    return {
        "ok": result.returncode == 0 and parsed_total is not None,
        "returncode": result.returncode,
        "wall_time_sec": end - start,
        "parsed_total": parsed_total,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "command": command,
    }


def fmt_float(x, digits=3):
    if x is None:
        return "-"
    fmt = "%%.%df" % digits
    return fmt % x


def summarize_scene(scene, correctness_result, timing_results):
    parsed_times = [r["parsed_total"] for r in timing_results if r["parsed_total"] is not None]
    wall_times = [r["wall_time_sec"] for r in timing_results]

    timing_successes = 0
    for r in timing_results:
        if r["ok"]:
            timing_successes += 1

    return {
        "scene": scene,
        "correct": correctness_result["ok"],
        "correct_returncode": correctness_result["returncode"],
        "correct_wall_time_sec": correctness_result["wall_time_sec"],
        "timing_runs": len(timing_results),
        "timing_successes": timing_successes,
        "parsed_min": min(parsed_times) if parsed_times else None,
        "parsed_avg": (sum(parsed_times) / float(len(parsed_times))) if parsed_times else None,
        "parsed_max": max(parsed_times) if parsed_times else None,
        "wall_avg": (sum(wall_times) / float(len(wall_times))) if wall_times else None,
        "raw_parsed_times": parsed_times,
    }


def print_table(rows):
    headers = [
        "Scene",
        "Correct",
        "RC",
        "Chk Wall(s)",
        "Timing OK",
        "Min Total",
        "Avg Total",
        "Max Total",
        "Avg Wall(s)",
    ]

    body = []
    for row in rows:
        body.append([
            row["scene"],
            "PASS" if row["correct"] else "FAIL",
            str(row["correct_returncode"]),
            fmt_float(row["correct_wall_time_sec"], 3),
            "%d/%d" % (row["timing_successes"], row["timing_runs"]),
            fmt_float(row["parsed_min"], 3),
            fmt_float(row["parsed_avg"], 3),
            fmt_float(row["parsed_max"], 3),
            fmt_float(row["wall_avg"], 3),
        ])

    widths = []
    for i, header in enumerate(headers):
        max_len = len(header)
        for row in body:
            max_len = max(max_len, len(row[i]))
        widths.append(max_len)

    def sep():
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def render_row(values):
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    print(sep())
    print(render_row(headers))
    print(sep())
    for row in body:
        print(render_row(row))
    print(sep())


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene",
            "correct",
            "correct_returncode",
            "correct_wall_time_sec",
            "timing_runs",
            "timing_successes",
            "parsed_min_total",
            "parsed_avg_total",
            "parsed_max_total",
            "avg_wall_time_sec",
            "raw_parsed_times",
        ])
        for row in rows:
            writer.writerow([
                row["scene"],
                int(bool(row["correct"])),
                row["correct_returncode"],
                row["correct_wall_time_sec"],
                row["timing_runs"],
                row["timing_successes"],
                row["parsed_min"],
                row["parsed_avg"],
                row["parsed_max"],
                row["wall_avg"],
                " ".join(str(x) for x in row["raw_parsed_times"]),
            ])


def run_all(render_binary, runs_per_scene):
    rows = []
    for scene in SCENE_NAMES:
        print("\nRunning scene: %s" % scene)

        correctness_result = check_correctness(render_binary, scene)
        if correctness_result["ok"]:
            print("  correctness: PASS")
        else:
            print("  correctness: FAIL (rc=%s)" % correctness_result["returncode"])

        timing_results = []
        for run_idx in range(1, runs_per_scene + 1):
            result = measure_time(render_binary, scene, run_idx)
            timing_results.append(result)
            if result["ok"]:
                print("  timing run %d: parsed Total=%s" % (run_idx, fmt_float(result["parsed_total"], 3)))
            else:
                if result["returncode"] != 0:
                    print("  timing run %d: FAIL (rc=%s, no usable Total)" % (run_idx, result["returncode"]))
                else:
                    print("  timing run %d: FAIL (could not parse Total)" % run_idx)

        rows.append(summarize_scene(scene, correctness_result, timing_results))

    return rows


def print_summary(rows):
    total_scenes = len(rows)
    correct_scenes = sum(1 for r in rows if r["correct"])
    fully_timed = sum(1 for r in rows if r["timing_successes"] == r["timing_runs"])

    print("\nSummary")
    print("=======")
    print("Python: %s" % sys.version.replace("\n", " "))
    print("Platform: %s / %s" % (platform.system(), platform.machine()))
    print("Scenes passed correctness: %d/%d" % (correct_scenes, total_scenes))
    print("Scenes with all timing runs parsed: %d/%d" % (fully_timed, total_scenes))
    print_table(rows)
    print("CSV written to: %s" % CSV_PATH)
    print("Logs directory: %s" % LOG_DIR)


def parse_args(argv):
    render_binary = "render"
    runs_per_scene = DEFAULT_RUNS

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-r", "--render"):
            if i + 1 >= len(argv):
                raise SystemExit("missing value after %s" % arg)
            render_binary = argv[i + 1]
            i += 2
        elif arg in ("-n", "--runs"):
            if i + 1 >= len(argv):
                raise SystemExit("missing value after %s" % arg)
            runs_per_scene = int(argv[i + 1])
            if runs_per_scene <= 0:
                raise SystemExit("runs must be > 0")
            i += 2
        elif arg in ("-h", "--help"):
            print("Usage: python3 render_selfcheck.py [-r render_binary] [-n runs_per_scene]")
            raise SystemExit(0)
        else:
            raise SystemExit("unknown argument: %s" % arg)

    return render_binary, runs_per_scene


def main(argv):
    render_binary, runs_per_scene = parse_args(argv)

    if not os.path.isfile(render_binary):
        if not os.path.isfile("./%s" % render_binary):
            raise SystemExit("render binary not found: %s" % render_binary)

    ensure_clean_log_dir(LOG_DIR)
    rows = run_all(render_binary, runs_per_scene)
    write_csv(rows, CSV_PATH)
    print_summary(rows)


if __name__ == "__main__":
    main(sys.argv)
