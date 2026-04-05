#!/usr/bin/env python3

import subprocess
import os
import re
import sys
import json
import platform
import shlex

element_counts = ["1000000", "10000000", "20000000", "40000000"]

perf_points = 1.25

# Set up logs directories
os.makedirs("logs/test", exist_ok=True)
os.makedirs("logs/ref", exist_ok=True)
subprocess.run("rm -rf logs/*", shell=True)
os.makedirs("logs/test", exist_ok=True)
os.makedirs("logs/ref", exist_ok=True)
if os.environ.get("GRADING_TOKEN"):
    subprocess.run("chown -R nobody:nogroup logs", shell=True)

# Command-line argument check
if len(sys.argv) != 2 or sys.argv[1] not in ["find_repeats", "scan"]:
    print("Usage: python3 checker.py <test>: test = scan, find_repeats")
    sys.exit(1)
else:
    test = sys.argv[1]
    print("Test: %s" % test)

print("\n--------------")
print("Running tests:")
print("--------------")


def run_shell(command, capture_output=False, as_nobody=False):
    kwargs = {"shell": True}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    if as_nobody:
        command = "env -i su -s /bin/sh nobody -c %s" % shlex.quote(command)

    return subprocess.run(command, **kwargs)


def check_correctness(test, element_count):
    correctness_cmd = (
        "./cudaScan -m %s -i random -n %s > ./logs/test/%s_correctness_%s.log"
        % (test, element_count, test, element_count)
    )
    if os.environ.get("GRADING_TOKEN"):
        result = run_shell(correctness_cmd, as_nobody=True)
    else:
        result = run_shell(correctness_cmd)
    return result.returncode == 0


def get_time(command):
    if os.environ.get("GRADING_TOKEN"):
        result = run_shell(command, capture_output=True, as_nobody=True)
    else:
        result = run_shell(command, capture_output=True)
    time_match = re.search(r"\d+(\.\d+)?", result.stdout.decode())
    return float(time_match.group()) if time_match else None


def run_tests():
    correct = {}
    your_times = {}
    fast_times = {}

    for element_count in element_counts:
        print("\nElement Count: %s" % element_count)

        # Correctness check
        correct[element_count] = check_correctness(test, element_count)
        if correct[element_count]:
            print("Correctness passed!")
        else:
            print("Correctness failed")

        # Get student time
        student_cmd = (
            "./cudaScan -m %s -i random -n %s | tee ./logs/test/%s_time_%s.log "
            "| grep 'Student GPU time:'"
            % (test, element_count, test, element_count)
        )
        your_times[element_count] = get_time(student_cmd)
        print("Student Time: %s" % your_times[element_count])

        ref_binary = (
            "cudaScan_ref_x86" if platform.machine() == "x86_64" else "cudaScan_ref"
        )

        # Get reference time
        ref_cmd = (
            "./%s -m %s -i random -n %s | tee ./logs/ref/%s_time_%s.log "
            "| grep 'Student GPU time:'"
            % (ref_binary, test, element_count, test, element_count)
        )
        fast_times[element_count] = get_time(ref_cmd)
        print("Ref Time: %s" % fast_times[element_count])

    return correct, your_times, fast_times


def calculate_scores(correct, your_times, fast_times):
    scores = []
    total_score = 0

    for element_count in element_counts:
        ref_time = fast_times[element_count]
        stu_time = your_times[element_count]

        if correct[element_count]:
            if stu_time <= 1.20 * ref_time:
                score = perf_points
            else:
                score = perf_points * (ref_time / stu_time)
        else:
            score = 0

        scores.append(
            {
                "element_count": element_count,
                "correct": correct[element_count],
                "ref_time": ref_time,
                "stu_time": stu_time,
                "score": score,
            }
        )
        total_score += score

    max_total_score = perf_points * len(element_counts)
    return scores, total_score, max_total_score


def print_score_table(scores, total_score, max_total_score):
    print("\n-------------------------")
    print("%s Score Table:" % test.capitalize())
    print("-------------------------")

    header = "| %-15s | %-15s | %-15s | %-15s |" % (
        "Element Count",
        "Ref Time",
        "Student Time",
        "Score",
    )
    dashes = "-" * len(header)
    print(dashes)
    print(header)
    print(dashes)

    for score in scores:
        element_count = score["element_count"]
        ref_time = score["ref_time"]
        stu_time = score["stu_time"]
        score_value = score["score"]

        if not score["correct"]:
            stu_time = "%s (F)" % stu_time

        print(
            "| %-15s | %-15s | %-15s | %-15s |"
            % (element_count, ref_time, stu_time, score_value)
        )

    print(dashes)
    print(
        "| %-33s | %-15s | %-15s |"
        % ("", "Total score:", "%s/%s" % (total_score, max_total_score))
    )
    print(dashes)


# Run tests and calculate scores
correct, your_times, fast_times = run_tests()
scores, total_score, max_total_score = calculate_scores(correct, your_times, fast_times)

# Output based on mode
GRADING_TOKEN = os.environ.get("GRADING_TOKEN")
if not GRADING_TOKEN:
    print_score_table(scores, total_score, max_total_score)
else:
    scores = json.dumps(scores)
    print("%s%s" % (GRADING_TOKEN, scores))
