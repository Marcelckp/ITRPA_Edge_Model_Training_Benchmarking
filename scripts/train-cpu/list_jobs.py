#!/usr/bin/env python3
"""Helpers for run_plan.sh (stdlib, Python >= 3.6).

    list_jobs.py JOBS.json        print "model strategy dataset lr" per job, PubMedQA jobs first
    list_jobs.py --same A B       exit 0 when learning rates A and B are equal
    list_jobs.py --output-dir     read a `train --dry-run` plan on stdin and print its output_dir
"""
import json
import sys


def main(argv):
    if argv[:1] == ["--same"]:
        return 0 if abs(float(argv[1]) - float(argv[2])) <= 1e-12 else 1
    if argv[:1] == ["--output-dir"]:
        print(json.load(sys.stdin)["output_dir"])
        return 0
    jobs = sorted(json.load(open(argv[0])), key=lambda j: j["dataset"] != "pubmedqa")
    for j in jobs:
        print(j["model"], j["strategy"], j["dataset"], repr(j["learning_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
