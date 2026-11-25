#!/usr/bin/env python3
"""Project CPU training-plan durations from the feasibility benchmark (planning only; stdlib, Python >= 3.6).

    python3 scripts/train-cpu/plan_estimate.py ../pilot-logs/train-cpu-bench.jsonl

Per run = n_train x mean_tokens / tokens_per_s(24) + n_val x val_s_per_item(4) / 6**alpha + overhead, with
tokens_per_s(24) = tokens_per_s(4) x 6**alpha and alpha from the 2 -> 4 thread run. QLoRA runs use their own benchmark
row when there is one with >= 6 timed steps, else the LoRA time x (LoRA / QLoRA tokens/s of the SmolLM2 PubMedQA pair).
"""
import json
import math
import sys

N_TRAIN = {"pubmedqa": 10000, "finqa": 6251}
N_VAL = {"pubmedqa": 500, "finqa": 883}
MODELS = ("smollm2-360m", "qwen2.5-0.5b")
OVERHEAD_S = 120.0  # container start, model load, save; tokenisation is cached after the first run per model/dataset
TOKENISE_S_PER_EX = {"pubmedqa": 0.004, "finqa": 0.006}  # bench token-stats runs: 35-40 s per full set
# Adapted-model evaluation per trained strategy, from the timing-pilot means in results/pilot-estimate.db:
# accuracy of record 1.1 h + edge 0.85 h per strategy (6 configs per plan x 2 models, 20 s server start each).
EVAL_H_PER_STRATEGY = 1.3
EXPORT_H_PER_MODEL = 0.03


def load(path):
    bench, tokens = {}, {}
    for line in open(path):
        d = json.loads(line)
        if d.get("token_stats"):
            tokens[(d["s"]["model"], d["s"]["dataset"])] = d["s"]
        elif d.get("tokens_per_s"):
            bench[(d["model"], d["strategy"], d["dataset"], d.get("threads"))] = d
    return bench, tokens


def alpha(bench):
    t2 = bench.get(("qwen2.5-0.5b", "lora", "finqa", 2))
    t4 = bench.get(("qwen2.5-0.5b", "lora", "finqa", 4))
    if not (t2 and t4):
        return None
    return math.log(t4["tokens_per_s"] / t2["tokens_per_s"]) / math.log(2)


def run_hours(bench, tokens, a, m, s, d, n):
    direct = bench.get((m, s, d, 4)) if s == "qlora" else None
    direct = direct if direct and direct.get("timed_steps", 0) >= 6 else None
    key = "lora" if s == "qlora" and not direct else s
    b = direct or bench[(m, key, d, 4)]
    tok = tokens.get((m, d))
    mean = tok["mean_tokens"] if tok else b["tokens"] / (b["optimizer_steps"] * 4)
    tps = b["tokens_per_s"] * 6 ** a
    if s == "qlora" and not direct:
        q = bench[("smollm2-360m", "qlora", "pubmedqa", 4)]
        tps *= q["tokens_per_s"] / bench[("smollm2-360m", "lora", "pubmedqa", 4)]["tokens_per_s"]
    val = N_VAL[d] * (b.get("val_s_per_item") or 0) / 6 ** a
    return (n * mean / tps + val + OVERHEAD_S) / 3600, mean, tps


def scenario(bench, tokens, a, strategies, finqa_cap=None, finqa_lrs=2, pubmed_lrs=2):
    train = 0.0
    for m in MODELS:
        for d in N_TRAIN:
            n = min(finqa_cap, N_TRAIN[d]) if d == "finqa" and finqa_cap else N_TRAIN[d]
            train += n * TOKENISE_S_PER_EX[d] / 3600
            for s in strategies:
                h, _, _ = run_hours(bench, tokens, a, m, s, d, n)
                train += h * (finqa_lrs if d == "finqa" else pubmed_lrs)
    export = EXPORT_H_PER_MODEL * len(MODELS) * len(N_TRAIN) * len(strategies)
    return train, export, EVAL_H_PER_STRATEGY * len(strategies)


def main(path):
    bench, tokens = load(path)
    a_meas = alpha(bench)
    print("alpha (2 -> 4 threads, Qwen LoRA FinQA): %s" % a_meas)
    for a in sorted({round(x, 3) for x in (a_meas, 0.8, 1.0) if x}):
        print("\nalpha = %.2f (x%.2f from 4 to 24 cores)" % (a, 6 ** a))
        for m in MODELS:
            for d in N_TRAIN:
                for s in ("fft", "lora", "qlora"):
                    h, mean, tps = run_hours(bench, tokens, a, m, s, d, N_TRAIN[d])
                    print("  %-13s %-5s %-8s mean_tokens=%5.0f tok/s@24=%5.0f run=%.2f h" % (m, s, d, mean, tps, h))
        rows = [("full DESIGN 3.3 (fft, lora, qlora; 2 LRs)", ("fft", "lora", "qlora"), {}),
                ("fft + lora; 2 LRs (QLoRA dropped)", ("fft", "lora"), {}),
                ("R1: all 3; FinQA at PubMedQA-selected LR", ("fft", "lora", "qlora"), {"finqa_lrs": 1}),
                ("R2: R1 + FinQA cap 3000", ("fft", "lora", "qlora"), {"finqa_lrs": 1, "finqa_cap": 3000}),
                ("R3: R1 + FinQA cap 2000", ("fft", "lora", "qlora"), {"finqa_lrs": 1, "finqa_cap": 2000}),
                ("R4: fft + lora; FinQA at PubMedQA LR", ("fft", "lora"), {"finqa_lrs": 1})]
        for label, strat, kw in rows:
            t, e, v = scenario(bench, tokens, a, strat, **kw)
            print("  %-44s train %5.1f h + export %.1f h + eval %.1f h = %5.1f h" % (label, t, e, v, t + e + v))


if __name__ == "__main__":
    main(sys.argv[1])
