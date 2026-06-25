"""
SafetyDrift Benchmark
Run: python -m benchmark.run

Outputs a human-readable report + benchmark.json for sharing with companies.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from benchmark.trace_generator import generate
from benchmark.runner import run_all, compute_metrics


def print_report(m) -> None:
    W = 58
    def bar(val, width=20):
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    print("\n" + "═" * W)
    print("  SafetyDrift — Benchmark Report")
    print("═" * W)
    print(f"  Total traces evaluated : {m.total}")
    print(f"  Violations in dataset  : {m.tp + m.fn}")
    print(f"  Benign in dataset      : {m.fp + m.tn}")
    print()

    print("  Core metrics")
    print("  " + "─" * (W - 2))
    print(f"  Precision  {bar(m.precision)}  {m.precision:.1%}")
    print(f"  Recall     {bar(m.recall)}    {m.recall:.1%}")
    print(f"  F1 Score   {bar(m.f1)}        {m.f1:.1%}")
    print(f"  Fls.Pos.   {bar(m.fpr)}       {m.fpr:.1%}")
    print()

    print("  Early detection")
    print("  " + "─" * (W - 2))
    print(f"  Caught BEFORE violation  : {m.caught_before_violation} / {m.tp}")
    if m.avg_steps_early > 0:
        print(f"  Avg steps ahead of harm  : {m.avg_steps_early:.1f} steps early")
    print()

    print("  Per-pattern breakdown")
    print("  " + "─" * (W - 2))
    print(f"  {'Pattern':<24} {'Prec':>6}  {'Rec':>6}  {'TP':>3}  {'FN':>3}")
    print("  " + "─" * (W - 2))
    for pat, d in sorted(m.per_pattern.items()):
        print(f"  {pat:<24} {d['precision']:>5.0%}   {d['recall']:>5.0%}   "
              f"{d['tp']:>3}   {d['fn']:>3}")
    print()
    print("  Confusion matrix")
    print("  " + "─" * (W - 2))
    print(f"  TP {m.tp:>4}  |  FP {m.fp:>4}")
    print(f"  FN {m.fn:>4}  |  TN {m.tn:>4}")
    print("═" * W + "\n")


def main():
    print("Generating 200 synthetic traces (100 violation / 100 benign)...")
    traces = generate(n_violation=100, n_benign=100, seed=42)

    print("Running SafetyDrift on all traces...")
    results = run_all(traces, task_type="default")

    m = compute_metrics(results)
    print_report(m)

    # Save JSON report for sharing
    report = {
        "total":                  m.total,
        "precision":              round(m.precision, 4),
        "recall":                 round(m.recall, 4),
        "f1":                     round(m.f1, 4),
        "false_positive_rate":    round(m.fpr, 4),
        "caught_before_violation": m.caught_before_violation,
        "avg_steps_early":        round(m.avg_steps_early, 2),
        "confusion_matrix":       {"tp": m.tp, "fp": m.fp,
                                   "fn": m.fn, "tn": m.tn},
        "per_pattern":            m.per_pattern,
    }

    out = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved → benchmark/benchmark_results.json\n")
    return report


if __name__ == "__main__":
    main()
