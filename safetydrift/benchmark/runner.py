"""
Benchmark runner for SafetyDrift.

Runs every trace through a Session, collects predictions,
then computes the metrics that matter to AI safety teams:

  - Precision / Recall / F1
  - Early detection rate (how many steps BEFORE violation)
  - False positive rate
  - Per-pattern breakdown
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from collections import defaultdict

from safetydrift import Session, InterventionAction
from benchmark.trace_generator import Trace


# ── Result for one trace ──────────────────────────────────────────────────────

@dataclass
class TraceResult:
    trace:           Trace
    predicted_flag:  bool          # did SafetyDrift raise PAUSE or BLOCK?
    flagged_at_step: int | None    # which step triggered the flag
    steps_early:     int | None    # how many steps before ground-truth violation

    @property
    def true_positive(self)  -> bool: return self.trace.is_violation and self.predicted_flag
    @property
    def false_positive(self) -> bool: return not self.trace.is_violation and self.predicted_flag
    @property
    def false_negative(self) -> bool: return self.trace.is_violation and not self.predicted_flag
    @property
    def true_negative(self)  -> bool: return not self.trace.is_violation and not self.predicted_flag


# ── Runner ────────────────────────────────────────────────────────────────────

def run_trace(trace: Trace, task_type: str = "default") -> TraceResult:
    """Run one trace through SafetyDrift and return a TraceResult."""
    session = Session(task_type=task_type)
    flagged_at = None

    for i, step in enumerate(trace.steps):
        result = session.gate(step.tool, step.args)
        if result.action in (InterventionAction.PAUSE, InterventionAction.BLOCK):
            flagged_at = i
            break   # stop at first intervention (realistic behaviour)

    predicted = flagged_at is not None
    steps_early = None
    if predicted and trace.violation_step is not None:
        steps_early = trace.violation_step - flagged_at

    return TraceResult(trace, predicted, flagged_at, steps_early)


def run_all(traces: list[Trace], task_type: str = "default") -> list[TraceResult]:
    return [run_trace(t, task_type) for t in traces]


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkMetrics:
    total:          int
    tp: int; fp: int; fn: int; tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    # Early detection
    early_detections: list[int] = field(default_factory=list)

    @property
    def avg_steps_early(self) -> float:
        return sum(self.early_detections) / len(self.early_detections) \
               if self.early_detections else 0.0

    @property
    def caught_before_violation(self) -> int:
        return sum(1 for s in self.early_detections if s > 0)

    # Per-pattern
    per_pattern: dict[str, dict] = field(default_factory=dict)


def compute_metrics(results: list[TraceResult]) -> BenchmarkMetrics:
    tp = sum(1 for r in results if r.true_positive)
    fp = sum(1 for r in results if r.false_positive)
    fn = sum(1 for r in results if r.false_negative)
    tn = sum(1 for r in results if r.true_negative)

    early = [r.steps_early for r in results
             if r.steps_early is not None]

    # Per-pattern breakdown
    by_pattern: dict[str, list[TraceResult]] = defaultdict(list)
    for r in results:
        by_pattern[r.trace.pattern].append(r)

    per_pattern = {}
    for pattern, rs in by_pattern.items():
        p_tp = sum(1 for r in rs if r.true_positive)
        p_fp = sum(1 for r in rs if r.false_positive)
        p_fn = sum(1 for r in rs if r.false_negative)
        p_tn = sum(1 for r in rs if r.true_negative)
        p_prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) else 0.0
        p_rec  = p_tp / (p_tp + p_fn) if (p_tp + p_fn) else 0.0
        per_pattern[pattern] = {
            "total": len(rs),
            "precision": round(p_prec, 2),
            "recall":    round(p_rec, 2),
            "tp": p_tp, "fp": p_fp, "fn": p_fn, "tn": p_tn,
        }

    m = BenchmarkMetrics(total=len(results),
                         tp=tp, fp=fp, fn=fn, tn=tn,
                         early_detections=early,
                         per_pattern=per_pattern)
    return m
