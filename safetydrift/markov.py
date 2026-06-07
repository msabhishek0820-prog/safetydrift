"""
safetydrift - Markov chain model

Implements the absorbing Markov chain analysis from SafetyDrift
(Dhodapkar & Pishori, arXiv:2603.27148).

Key insight from the paper: in communication tasks, agents that reach
a mild risk state have an 85% chance of violating safety within 5 steps.
In technical tasks, the probability stays below 5% from any state.

We model safety state as an absorbing Markov chain where:
 - States 0-4 correspond to RiskLevel (SAFE → CRITICAL)
 - State 5 is the absorbing "Violation" state
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from .types import RiskLevel, SafetyState


# ── Task-type priors ─────────────────────────────────────────────────────────
# Calibrated to match the paper's key findings:
#   - communication: 85% violation within 5 steps from mild-risk state
#   - technical: below 5% from any state
#
# (base_step_violation_rate_at_low_risk, escalation_multiplier_per_level)

TASK_PRIORS: dict[str, Tuple[float, float]] = {
    "communication": (0.18, 1.8),  # emails/messaging — fast to violate
    "information":   (0.10, 1.5),  # browsing/summarising
    "technical":     (0.01, 1.2),  # code editing, local ops — slow to violate
    "autonomous":    (0.12, 1.6),  # multi-step tasks
    "default":       (0.08, 1.4),
}


def _build_transition_matrix(
    current_risk: int,
    task_type: str = "default",
    context_pressure: float = 0.0,
) -> np.ndarray:
    """
    Build a 6×6 absorbing Markov chain transition matrix.

    States 0–4 = RiskLevel (SAFE→CRITICAL). State 5 = absorbing Violation.
    context_pressure ∈ [0,1] increases transition rates for long sessions.
    """
    base_rate, accel = TASK_PRIORS.get(task_type, TASK_PRIORS["default"])
    n = 6

    T = np.zeros((n, n))

    for s in range(5):
        # Violation rate grows with risk level and context pressure
        escalation_rate = base_rate * (accel ** s) * (1.0 + context_pressure * 0.5)
        escalation_rate = min(escalation_rate, 0.90)

        # Direct violation probability is higher at CRITICAL (s=4)
        if s == 4:
            violation_rate = min(escalation_rate * 0.8, 0.85)
        else:
            violation_rate = escalation_rate * max(0.05, (s / 8.0))

        # Probability of advancing to next risk level
        advance_rate = escalation_rate - violation_rate
        advance_rate = max(0.0, min(advance_rate, 0.9))

        stay_rate = max(0.0, 1.0 - advance_rate - violation_rate)

        T[s, s]   = stay_rate
        T[s, 5]   = violation_rate  # absorbing violation

        if s < 4:
            T[s, s + 1] = advance_rate * 0.75
            if s < 3:
                T[s, s + 2] = advance_rate * 0.25

        # Normalize row
        row_sum = T[s].sum()
        if row_sum > 0:
            T[s] /= row_sum

    # Absorbing state
    T[5, 5] = 1.0
    return T


def violation_probability(
    state: SafetyState,
    horizon: int = 5,
    task_type: str = "default",
    context_pressure: float = 0.0,
) -> float:
    """
    Compute P(violation within `horizon` steps | current state).

    Uses closed-form absorption analysis: P_absorb(n) = [T^n]_{s,5}
    where s is the current risk level and 5 is the absorbing violation state.
    """
    s = int(state.risk_level)
    T = _build_transition_matrix(s, task_type, context_pressure)
    T_n = np.linalg.matrix_power(T, horizon)
    prob = float(T_n[s, 5])
    return round(min(max(prob, 0.0), 1.0), 4)


def steps_to_threshold(
    state: SafetyState,
    threshold: float = 0.5,
    max_steps: int = 20,
    task_type: str = "default",
    context_pressure: float = 0.0,
) -> int:
    """Number of steps until P(violation) first exceeds `threshold`."""
    for n in range(1, max_steps + 1):
        p = violation_probability(state, horizon=n,
                                  task_type=task_type,
                                  context_pressure=context_pressure)
        if p >= threshold:
            return n
    return max_steps


def context_pressure_from_session(step_count: int, permission_breadth: int = 0) -> float:
    """Estimate context pressure from session length and permission breadth."""
    session_pressure   = math.tanh(step_count / 20.0) * 0.5
    permission_pressure = (permission_breadth / 5.0) * 0.5
    return round(min(session_pressure + permission_pressure, 1.0), 3)
