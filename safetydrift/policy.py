"""
safetydrift - Policy engine

Evaluates a tool call against the current safety state and returns
a DriftAssessment with a recommended InterventionAction.

Policies are configurable per deployment. The default mirrors the
thresholds implied by SafetyDrift's results:
  - P > 0.85 in communication tasks → BLOCK
  - P > 0.50 → PAUSE (human approval)
  - P > 0.25 → WARN
  - Otherwise  → LOG_ONLY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .classifier import classify
from .markov import (
    context_pressure_from_session,
    violation_probability,
    steps_to_threshold,
)
from .types import (
    DataExposure,
    DriftAssessment,
    InterventionAction,
    RiskLevel,
    SafetyState,
    ToolCall,
    ToolEscalation,
)


@dataclass
class PolicyThreshold:
    """A single policy rule: if P(violation) ≥ prob_threshold, apply action."""
    prob_threshold: float
    action:         InterventionAction
    reason_template: str


# Default thresholds (tunable)
DEFAULT_THRESHOLDS: list[PolicyThreshold] = [
    PolicyThreshold(0.85, InterventionAction.BLOCK,
                    "P(violation)={prob:.0%} exceeds critical threshold — tool call blocked"),
    PolicyThreshold(0.50, InterventionAction.PAUSE,
                    "P(violation)={prob:.0%} exceeds 50% — pausing for human approval"),
    PolicyThreshold(0.25, InterventionAction.WARN,
                    "P(violation)={prob:.0%} elevated — proceeding with warning"),
    PolicyThreshold(0.00, InterventionAction.LOG_ONLY,
                    "P(violation)={prob:.0%} within safe bounds — logged"),
]

# Hard-override: always block these regardless of probability
_ALWAYS_BLOCK_PATTERNS: set[str] = {
    "delete_production_database",
    "wipe_all_data",
    "send_mass_email",
}


@dataclass
class PolicyConfig:
    """Configuration for a safetydrift deployment."""
    thresholds:       list[PolicyThreshold]  = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    horizon:          int                    = 5      # steps ahead to evaluate
    task_type:        str                    = "default"
    always_block:     set[str]               = field(default_factory=lambda: set(_ALWAYS_BLOCK_PATTERNS))
    # Optional async callback for PAUSE actions (approval flow)
    approval_callback: Optional[Callable[[DriftAssessment], bool]] = None
    # Override: these tools are always allowed (trusted ops)
    always_allow:     set[str]               = field(default_factory=set)


class PolicyEngine:
    """
    Stateless policy evaluator.

    Usage:
        engine = PolicyEngine(config)
        assessment = engine.evaluate(tool_call, current_state)
        if assessment.action != InterventionAction.BLOCK:
            new_state = assessment.after_state
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate(
        self,
        tool_call: ToolCall,
        state: SafetyState,
    ) -> DriftAssessment:
        """
        Evaluate a tool call and return a DriftAssessment.

        The assessment contains:
        - after_state: new state if the call proceeds
        - violation_probability: P(violation in next `horizon` steps)
        - action: recommended InterventionAction
        - reason: human-readable explanation
        """
        # Hard overrides
        if tool_call.name in self.config.always_block:
            return DriftAssessment(
                tool_call=tool_call,
                before_state=state,
                after_state=state,
                violation_probability=1.0,
                horizon=0,
                action=InterventionAction.BLOCK,
                reason=f"Tool '{tool_call.name}' is in the always-block list",
            )

        if tool_call.name in self.config.always_allow:
            after = state.update(tool_call)
            return DriftAssessment(
                tool_call=tool_call,
                before_state=state,
                after_state=after,
                violation_probability=0.0,
                horizon=self.config.horizon,
                action=InterventionAction.LOG_ONLY,
                reason=f"Tool '{tool_call.name}' is in the always-allow list",
            )

        # Project the state forward if the call proceeds
        after_state = state.update(tool_call)

        # Compute context pressure from session length
        pressure = context_pressure_from_session(
            step_count=state.step_count,
            permission_breadth=int(after_state.tool_escalation),
        )

        # P(violation within horizon steps | after_state)
        prob = violation_probability(
            state=after_state,
            horizon=self.config.horizon,
            task_type=self.config.task_type,
            context_pressure=pressure,
        )

        # Map probability to action
        action = InterventionAction.LOG_ONLY
        reason = ""
        for threshold in sorted(self.config.thresholds,
                                 key=lambda t: -t.prob_threshold):
            if prob >= threshold.prob_threshold:
                action = threshold.action
                reason = threshold.reason_template.format(prob=prob)
                break

        return DriftAssessment(
            tool_call=tool_call,
            before_state=state,
            after_state=after_state,
            violation_probability=prob,
            horizon=self.config.horizon,
            action=action,
            reason=reason,
        )

    def evaluate_by_name(
        self,
        name: str,
        arguments: dict | None = None,
        state: SafetyState | None = None,
    ) -> DriftAssessment:
        """
        Convenience method: classify by name, then evaluate.
        Creates a fresh SafetyState if none is provided.
        """
        call = classify(name, arguments)
        return self.evaluate(call, state or SafetyState())
