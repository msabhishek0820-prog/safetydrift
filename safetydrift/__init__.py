"""
SafetyDrift — Safety drift prediction for AI agents.

The first open-source implementation of the SafetyDrift framework
(Dhodapkar & Pishori, arXiv:2603.27148, March 2026).

Predicts when individually safe AI agent actions are about to compound
into a safety violation, using absorbing Markov chain analysis.

Quick start:
    from safetydrift import Session, InterventionAction

    session = Session(task_type="communication")

    # Before any tool call:
    result = session.gate("send_email", {"to": "boss@corp.com", "body": "..."})

    if result.action == InterventionAction.BLOCK:
        raise RuntimeError(f"safetydrift blocked: {result.reason}")
    elif result.action == InterventionAction.PAUSE:
        approved = ask_human_approval(result)
        if not approved:
            raise RuntimeError("Human rejected action")

    # Safe to proceed
    actually_send_email(...)

"""

from .drift_types import (
    DataExposure,
    DriftAssessment,
    InterventionAction,
    Reversibility,
    RiskLevel,
    SafetyState,
    ToolCall,
    ToolEscalation,
)
from .classifier import classify, add_rule, ClassifierRule
from .markov import violation_probability, steps_to_threshold
from .policy import PolicyConfig, PolicyEngine, PolicyThreshold
from .session import Session, AuditEntry

__version__ = "0.1.0"
__all__ = [
    # Types
    "DataExposure", "ToolEscalation", "Reversibility", "RiskLevel",
    "InterventionAction", "SafetyState", "ToolCall", "DriftAssessment",
    # Classifier
    "classify", "add_rule", "ClassifierRule",
    # Markov
    "violation_probability", "steps_to_threshold",
    # Policy
    "PolicyConfig", "PolicyEngine", "PolicyThreshold",
    # Session
    "Session", "AuditEntry",
]
