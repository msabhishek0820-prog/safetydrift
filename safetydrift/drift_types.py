"""
DriftGuard - Core types

Safety state is modelled across 3 dimensions, matching the SafetyDrift paper
(Dhodapkar & Pishori, arXiv:2603.27148, March 2026).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


# ── Dimension enums ──────────────────────────────────────────────────────────

class DataExposure(IntEnum):
    """What sensitivity of data has the agent accessed?"""
    NONE         = 0   # No data read yet
    PUBLIC       = 1   # Only publicly available data
    INTERNAL     = 2   # Internal/organisational data
    CONFIDENTIAL = 3   # Confidential (PII, credentials, configs)
    SENSITIVE    = 4   # Highly sensitive (keys, medical, financial)


class ToolEscalation(IntEnum):
    """What level of capability has the agent gained?"""
    NONE     = 0  # No tool use
    READ     = 1  # Read-only ops (list, get, search)
    WRITE    = 2  # Local write/edit ops
    NETWORK  = 3  # Outbound network ops (fetch, download)
    EXTERNAL = 4  # External service writes (email, post, deploy, pay)


class Reversibility(IntEnum):
    """How reversible are the actions taken so far?"""
    FULLY       = 0  # Everything is undoable
    MOSTLY      = 1  # Minor irreversible side effects
    MIXED       = 2  # Mix of reversible and not
    MOSTLY_NOT  = 3  # Mostly irreversible
    IRREVERSIBLE= 4  # Hard to undo (sent email, deployed, deleted)


class RiskLevel(IntEnum):
    """Composite risk level derived from all three dimensions."""
    SAFE     = 0
    LOW      = 1
    MODERATE = 2
    HIGH     = 3
    CRITICAL = 4


class InterventionAction(IntEnum):
    """What SafetyDrift does when risk exceeds a threshold."""
    LOG_ONLY  = 0  # Record to audit log, proceed
    WARN      = 1  # Warn caller, proceed
    PAUSE     = 2  # Pause and request human approval
    BLOCK     = 3  # Block the tool call entirely


# ── State snapshot ───────────────────────────────────────────────────────────

@dataclass
class SafetyState:
    """
    Cumulative safety state for one agent session.

    Dimensions only ever increase (monotonic) — consistent with the
    paper's absorbing Markov chain design where states don't regress.
    """
    data_exposure:  DataExposure  = DataExposure.NONE
    tool_escalation: ToolEscalation = ToolEscalation.NONE
    reversibility:  Reversibility = Reversibility.FULLY
    step_count:     int           = 0

    @property
    def risk_level(self) -> RiskLevel:
        """Derive composite risk level from all three dimensions."""
        score = (
            int(self.data_exposure)  * 0.4 +
            int(self.tool_escalation) * 0.35 +
            int(self.reversibility)  * 0.25
        )
        if score < 0.6:  return RiskLevel.SAFE
        if score < 1.2:  return RiskLevel.LOW
        if score < 2.0:  return RiskLevel.MODERATE
        if score < 3.0:  return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def update(self, call: "ToolCall") -> "SafetyState":
        """Return a new state after absorbing a tool call (monotonic update)."""
        return SafetyState(
            data_exposure   = max(self.data_exposure,   call.data_exposure),
            tool_escalation = max(self.tool_escalation, call.tool_escalation),
            reversibility   = max(self.reversibility,   call.reversibility),
            step_count      = self.step_count + 1,
        )

    def to_dict(self) -> dict:
        return {
            "data_exposure":   self.data_exposure.name,
            "tool_escalation": self.tool_escalation.name,
            "reversibility":   self.reversibility.name,
            "risk_level":      self.risk_level.name,
            "step_count":      self.step_count,
        }


# ── Tool call descriptor ─────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """Represents a single MCP tool invocation with its risk dimensions."""
    name:            str
    arguments:       dict[str, Any] = field(default_factory=dict)
    data_exposure:   DataExposure   = DataExposure.NONE
    tool_escalation: ToolEscalation = ToolEscalation.NONE
    reversibility:   Reversibility  = Reversibility.FULLY

    def to_dict(self) -> dict:
        return {
            "name":            self.name,
            "data_exposure":   self.data_exposure.name,
            "tool_escalation": self.tool_escalation.name,
            "reversibility":   self.reversibility.name,
        }


# ── Drift assessment ─────────────────────────────────────────────────────────

@dataclass
class DriftAssessment:
    """
    Result of evaluating a tool call against the current safety state.

    violation_probability is P(violation within `horizon` steps) computed
    via the Markov chain model in markov.py.
    """
    tool_call:             ToolCall
    before_state:          SafetyState
    after_state:           SafetyState
    violation_probability: float            # 0.0 – 1.0
    horizon:               int              # number of future steps evaluated
    action:                InterventionAction
    reason:                str = ""

    @property
    def risk_delta(self) -> int:
        return int(self.after_state.risk_level) - int(self.before_state.risk_level)

    def to_dict(self) -> dict:
        return {
            "tool_call":             self.tool_call.to_dict(),
            "before_state":          self.before_state.to_dict(),
            "after_state":           self.after_state.to_dict(),
            "violation_probability": round(self.violation_probability, 4),
            "horizon":               self.horizon,
            "action":                self.action.name,
            "reason":                self.reason,
        }
