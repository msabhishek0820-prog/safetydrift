"""
safetydrift - Session manager

Tracks cumulative safety state across a full agent session, maintains
an audit log, and exposes the gate() method that agents call before
each tool invocation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .classifier import classify
from .policy import PolicyConfig, PolicyEngine
from .types import (
    DriftAssessment,
    InterventionAction,
    SafetyState,
    ToolCall,
)


@dataclass
class AuditEntry:
    timestamp:   float
    session_id:  str
    assessment:  dict    # serialised DriftAssessment


@dataclass
class Session:
    """
    One agent session.  Instantiate at the start of each agent run.

    Example:
        session = Session(task_type="communication")
        ...
        result = session.gate("send_email", {"to": "boss@corp.com"})
        if result.action != InterventionAction.BLOCK:
            actually_send_email(...)
    """
    session_id:  str              = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type:   str              = "default"
    config:      PolicyConfig     = field(default_factory=PolicyConfig)
    _state:      SafetyState      = field(default_factory=SafetyState, init=False)
    _engine:     PolicyEngine     = field(default=None, init=False)
    _audit_log:  list[AuditEntry] = field(default_factory=list, init=False)
    # Optional callback: called on BLOCK/PAUSE assessments
    on_intervention: Optional[Callable[[DriftAssessment], None]] = None

    def __post_init__(self) -> None:
        self.config.task_type = self.task_type
        self._engine = PolicyEngine(self.config)

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def audit_log(self) -> list[AuditEntry]:
        return list(self._audit_log)

    def gate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        pre_classified: ToolCall | None = None,
    ) -> DriftAssessment:
        """
        Evaluate a tool call.  Call this BEFORE executing any tool.

        Args:
            tool_name: MCP tool name (e.g. "send_email")
            arguments: tool arguments dict (optional — improves classification)
            pre_classified: if you already classified the call, pass it here

        Returns:
            DriftAssessment with .action indicating what to do next.
            If .action is BLOCK, do NOT proceed with the tool call.
            If .action is PAUSE, await human approval before proceeding.

        Side effects:
            - Advances _state if assessment.action != BLOCK
            - Appends to _audit_log
        """
        call = pre_classified or classify(tool_name, arguments)
        assessment = self._engine.evaluate(call, self._state)

        # Advance state only if we're not blocking
        if assessment.action != InterventionAction.BLOCK:
            self._state = assessment.after_state

        # Record to audit log
        self._audit_log.append(AuditEntry(
            timestamp=time.time(),
            session_id=self.session_id,
            assessment=assessment.to_dict(),
        ))

        # Fire intervention callback
        if assessment.action in (InterventionAction.BLOCK, InterventionAction.PAUSE):
            if self.on_intervention:
                self.on_intervention(assessment)

        return assessment

    def summary(self) -> dict:
        """Return a human-readable session summary."""
        blocked = [e for e in self._audit_log
                   if e.assessment["action"] == "BLOCK"]
        paused  = [e for e in self._audit_log
                   if e.assessment["action"] == "PAUSE"]
        return {
            "session_id":   self.session_id,
            "task_type":    self.task_type,
            "steps":        self._state.step_count,
            "final_state":  self._state.to_dict(),
            "total_calls":  len(self._audit_log),
            "blocked":      len(blocked),
            "paused":       len(paused),
        }

    def export_audit_log(self, path: str) -> None:
        """Write the full audit log as JSONL."""
        with open(path, "w") as f:
            for entry in self._audit_log:
                f.write(json.dumps({
                    "timestamp":  entry.timestamp,
                    "session_id": entry.session_id,
                    **entry.assessment,
                }) + "\n")
