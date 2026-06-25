"""
SafetyDrift - Cross-session memory

Persists safety state across agent sessions using SQLite (stdlib, zero deps).

Problem it solves:
  Session 1: agent reads /etc/secrets.env  → ends cleanly, no violation
  Session 2: agent emails secrets externally → looks like a fresh safe start
  Without memory: Session 2 starts at SAFE state → violation not predicted
  With memory:    Session 2 inherits Session 1 risk → caught immediately

Design:
  - SQLite for persistence (built into Python, no install needed)
  - Risk decay over time (old sessions carry less weight)
  - agent_id links sessions belonging to the same agent/user
  - Drop-in: pass store= to Session, everything else unchanged
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .drift_types import DataExposure, Reversibility, SafetyState, ToolEscalation


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    data_exposure   INTEGER NOT NULL DEFAULT 0,
    tool_escalation INTEGER NOT NULL DEFAULT 0,
    reversibility   INTEGER NOT NULL DEFAULT 0,
    step_count      INTEGER NOT NULL DEFAULT 0,
    task_type       TEXT NOT NULL DEFAULT 'default',
    closed          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    tool_name   TEXT NOT NULL,
    action      TEXT NOT NULL,
    risk_level  TEXT NOT NULL,
    violation_p REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id);
"""


# ── Decay model ───────────────────────────────────────────────────────────────

def _decayed_level(level: int, age_hours: float, half_life_hours: float = 24.0) -> int:
    """
    Apply exponential decay to a risk dimension level.

    Risk from old sessions fades: a CONFIDENTIAL exposure 48h ago
    carries half the weight of one from 24h ago.
    Minimum decay floor = 0 (risk never goes negative).
    """
    if level == 0:
        return 0
    decayed = level * math.exp(-math.log(2) * age_hours / half_life_hours)
    return max(0, round(decayed))


# ── Store ─────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    SQLite-backed cross-session memory.

    Usage:
        store = SessionStore("safetydrift.db")
        session = Session(agent_id="user_123", store=store)
        # State is automatically saved and loaded across runs
    """

    def __init__(
        self,
        db_path:          str  = "safetydrift.db",
        decay_half_life_h: float = 24.0,   # risk halves every 24h
        max_history_days:  int  = 7,        # ignore sessions older than this
    ) -> None:
        self.db_path           = db_path
        self.decay_half_life_h = decay_half_life_h
        self.max_history_days  = max_history_days
        self._conn             = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_prior_state(self, agent_id: str) -> SafetyState:
        """
        Load the accumulated safety state for an agent across all prior sessions.

        Applies time decay: older sessions carry less weight.
        Returns SafetyState() (all zeros) if no history exists.
        """
        cutoff = time.time() - (self.max_history_days * 86400)
        cur = self._conn.execute(
            """SELECT data_exposure, tool_escalation, reversibility,
                      step_count, updated_at
               FROM sessions
               WHERE agent_id=? AND closed=1 AND updated_at > ?
               ORDER BY updated_at DESC""",
            (agent_id, cutoff),
        )
        rows = cur.fetchall()
        if not rows:
            return SafetyState()

        now = time.time()
        max_de = max_te = max_rv = total_steps = 0

        for de, te, rv, steps, updated_at in rows:
            age_h = (now - updated_at) / 3600.0
            max_de    = max(max_de, _decayed_level(de, age_h, self.decay_half_life_h))
            max_te    = max(max_te, _decayed_level(te, age_h, self.decay_half_life_h))
            max_rv    = max(max_rv, _decayed_level(rv, age_h, self.decay_half_life_h))
            total_steps += steps

        return SafetyState(
            data_exposure   = DataExposure(max_de),
            tool_escalation = ToolEscalation(max_te),
            reversibility   = Reversibility(max_rv),
            step_count      = total_steps,
        )

    def get_agent_history(self, agent_id: str, limit: int = 20) -> list[dict]:
        """Return recent audit log entries for an agent."""
        cur = self._conn.execute(
            """SELECT timestamp, tool_name, action, risk_level, violation_p
               FROM audit_log WHERE agent_id=?
               ORDER BY timestamp DESC LIMIT ?""",
            (agent_id, limit),
        )
        return [
            {"timestamp": r[0], "tool": r[1], "action": r[2],
             "risk": r[3], "p_violation": r[4]}
            for r in cur.fetchall()
        ]

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_session(self, session_id: str, agent_id: str,
                     state: SafetyState, task_type: str) -> None:
        """Upsert current session state."""
        now = time.time()
        self._conn.execute(
            """INSERT INTO sessions
               (session_id, agent_id, created_at, updated_at,
                data_exposure, tool_escalation, reversibility,
                step_count, task_type, closed)
               VALUES (?,?,?,?,?,?,?,?,?,0)
               ON CONFLICT(session_id) DO UPDATE SET
                 updated_at=excluded.updated_at,
                 data_exposure=excluded.data_exposure,
                 tool_escalation=excluded.tool_escalation,
                 reversibility=excluded.reversibility,
                 step_count=excluded.step_count""",
            (session_id, agent_id, now, now,
             int(state.data_exposure), int(state.tool_escalation),
             int(state.reversibility), state.step_count, task_type),
        )
        self._conn.commit()

    def close_session(self, session_id: str) -> None:
        """Mark session as closed so it contributes to future prior state."""
        self._conn.execute(
            "UPDATE sessions SET closed=1, updated_at=? WHERE session_id=?",
            (time.time(), session_id),
        )
        self._conn.commit()

    def log_action(self, session_id: str, agent_id: str,
                   tool_name: str, action: str,
                   risk_level: str, violation_p: float) -> None:
        """Append one tool call to the audit log."""
        self._conn.execute(
            """INSERT INTO audit_log
               (session_id, agent_id, timestamp, tool_name, action, risk_level, violation_p)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, agent_id, time.time(),
             tool_name, action, risk_level, violation_p),
        )
        self._conn.commit()

    def agent_summary(self, agent_id: str) -> dict:
        """Risk summary for an agent across all sessions."""
        cur = self._conn.execute(
            """SELECT COUNT(*), SUM(step_count),
                      MAX(data_exposure), MAX(tool_escalation), MAX(reversibility)
               FROM sessions WHERE agent_id=?""", (agent_id,)
        )
        row = cur.fetchone()
        blocks = self._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE agent_id=? AND action='BLOCK'",
            (agent_id,)
        ).fetchone()[0]
        pauses = self._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE agent_id=? AND action='PAUSE'",
            (agent_id,)
        ).fetchone()[0]
        return {
            "agent_id":      agent_id,
            "total_sessions": row[0] or 0,
            "total_steps":   row[1] or 0,
            "peak_exposure": DataExposure(row[2] or 0).name,
            "peak_escalation": ToolEscalation(row[3] or 0).name,
            "peak_reversibility": Reversibility(row[4] or 0).name,
            "total_blocks":  blocks,
            "total_pauses":  pauses,
        }

    def close(self) -> None:
        self._conn.close()