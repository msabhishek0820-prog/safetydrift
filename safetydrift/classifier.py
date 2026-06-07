"""
SafetyDrift - Tool call classifier

Maps MCP tool names and arguments to risk dimensions (DataExposure,
ToolEscalation, Reversibility).

The classifier uses a layered approach:
  1. Exact name match against a built-in registry
  2. Keyword pattern matching against the tool NAME only (for escalation)
  3. Keyword matching against argument KEYS (for context)
  4. Fallback to INTERNAL / READ / FULLY

Users can extend the registry with custom rules via add_rule().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import DataExposure, Reversibility, ToolCall, ToolEscalation


@dataclass(frozen=True)
class ClassifierRule:
    """A single classification rule keyed on a name pattern."""
    pattern:         str             # regex matched against tool NAME only
    data_exposure:   DataExposure
    tool_escalation: ToolEscalation
    reversibility:   Reversibility
    description:     str = ""


# ── Built-in rules ───────────────────────────────────────────────────────────
# Ordered: first match wins. Rules at the top have highest priority.

_BUILTIN_RULES: list[ClassifierRule] = [

    # ── Web / public search (must be before database rules) ──────────────
    ClassifierRule(r"(web.?search|browse|google|bing|duckduckgo|tavily|serp)",
                   DataExposure.PUBLIC, ToolEscalation.NETWORK,
                   Reversibility.FULLY,
                   "Web search — public data only"),

    # ── Highly sensitive / irreversible ────────────────────────────────────
    ClassifierRule(r"send.?(email|mail|message|slack|teams|discord|webhook|sms|notify)",
                   DataExposure.CONFIDENTIAL, ToolEscalation.EXTERNAL,
                   Reversibility.IRREVERSIBLE,
                   "Outbound communication — cannot be recalled"),

    ClassifierRule(r"(deploy|publish|release|push.to.prod|merge|force.push)",
                   DataExposure.INTERNAL, ToolEscalation.EXTERNAL,
                   Reversibility.IRREVERSIBLE,
                   "Deployment / release — hard to roll back"),

    ClassifierRule(r"(delete|remove|drop|truncate|purge|wipe)",
                   DataExposure.INTERNAL, ToolEscalation.WRITE,
                   Reversibility.IRREVERSIBLE,
                   "Destructive operation"),

    ClassifierRule(r"(payment|charge|invoice|transfer|billing|stripe|paypal)",
                   DataExposure.SENSITIVE, ToolEscalation.EXTERNAL,
                   Reversibility.IRREVERSIBLE,
                   "Financial transaction"),

    # ── Credential / secret access ─────────────────────────────────────────
    ClassifierRule(r"(get.?secret|get.?api.?key|get.?credential|get.?password|get.?token)",
                   DataExposure.SENSITIVE, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Credential / secret read"),

    # ── External network write ops ─────────────────────────────────────────
    ClassifierRule(r"(post|create|update|patch|put).*(api|endpoint|webhook|http)",
                   DataExposure.INTERNAL, ToolEscalation.EXTERNAL,
                   Reversibility.MOSTLY_NOT,
                   "External API write"),

    ClassifierRule(r"(upload|s3|gcs|blob|bucket|azure.storage)",
                   DataExposure.INTERNAL, ToolEscalation.EXTERNAL,
                   Reversibility.MOSTLY_NOT,
                   "Cloud storage write"),

    # ── Network reads ──────────────────────────────────────────────────────
    ClassifierRule(r"(fetch|http.?get|download|curl|wget|url)",
                   DataExposure.PUBLIC, ToolEscalation.NETWORK,
                   Reversibility.FULLY,
                   "Network read"),

    # ── Local write ops ────────────────────────────────────────────────────
    ClassifierRule(r"(write.file|edit.file|create.file|str.?replace|append.to)",
                   DataExposure.INTERNAL, ToolEscalation.WRITE,
                   Reversibility.MOSTLY,
                   "Local file write (usually undoable with git)"),

    ClassifierRule(r"(execute|run.command|shell|bash|subprocess|eval)",
                   DataExposure.INTERNAL, ToolEscalation.WRITE,
                   Reversibility.MIXED,
                   "Shell / code execution"),

    ClassifierRule(r"(git.commit|git.push|git.merge)",
                   DataExposure.INTERNAL, ToolEscalation.NETWORK,
                   Reversibility.MOSTLY_NOT,
                   "Git push — affects remote"),

    # ── Database access (after web_search to avoid false positives) ────────
    ClassifierRule(r"(sql|postgres|mysql|mongo|redis|sqlite|dynamodb|database.?query)",
                   DataExposure.CONFIDENTIAL, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Database read"),

    # ── Confidential reads ─────────────────────────────────────────────────
    ClassifierRule(r"(read|get|open|load).*(private|confidential|secret|credential|env)",
                   DataExposure.CONFIDENTIAL, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Confidential file read"),

    ClassifierRule(r"(gmail|calendar|gdrive|outlook|contacts|drive)",
                   DataExposure.CONFIDENTIAL, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Personal / organisational data read"),

    # ── Standard local reads ───────────────────────────────────────────────
    ClassifierRule(r"(read.file|open.file|list.files|list.dir|view.file|cat)",
                   DataExposure.INTERNAL, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Local file read"),

    ClassifierRule(r"(search.code|find.files|grep|ripgrep|glob)",
                   DataExposure.INTERNAL, ToolEscalation.READ,
                   Reversibility.FULLY,
                   "Code/file search"),
]

# User-extensible registry (prepended, so user rules take priority)
_custom_rules: list[ClassifierRule] = []


def add_rule(rule: ClassifierRule) -> None:
    """Register a custom classifier rule (prepended for priority)."""
    _custom_rules.insert(0, rule)


def classify(name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
    """
    Classify a tool call by name.

    Matches rules against the tool NAME only (not argument values),
    to avoid false positives from argument content.

    Returns a ToolCall with risk dimensions assigned.
    Falls back to INTERNAL / READ / FULLY if no rule matches.
    """
    arguments = arguments or {}
    # Match against name only (not argument values) to avoid false positives
    for rule in _custom_rules + _BUILTIN_RULES:
        if re.search(rule.pattern, name, re.IGNORECASE):
            return ToolCall(
                name=name,
                arguments=arguments,
                data_exposure=rule.data_exposure,
                tool_escalation=rule.tool_escalation,
                reversibility=rule.reversibility,
            )

    # Default fallback: treat as a low-risk read
    return ToolCall(
        name=name,
        arguments=arguments,
        data_exposure=DataExposure.INTERNAL,
        tool_escalation=ToolEscalation.READ,
        reversibility=Reversibility.FULLY,
    )
