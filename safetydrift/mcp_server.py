"""
safetydrift - MCP Server

Exposes safetydrift itself as an MCP server so any MCP-compatible
agent (Claude Code, Cursor, Copilot, etc.) can call it directly.

Tools exposed:
  - dg_gate         : evaluate a tool call before executing it
  - dg_session_state: get current session risk state
  - dg_summary      : get session summary and audit stats
  - dg_reset        : reset session state (new task)

Run with:
    python -m safetydrift.mcp_server

Or use as an MCP server URL in mcp.json:
    {"safetydrift": {"command": "python", "args": ["-m", "safetydrift.mcp_server"]}}
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .policy import PolicyConfig
from .session import Session
from .drift_types import InterventionAction


# Global session (one per server process — suitable for single-agent use)
_session: Session = Session(task_type="default")


def _handle_request(req: dict) -> dict:
    """Dispatch a single JSON-RPC 2.0 request."""
    method = req.get("method", "")
    params = req.get("params", {})
    req_id = req.get("id")

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, msg: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": code, "message": msg}}

    # ── MCP initialization ─────────────────────────────────────────────────
    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "safetydrift", "version": "0.1.0"},
        })

    if method == "notifications/initialized":
        return {}  # no response needed

    # ── Tool listing ───────────────────────────────────────────────────────
    if method == "tools/list":
        return ok({"tools": [
            {
                "name": "dg_gate",
                "description": (
                    "Evaluate a tool call BEFORE executing it. "
                    "Returns a DriftAssessment with action=ALLOW, WARN, PAUSE, or BLOCK. "
                    "ALWAYS call this before executing any tool that reads files, "
                    "writes data, or communicates externally."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string",
                                      "description": "Name of the tool to evaluate"},
                        "arguments": {"type": "object",
                                      "description": "Arguments you would pass to the tool"},
                    },
                    "required": ["tool_name"],
                },
            },
            {
                "name": "dg_session_state",
                "description": "Get the current cumulative safety state of this session.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "dg_summary",
                "description": "Get session summary: steps taken, blocks, pauses, risk level.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "dg_reset",
                "description": "Reset session state for a new task. Pass task_type: communication|information|technical|autonomous|default.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["communication", "information",
                                     "technical", "autonomous", "default"],
                        },
                    },
                },
            },
        ]})

    # ── Tool execution ─────────────────────────────────────────────────────
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})

        global _session

        if name == "dg_gate":
            tool_name = args.get("tool_name", "")
            tool_args = args.get("arguments", {})
            assessment = _session.gate(tool_name, tool_args)
            result = assessment.to_dict()
            # Add a plain-English verdict for the agent
            action = assessment.action
            if action == InterventionAction.BLOCK:
                verdict = "🚫 BLOCKED — do not execute this tool call"
            elif action == InterventionAction.PAUSE:
                verdict = "⏸ PAUSED — request human approval before proceeding"
            elif action == InterventionAction.WARN:
                verdict = "⚠️ WARNING — proceed with caution and log this action"
            else:
                verdict = "✅ SAFE — proceed normally"
            result["verdict"] = verdict
            return ok({"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

        elif name == "dg_session_state":
            return ok({"content": [{"type": "text",
                                     "text": json.dumps(_session.state.to_dict(), indent=2)}]})

        elif name == "dg_summary":
            return ok({"content": [{"type": "text",
                                     "text": json.dumps(_session.summary(), indent=2)}]})

        elif name == "dg_reset":
            task_type = args.get("task_type", "default")
            _session = Session(task_type=task_type)
            return ok({"content": [{"type": "text",
                                     "text": f"Session reset. task_type={task_type}"}]})

        return err(-32601, f"Unknown tool: {name}")

    return err(-32601, f"Unknown method: {method}")


def run_stdio() -> None:
    """Run as a stdio MCP server (for use with Claude Code, Cursor, etc.)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle_request(req)
            if resp:  # skip empty responses (notifications)
                print(json.dumps(resp), flush=True)
        except json.JSONDecodeError as e:
            print(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32603, "message": f"Internal error: {e}"},
            }), flush=True)


if __name__ == "__main__":
    run_stdio()
