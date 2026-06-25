"""
SafetyDrift - Framework Integrations (updated June 2026)

Changes from initial version:
  - OpenAI Agents SDK: InputGuardrail → ToolGuardrail (runs on every tool call)
  - LangGraph 1.2.6: uses interrupt() for PAUSE actions (built-in HITL)
  - LangChain 1.3.10: removed legacy langchain.tools fallback import
  - AutoGen / CrewAI: unchanged (no breaking changes)

Adapters: LangChain, LangGraph, OpenAI Agents SDK, Anthropic Claude SDK,
          AutoGen / AG2, CrewAI
"""

from __future__ import annotations
from typing import Any, Callable

from .session import Session, PersistentSession
from .store   import SessionStore
from .drift_types   import InterventionAction


class BlockedByDriftGuard(RuntimeError):
    """Raised when SafetyDrift blocks a tool call."""


def _gate(session: Session, tool_name: str,
          arguments: dict | None = None):
    """Evaluate a tool call. Raises BlockedByDriftGuard if BLOCK."""
    result = session.gate(tool_name, arguments or {})
    if result.action == InterventionAction.BLOCK:
        raise BlockedByDriftGuard(
            f"[SafetyDrift] BLOCKED '{tool_name}' — {result.reason} "
            f"(P={result.violation_probability:.0%})"
        )
    return result


# ── 1. LangChain 1.3.10 ───────────────────────────────────────────────────────

class LangChainGuard:
    """
    Wraps any LangChain BaseTool with SafetyDrift pre-execution checks.

    Fix: removed legacy langchain.tools fallback — langchain_core.tools
    is the only correct import path in LangChain v1.3.x.

    Example:
        from langchain_core.tools import tool
        from safetydrift.integrations import LangChainGuard

        @tool
        def send_email(to: str, body: str) -> str:
            \"\"\"Send an email.\"\"\"
            ...

        guard     = LangChainGuard(task_type="communication")
        safe_tool = guard.wrap(send_email)
    """

    def __init__(self, task_type: str = "default",
                 session: Session | None = None,
                 store: SessionStore | None = None,
                 agent_id: str | None = None) -> None:
        if store and agent_id:
            self.session = PersistentSession(
                agent_id=agent_id, store=store, task_type=task_type)
        else:
            self.session = session or Session(task_type=task_type)

    def wrap(self, tool: Any) -> Any:
        """Wrap a LangChain tool with drift checking."""
        from langchain_core.tools import BaseTool  # v1.0+ only

        original_run  = tool._run
        original_arun = getattr(tool, "_arun", None)
        _session      = self.session
        tool_name     = tool.name

        def guarded_run(*args, **kwargs):
            _gate(_session, tool_name, kwargs or {})
            return original_run(*args, **kwargs)

        async def guarded_arun(*args, **kwargs):
            _gate(_session, tool_name, kwargs or {})
            if original_arun:
                return await original_arun(*args, **kwargs)
            return guarded_run(*args, **kwargs)

        tool._run  = guarded_run
        tool._arun = guarded_arun
        return tool

    def wrap_all(self, tools: list) -> list:
        return [self.wrap(t) for t in tools]


# ── 2. LangGraph 1.2.6 ───────────────────────────────────────────────────────

class LangGraphGuard:
    """
    LangGraph 1.2.6 integration using interrupt() for PAUSE actions.

    What interrupt() does: pauses the graph at the exact node and surfaces
    a payload to the human reviewer. When they approve, the graph resumes
    from the same state — no work is lost. This is LangGraph's built-in
    HITL primitive added in v1.0.

    Think of it like a pull request approval gate in a CI/CD pipeline:
    the pipeline stops at that step, a human reviews, then it continues.

    Example:
        from langgraph.graph import StateGraph
        from safetydrift.integrations import LangGraphGuard

        guard = LangGraphGuard(task_type="communication")

        def tool_node(state):
            guard.check(state["next_tool"], state["tool_args"])
            return {"result": run_tool(state["next_tool"], state["tool_args"])}

        graph = StateGraph(...)
        graph.add_node("tools", tool_node)
    """

    def __init__(self, task_type: str = "default",
                 session: Session | None = None,
                 store: SessionStore | None = None,
                 agent_id: str | None = None) -> None:
        if store and agent_id:
            self.session = PersistentSession(
                agent_id=agent_id, store=store, task_type=task_type)
        else:
            self.session = session or Session(task_type=task_type)

    def check(self, tool_name: str, arguments: dict | None = None) -> None:
        """
        Gate a tool call inside a LangGraph node.

        BLOCK → raises BlockedByDriftGuard (graph stops entirely)
        PAUSE → calls interrupt() (graph pauses for human approval)
        WARN  → proceeds, logged
        LOG   → proceeds silently
        """
        from langgraph.drift_types import interrupt  # LangGraph 1.0+

        result = self.session.gate(tool_name, arguments or {})

        if result.action == InterventionAction.BLOCK:
            raise BlockedByDriftGuard(
                f"[SafetyDrift] BLOCKED '{tool_name}' — {result.reason}"
            )

        if result.action == InterventionAction.PAUSE:
            # Pause graph and surface risk details to the human reviewer
            interrupt({
                "tool":          tool_name,
                "arguments":     arguments,
                "reason":        result.reason,
                "p_violation":   result.violation_probability,
                "risk_level":    result.after_state.risk_level.name,
            })


# ── 3. OpenAI Agents SDK — ToolGuardrail (fix from InputGuardrail) ────────────

def openai_tool_guardrail(task_type: str = "default",
                           session: Session | None = None) -> Any:
    """
    Returns an OpenAI Agents SDK ToolGuardrail that runs SafetyDrift
    before EVERY tool invocation in the agent chain.

    Why ToolGuardrail not InputGuardrail:
      InputGuardrail only runs for the FIRST agent in the chain.
      ToolGuardrail runs on every custom function-tool call — which is
      what SafetyDrift needs since risk accumulates across all steps.

      Analogy: InputGuardrail checks you at the hotel entrance.
      ToolGuardrail checks you at the door of every room.

    Example:
        from agents import Agent, function_tool
        from safetydrift.integrations import openai_tool_guardrail

        guardrail = openai_tool_guardrail(task_type="communication")

        @function_tool(input_guardrails=[guardrail])
        def send_email(to: str, body: str) -> str:
            ...

        agent = Agent(name="MyAgent", tools=[send_email])
    """
    try:
        from agents import ToolGuardrail, GuardrailFunctionOutput, RunContextWrapper
    except ImportError:
        raise ImportError("pip install openai-agents")

    _session = session or Session(task_type=task_type)

    async def drift_check(
        ctx: RunContextWrapper,
        tool_name: str,
        tool_input: Any,
    ) -> GuardrailFunctionOutput:
        args = tool_input if isinstance(tool_input, dict) \
               else {"input": str(tool_input)[:200]}
        result = _session.gate(tool_name, args)
        triggered = result.action in (
            InterventionAction.BLOCK, InterventionAction.PAUSE)
        return GuardrailFunctionOutput(
            output_info={
                "risk_level":  result.after_state.risk_level.name,
                "p_violation": result.violation_probability,
                "action":      result.action.name,
                "reason":      result.reason,
            },
            tripwire_triggered=triggered,
        )

    return ToolGuardrail(guardrail_function=drift_check)


# ── 4. Anthropic Claude SDK (works with all models incl. Fable 5) ─────────────

class AnthropicHook:
    """
    Pre-tool hook for the Anthropic Claude SDK.

    Compatible with all Claude models: Haiku 4.5, Sonnet 4.6,
    Opus 4.8, and Claude Fable 5 (June 2026).

    Example:
        from anthropic import Anthropic
        from safetydrift.integrations import AnthropicHook

        hook   = AnthropicHook(task_type="default")
        client = Anthropic()

        for block in response.content:
            if block.type == "tool_use":
                hook.before_tool(block.name, block.input)
                result = execute_tool(block.name, block.input)
    """

    def __init__(self, task_type: str = "default",
                 store: SessionStore | None = None,
                 agent_id: str | None = None) -> None:
        if store and agent_id:
            self.session: Session = PersistentSession(
                agent_id=agent_id, store=store, task_type=task_type)
        else:
            self.session = Session(task_type=task_type)

    def before_tool(self, tool_name: str,
                    tool_input: dict | None = None) -> dict:
        """
        Call before executing any Claude tool_use block.
        Raises BlockedByDriftGuard if blocked. Returns assessment dict.
        """
        result = self.session.gate(tool_name, tool_input or {})
        if result.action == InterventionAction.BLOCK:
            raise BlockedByDriftGuard(
                f"[SafetyDrift] BLOCKED '{tool_name}' — {result.reason} "
                f"(P={result.violation_probability:.0%})"
            )
        return result.to_dict()

    def summary(self) -> dict:
        return self.session.summary()


# ── 5. AutoGen / AG2 (no changes needed) ────────────────────────────────────

class AutoGenMiddleware:
    """
    AutoGen / AG2 middleware — no breaking changes in 2026.

    Example:
        from safetydrift.integrations import AutoGenMiddleware

        middleware = AutoGenMiddleware(task_type="default")
        agent.register_function(
            function_map={
                name: middleware.wrap_fn(name, fn)
                for name, fn in tools.items()
            }
        )
    """

    def __init__(self, task_type: str = "default",
                 session: Session | None = None) -> None:
        self.session = session or Session(task_type=task_type)

    def wrap_fn(self, tool_name: str, fn: Callable) -> Callable:
        _session = self.session

        def guarded(**kwargs):
            _gate(_session, tool_name, kwargs)
            return fn(**kwargs)

        guarded.__name__ = fn.__name__
        guarded.__doc__  = fn.__doc__
        return guarded

    def check(self, tool_name: str, args: dict | None = None) -> bool:
        _gate(self.session, tool_name, args or {})
        return True


# ── 6. CrewAI (no changes needed) ────────────────────────────────────────────

class CrewAIGuard:
    """
    CrewAI guard — no breaking changes in 2026, _run API still valid.

    Example:
        from safetydrift.integrations import CrewAIGuard

        guard     = CrewAIGuard(task_type="communication")
        safe_tool = guard.wrap(SendEmailTool())
    """

    def __init__(self, task_type: str = "default",
                 session: Session | None = None) -> None:
        self.session = session or Session(task_type=task_type)

    def wrap(self, tool: Any) -> Any:
        original_run = tool._run
        _session     = self.session
        tool_name    = getattr(tool, "name", type(tool).__name__)

        def guarded_run(*args, **kwargs):
            _gate(_session, tool_name, kwargs or {})
            return original_run(*args, **kwargs)

        tool._run = guarded_run
        return tool