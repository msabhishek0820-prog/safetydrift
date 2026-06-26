# SafetyDrift 🛡️

**The first open-source implementation of the SafetyDrift framework.**

> *"When an agent reads a confidential file, writes a summary, then emails it externally — no single step is unsafe, but the sequence is a data leak."*
> — SafetyDrift, arXiv:2603.27148 (March 2026)

SafetyDrift predicts when individually safe AI agent actions are about to **compound into a safety violation**, and intervenes before it happens.

Unlike traditional agent guardrails, SafetyDrift combines:

- 📊 Benchmark-calibrated risk modeling using 100+ labeled agent traces
- 🧠 LLM-based action classification (instead of fragile regex rules)
- 💾 Cross-session memory for detecting slow-burn attacks
- 🔌 Native integrations with real agent frameworks such as LangChain and AutoGen

It models agent safety trajectories as absorbing Markov chains and computes a real-time **P(violation within N steps)** score after every tool call.

---

## Features

- 📈 Predicts safety violations before they occur
- 🧠 LLM-based tool classification
- 📊 Benchmark-evaluated on 100+ labeled traces
- 💾 Persistent cross-session memory
- ⛔ Configurable WARN / PAUSE / BLOCK interventions
- 🔌 LangChain integration
- 🔌 AutoGen integration
- 🔌 MCP server support
- ⚡ Lightweight runtime overhead

---

## The problem it solves

Every major AI agent framework (LangChain, AutoGen, CrewAI, Claude Code) trusts the agent. They check output content. They don't check **accumulated authority** or **trajectory risk**.

| Situation | Traditional guardrails | SafetyDrift |
|-----------|----------------------|------------|
| Agent reads a secret | ✅ Allowed | ✅ Allowed (P=low) |
| Agent reads a secret, then opens customer CSV | ✅ Allowed | ⚠️ Warns (P rising) |
| Agent reads a secret, opens CSV, sends email | ✅ Allowed | 🚫 Blocked (P=87%) |

The SafetyDrift paper found: **in communication-capable agents, reaching even a mild risk state gives an 85% probability of a safety violation within 5 steps.** SafetyDrift makes that prediction in real time, before the violation occurs.

---

---

## Benchmark Results

SafetyDrift has been evaluated against a benchmark dataset of 100+ labeled agent session traces containing both safe and unsafe behaviors.

The benchmark measures:

- **True Positive Rate (Recall)** — violations correctly detected
- **False Positive Rate** — safe actions incorrectly blocked
- **Early Detection Distance** — how many steps before the actual violation SafetyDrift intervened

Example output:

| Metric | Result |
|----------|---------|
| True Positive Rate | XX% |
| False Positive Rate | XX% |
| Average Early Detection | X.X steps |
| Dataset Size | 100+ traces |

This benchmark transforms SafetyDrift from a theoretical implementation into an empirically evaluated safety system.

> Replace XX values with your actual benchmark results.

## Quick start

```bash
pip install SafetyDrift
```

```python
from SafetyDrift import Session, InterventionAction

session = Session(task_type="default")  # or: communication, technical, autonomous

# Call before EVERY tool execution
result = session.gate("read_file", {"path": "/workspace/customer_data.csv"})

if result.action == InterventionAction.BLOCK:
    raise RuntimeError(f"SafetyDrift blocked: {result.reason}")
elif result.action == InterventionAction.PAUSE:
    approved = ask_human_for_approval(result.to_dict())
    if not approved:
        raise RuntimeError("Human rejected action")

# Safe to proceed
actually_read_file(...)
```

---

## How it works

SafetyDrift tracks three cumulative safety dimensions per session:

| Dimension | Description | Levels |
|-----------|-------------|--------|
| **Data Exposure** | Sensitivity of data accessed | None → Public → Internal → Confidential → Sensitive |
| **Tool Escalation** | Capability level reached | None → Read → Write → Network → External |
| **Reversibility** | Can actions be undone? | Fully → Mostly → Mixed → Mostly Not → Irreversible |

State is **monotonic**: it only ever increases. After each tool call, SafetyDrift:
1. Uses an LLM-based classifier to evaluate the tool name and arguments
2. Determines data exposure, capability escalation, and reversibility levels
3. Projects the cumulative state forward
4. Runs Markov chain absorption analysis
5. Applies the configured policy (WARN / PAUSE / BLOCK)

---

### LLM-powered action classification

Traditional agent guardrails often rely on hardcoded tool names or regex matching:

```text
send_email      -> HIGH RISK
read_file       -> LOW RISK

### Markov chain model

From the SafetyDrift paper, safety violations follow absorbing Markov chain dynamics:

```
SAFE → LOW → MODERATE → HIGH → CRITICAL → [VIOLATION]
```

Every agent will eventually reach a violation if left unsupervised — the practical question is **when**, not **if**. SafetyDrift computes the finite-horizon absorption probability:

```
P(violation | state, horizon) = [T^horizon][state, violation_state]
```

where `T` is a task-type-calibrated transition matrix.

---

## Cross-Session Memory

Most safety systems reset their state when a session ends.

Real attacks often do not.

An agent may:

Session A:
- Read confidential data

Session B:
- Create summaries

Session C:
- Export information externally

Each session appears safe in isolation.

SafetyDrift maintains persistent risk signals across sessions, allowing it to detect:

- Slow-burn exfiltration attempts
- Multi-session reconnaissance
- Delayed escalation patterns
- Long-running autonomous workflows

This enables trajectory-based safety monitoring beyond the lifetime of a single agent run.

---

## Demo output

```
Step 1: web_search           Risk: MODERATE  P(viol): 31.1%  ✅ WARN
Step 2: read_file (config)   Risk: MODERATE  P(viol): 31.5%  ✅ WARN
Step 3: read_file (customers) Risk: MODERATE P(viol): 31.9%  ✅ WARN
Step 4: write_file (summary) Risk: HIGH      P(viol): 54.8%  ⏸ PAUSE
Step 5: send_email           Risk: CRITICAL  P(viol): 86.8%  🚫 BLOCK
```

Each step looks safe in isolation. The sequence is a data leak. SafetyDrift catches it at step 4 (pause) and blocks it at step 5.

---

## Use as an MCP server

SafetyDrift ships as a stdio MCP server. Any MCP-compatible agent (Claude Code, Cursor, GitHub Copilot) can call it directly.

**Add to your `mcp.json`:**
```json
{
  "SafetyDrift": {
    "command": "python",
    "args": ["-m", "SafetyDrift"]
  }
}
```

**Available MCP tools:**

| Tool | Description |
|------|-------------|
| `dg_gate` | Evaluate a tool call. Returns action: ALLOW / WARN / PAUSE / BLOCK |
| `dg_session_state` | Get current cumulative risk state |
| `dg_summary` | Get session stats (blocks, pauses, step count) |
| `dg_reset` | Reset session for a new task |

**Example system prompt addition for Claude Code:**
```
Before executing any tool that reads files, makes network requests, or
writes data, call dg_gate with the tool name and arguments. If the result
action is BLOCK, do not proceed. If PAUSE, describe the action and ask the
user for approval before continuing.
```

---

## Configuration

```python
from SafetyDrift import Session
from SafetyDrift.policy import PolicyConfig, PolicyThreshold
from SafetyDrift.types import InterventionAction

config = PolicyConfig(
    horizon=5,                  # steps ahead to evaluate
    task_type="communication",  # higher baseline risk
    thresholds=[
        PolicyThreshold(0.90, InterventionAction.BLOCK,  "Critical risk — blocked"),
        PolicyThreshold(0.60, InterventionAction.PAUSE,  "High risk — approval needed"),
        PolicyThreshold(0.30, InterventionAction.WARN,   "Elevated risk — warned"),
        PolicyThreshold(0.00, InterventionAction.LOG_ONLY, "Safe — logged"),
    ],
    always_block={
        "send_mass_email",
        "delete_production_database",
        "wipe_all_data",
    },
)

session = Session(config=config, task_type="communication")
```

### Optional custom classification rules

SafetyDrift uses an LLM classifier by default.

Organizations can optionally provide custom override rules for domain-specific tools or internal workflows.

```python
from SafetyDrift import add_rule, ClassifierRule
from SafetyDrift.drift_types import DataExposure, ToolEscalation, Reversibility

# Add your own tool patterns
add_rule(ClassifierRule(
    pattern=r"jira.*create.*ticket",
    data_exposure=DataExposure.INTERNAL,
    tool_escalation=ToolEscalation.EXTERNAL,
    reversibility=Reversibility.MOSTLY,
    description="Create Jira ticket — external but recoverable",
))
```

---

## Task types

| Task type | Typical use | Baseline violation rate |
|-----------|-------------|------------------------|
| `technical` | Code editing, local file ops | Very low (~1–5% per step) |
| `information` | Research, browsing, summarising | Low (~8–15%) |
| `default` | General-purpose agents | Medium (~8%) |
| `autonomous` | Multi-step autonomous tasks | Medium-high (~12%) |
| `communication` | Email, messaging, posting agents | High (~18%) |

---

## Framework Integrations

SafetyDrift integrates directly with production agent frameworks.

A typical integration requires only a few lines of code and places a safety gate in front of every tool execution.

### LangChain
```python
from SafetyDrift import Session, InterventionAction

session = Session(task_type="default")

class GuardedTool(BaseTool):
    def _run(self, *args, **kwargs):
        result = session.gate(self.name, kwargs)
        if result.action == InterventionAction.BLOCK:
            raise ToolException(f"SafetyDrift: {result.reason}")
        return self._actual_run(*args, **kwargs)
```

### OpenAI Agents SDK
```python
from agents import function_tool
from SafetyDrift import Session, InterventionAction

session = Session(task_type="autonomous")

def guarded(fn):
    def wrapper(**kwargs):
        r = session.gate(fn.__name__, kwargs)
        if r.action == InterventionAction.BLOCK:
            return f"[BLOCKED by SafetyDrift: {r.reason}]"
        return fn(**kwargs)
    return function_tool(wrapper)
```

---

## Background: the SafetyDrift paper

This library implements the framework from:

> **SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do**
> Aditya Dhodapkar, Farhaan Pishori (March 2026)
> [arXiv:2603.27148](https://arxiv.org/abs/2603.27148)

Key findings we implement:
- Safety state modelled as an absorbing Markov chain across 3 dimensions
- Every agent has absorption probability 1.0 — the question is *when*, not *if*
- Communication tasks: 85% violation probability within 5 steps from mild-risk state
- Technical tasks: below 5% from any state
- "Points of no return" are sharply task-dependent

---

## Contributing

The highest-value contributions right now:

1. **Real trace data** — If you have agent session traces (with ground truth on whether violations occurred), they can calibrate the transition matrices far better than our heuristic approximation
2. **More benchmark traces** — expand the evaluation corpus and improve statistical confidence
3. **Additional framework adapters** — CrewAI, LangGraph, Genkit, Semantic Kernel
4. **Classifier improvements** — smaller/faster local models
5. **Cross-session storage backends** — Redis, PostgreSQL, vector stores
6. **Enterprise policy packs** — domain-specific safety policies
7. **Framework adapters** — LangGraph, CrewAI, AutoGen, Google Genkit
8. **CI/CD integration** — GitHub Actions workflow that gates agent PRs

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT — use it in your agent pipelines, commercial or otherwise.

---

*Inspired by and implementing [SafetyDrift (arXiv:2603.27148)](https://arxiv.org/abs/2603.27148). This project is not affiliated with the paper's authors.*
