# DriftGuard 🛡️

**The first open-source implementation of the SafetyDrift framework.**

> *"When an agent reads a confidential file, writes a summary, then emails it externally — no single step is unsafe, but the sequence is a data leak."*
> — SafetyDrift, arXiv:2603.27148 (March 2026)

DriftGuard predicts when individually safe AI agent actions are about to **compound into a safety violation**, and intervenes before it happens. It models agent safety trajectories as absorbing Markov chains — giving you a **P(violation within N steps)** score after every tool call.

---

## The problem it solves

Every major AI agent framework (LangChain, AutoGen, CrewAI, Claude Code) trusts the agent. They check output content. They don't check **accumulated authority** or **trajectory risk**.

| Situation | Traditional guardrails | DriftGuard |
|-----------|----------------------|------------|
| Agent reads a secret | ✅ Allowed | ✅ Allowed (P=low) |
| Agent reads a secret, then opens customer CSV | ✅ Allowed | ⚠️ Warns (P rising) |
| Agent reads a secret, opens CSV, sends email | ✅ Allowed | 🚫 Blocked (P=87%) |

The SafetyDrift paper found: **in communication-capable agents, reaching even a mild risk state gives an 85% probability of a safety violation within 5 steps.** DriftGuard makes that prediction in real time, before the violation occurs.

---

## Quick start

```bash
pip install driftguard
```

```python
from driftguard import Session, InterventionAction

session = Session(task_type="default")  # or: communication, technical, autonomous

# Call before EVERY tool execution
result = session.gate("read_file", {"path": "/workspace/customer_data.csv"})

if result.action == InterventionAction.BLOCK:
    raise RuntimeError(f"DriftGuard blocked: {result.reason}")
elif result.action == InterventionAction.PAUSE:
    approved = ask_human_for_approval(result.to_dict())
    if not approved:
        raise RuntimeError("Human rejected action")

# Safe to proceed
actually_read_file(...)
```

---

## How it works

DriftGuard tracks three cumulative safety dimensions per session:

| Dimension | Description | Levels |
|-----------|-------------|--------|
| **Data Exposure** | Sensitivity of data accessed | None → Public → Internal → Confidential → Sensitive |
| **Tool Escalation** | Capability level reached | None → Read → Write → Network → External |
| **Reversibility** | Can actions be undone? | Fully → Mostly → Mixed → Mostly Not → Irreversible |

State is **monotonic**: it only ever increases. After each tool call, DriftGuard:
1. Classifies the call into risk dimensions
2. Projects the cumulative state forward
3. Runs Markov chain absorption analysis: **P(violation within N steps)**
4. Applies the configured policy (WARN / PAUSE / BLOCK)

### Markov chain model

From the SafetyDrift paper, safety violations follow absorbing Markov chain dynamics:

```
SAFE → LOW → MODERATE → HIGH → CRITICAL → [VIOLATION]
```

Every agent will eventually reach a violation if left unsupervised — the practical question is **when**, not **if**. DriftGuard computes the finite-horizon absorption probability:

```
P(violation | state, horizon) = [T^horizon][state, violation_state]
```

where `T` is a task-type-calibrated transition matrix.

---

## Demo output

```
Step 1: web_search           Risk: MODERATE  P(viol): 31.1%  ✅ WARN
Step 2: read_file (config)   Risk: MODERATE  P(viol): 31.5%  ✅ WARN
Step 3: read_file (customers) Risk: MODERATE P(viol): 31.9%  ✅ WARN
Step 4: write_file (summary) Risk: HIGH      P(viol): 54.8%  ⏸ PAUSE
Step 5: send_email           Risk: CRITICAL  P(viol): 86.8%  🚫 BLOCK
```

Each step looks safe in isolation. The sequence is a data leak. DriftGuard catches it at step 4 (pause) and blocks it at step 5.

---

## Use as an MCP server

DriftGuard ships as a stdio MCP server. Any MCP-compatible agent (Claude Code, Cursor, GitHub Copilot) can call it directly.

**Add to your `mcp.json`:**
```json
{
  "driftguard": {
    "command": "python",
    "args": ["-m", "driftguard"]
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
from driftguard import Session
from driftguard.policy import PolicyConfig, PolicyThreshold
from driftguard.types import InterventionAction

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

### Custom classifier rules

```python
from driftguard import add_rule, ClassifierRule
from driftguard.types import DataExposure, ToolEscalation, Reversibility

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

## Integration examples

### LangChain
```python
from driftguard import Session, InterventionAction

session = Session(task_type="default")

class GuardedTool(BaseTool):
    def _run(self, *args, **kwargs):
        result = session.gate(self.name, kwargs)
        if result.action == InterventionAction.BLOCK:
            raise ToolException(f"DriftGuard: {result.reason}")
        return self._actual_run(*args, **kwargs)
```

### OpenAI Agents SDK
```python
from agents import function_tool
from driftguard import Session, InterventionAction

session = Session(task_type="autonomous")

def guarded(fn):
    def wrapper(**kwargs):
        r = session.gate(fn.__name__, kwargs)
        if r.action == InterventionAction.BLOCK:
            return f"[BLOCKED by DriftGuard: {r.reason}]"
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
2. **Framework adapters** — LangGraph, CrewAI, AutoGen, Google Genkit
3. **CI/CD integration** — GitHub Actions workflow that gates agent PRs

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT — use it in your agent pipelines, commercial or otherwise.

---

*Inspired by and implementing [SafetyDrift (arXiv:2603.27148)](https://arxiv.org/abs/2603.27148). This project is not affiliated with the paper's authors.*
