# free-tier-agent

A production-ready agentic loop that aggressively maximizes free-tier LLM APIs using automated failover routing, stateful memory, and context compression.

## Architecture

```
START
  |
  v
[planner]  <---------+
  |                   |
  v                   |
[executor]            |  loop back if not done
  |                   |
  v                   |
[summarizer]          |
  |                   |
  v                   |
[loop_router] --------+
  |
  v  (done=True)
 END
```

### Components

| Layer | Technology | Purpose |
|---|---|---|
| **Router** | LiteLLM | SDK-level + manual failover across free-tier models |
| **State** | LangGraph `TypedDict` | Shared memory — models never talk directly |
| **Compressor** | Summarizer node | Rewrites scratchpad after every loop to prevent context blowup |
| **Tools** | Python functions | Calculator (live) + Web search (stub, plug in your key) |

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/aharonxox/free-tier-agent.git
cd free-tier-agent
pip install -r requirements.txt
```

### 2. Set up your API keys

```bash
cp .env.example .env
# Edit .env and fill in your real keys
```

Free-tier key sources:
- **Gemini Flash** — https://aistudio.google.com/app/apikey
- **Groq / Llama-3** — https://console.groq.com/keys
- **Mistral** — https://console.mistral.ai/api-keys

### 3. Run

```bash
# Default demo goal (SaaS revenue calculator)
python agent_system.py

# Custom goal
python agent_system.py "Research the best Python web frameworks and compare their performance"
```

## Failover Logic

Two layers of failover are implemented in `call_llm()`:

**Layer 1 — LiteLLM SDK native fallbacks:**
Passes `fallbacks=[...]` to `litellm.completion()`. The SDK automatically retries with the next model on any non-2xx response (429, 5xx, timeout).

**Layer 2 — Manual sweep:**
If the entire SDK chain fails, an explicit `for` loop walks `MODEL_REGISTRY` and tries each model with the same prompt. 429s and context-window errors skip to the next model rather than raising.

## Context Compression

The `summarizer_node` runs after every planner+executor iteration. It:
1. Reads the full (potentially large) scratchpad
2. Calls the LLM with a summarisation prompt capped at 450 tokens
3. **Overwrites** `state["scratchpad"]` with the compact bullet summary
4. Updates `state["final_output"]` with the latest answer

This keeps every subsequent prompt well within free-tier context limits regardless of how many loop iterations run.

## Adding Tools

Add a function to `TOOL_REGISTRY` in `agent_system.py`:

```python
def tool_my_tool(argument: str) -> str:
    # your logic here
    return "result string"

TOOL_REGISTRY["my_tool"] = tool_my_tool
```

The LLM will call it by emitting:
```
TOOL:my_tool your argument here
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini free-tier key |
| `GROQ_API_KEY` | — | Groq free-tier key |
| `MISTRAL_API_KEY` | — | Mistral free-tier key |
| `MAX_STEPS` | `5` | Max agentic loop iterations |
| `LOG_LEVEL` | `INFO` | Python logging level |

## File Structure

```
free-tier-agent/
  agent_system.py   # All logic: router, state, nodes, graph, CLI
  requirements.txt  # Python dependencies
  .env.example      # Template for API keys
  README.md         # This file
```
