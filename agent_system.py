"""
free-tier-agent: agent_system.py
=================================
Full agentic loop with:
  1. LiteLLM Router  - SDK-level + manual failover across free-tier models
  2. LangGraph State - TypedDict shared memory, no direct model-to-model talk
  3. Context Compressor - summarizer node that rewrites scratchpad each loop
  4. Tool Node       - dummy calculator + web-search stub

Quickstart:
  cp .env.example .env   # fill in your real keys
  pip install -r requirements.txt
  python agent_system.py
"""

import os
import re
import logging
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import litellm
from litellm import completion

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
console = Console()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("free-tier-agent")

# Silence litellm's very chatty internal logger unless debugging
if LOG_LEVEL != "DEBUG":
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 1.  ROUTER LAYER  (LiteLLM with aggressive failover)
# ---------------------------------------------------------------------------

# ---- Model registry --------------------------------------------------------
# Add / remove models here.  Keys used by LiteLLM match their provider prefix
# convention: https://docs.litellm.ai/docs/providers
#
#   google/gemini-*   -> needs GEMINI_API_KEY
#   groq/*            -> needs GROQ_API_KEY
#   mistral/*         -> needs MISTRAL_API_KEY
# ---------------------------------------------------------------------------

MODEL_REGISTRY: List[dict] = [
    {
        "id": "primary",
        "model": "gemini/gemini-1.5-flash",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
    },
    {
        "id": "fallback_1",
        "model": "groq/llama3-8b-8192",
        "api_key": os.getenv("GROQ_API_KEY", ""),
    },
    {
        "id": "fallback_2",
        "model": "mistral/mistral-small-latest",
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
    },
]

# Build the list LiteLLM expects for its native fallback param
# Format: [{"model": "...", "api_key": "..."}, ...] minus the primary
LITELLM_FALLBACKS: List[dict] = [
    {"model": m["model"], "api_key": m["api_key"]}
    for m in MODEL_REGISTRY[1:]
]


class RouterError(Exception):
    """All models in the failover chain have been exhausted."""


def call_llm(prompt: str, max_tokens: int = 600) -> str:
    """
    Central LLM call.  Failover strategy (two layers):

    LAYER 1 - LiteLLM native fallbacks
    -----------------------------------
    Pass `fallbacks=LITELLM_FALLBACKS` so the SDK automatically retries
    with the next model on *any* non-success response (429, 5xx, etc.).
    This is the fastest path and requires zero extra code per-model.

    LAYER 2 - Explicit manual loop  <-- FAILOVER HAPPENS HERE
    -----------------------------------------------------------
    If LiteLLM's SDK-level fallback itself raises (e.g. all fallbacks
    also 429'd), we iterate MODEL_REGISTRY manually, passing the exact
    same `prompt` to each model until one succeeds or all are exhausted.

    Context-window failures are caught separately: LiteLLM raises
    `ContextWindowExceededError`; we route immediately to the next model
    instead of retrying (larger context won't help the same model).
    """
    primary = MODEL_REGISTRY[0]

    # ---- LAYER 1: SDK-level failover ----------------------------------------
    try:
        log.debug("Trying primary model: %s", primary["model"])
        response = completion(
            model=primary["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            api_key=primary["api_key"],
            # LiteLLM walks this list automatically on failure
            fallbacks=LITELLM_FALLBACKS,
            num_retries=1,
            timeout=60,
        )
        model_used = response.get("model", primary["model"])
        log.info("[ROUTER] Success via model: %s", model_used)
        return response["choices"][0]["message"]["content"]

    # ---- LAYER 2: explicit manual loop on total SDK failure -----------------
    except litellm.ContextWindowExceededError as exc:
        # Context too large for primary + all fallbacks tried by SDK.
        # Re-try manually skipping primary (it already failed).
        log.warning("[ROUTER] Context window exceeded on SDK chain: %s", exc)

    except Exception as exc:
        # SDK-level fallback exhausted or unexpected error
        log.warning("[ROUTER] SDK-level fallback exhausted: %s", exc)

    # Manual fallback sweep
    last_exc: Optional[Exception] = None
    for cfg in MODEL_REGISTRY:
        try:
            log.info("[ROUTER][MANUAL FAILOVER] Trying: %s", cfg["model"])
            resp = completion(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
                api_key=cfg["api_key"],
                num_retries=1,
                timeout=60,
            )
            log.info("[ROUTER][MANUAL FAILOVER] Success: %s", cfg["model"])
            return resp["choices"][0]["message"]["content"]

        except litellm.RateLimitError as exc:
            # ---- 429 FAILOVER POINT ----
            # Do NOT raise; just skip to next model with the same prompt.
            log.warning(
                "[ROUTER] 429 RateLimit on %s - routing to next model",
                cfg["model"],
            )
            last_exc = exc
            continue

        except litellm.ContextWindowExceededError as exc:
            # ---- CONTEXT-WINDOW FAILOVER POINT ----
            # Skip to next model; retrying same model won't help.
            log.warning(
                "[ROUTER] Context window exceeded on %s - routing to next model",
                cfg["model"],
            )
            last_exc = exc
            continue

        except Exception as exc:
            log.warning("[ROUTER] Error on %s: %s", cfg["model"], exc)
            last_exc = exc
            continue

    raise RouterError(
        f"All {len(MODEL_REGISTRY)} models exhausted. Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# 2.  AGENTIC STATE  (LangGraph TypedDict)
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    Central shared memory for the entire agent loop.

    RULES:
      - Every node reads from this dict and returns an updated copy.
      - Models NEVER communicate directly with each other.
      - All data flows: node -> state -> next node.

    Fields
    ------
    user_goal     : The original task given by the user. Never mutated.
    current_step  : Short label of what the agent is doing right now.
    scratchpad    : Intermediate reasoning + tool outputs.  Gets compressed
                   after each loop iteration by the summarizer node.
    final_output  : Latest condensed answer; always safe to show the user.
    step_counter  : How many planner iterations have run.
    tool_calls    : List of tool calls made this iteration (cleared each loop).
    done          : True when the loop_router decides we are finished.
    """

    user_goal: str
    current_step: str
    scratchpad: str
    final_output: str
    step_counter: int
    tool_calls: List[str]
    done: bool


# ---------------------------------------------------------------------------
# 3.  TOOLS
# ---------------------------------------------------------------------------
# Each tool is a plain Python function.  The executor node detects TOOL: lines
# in the LLM output and dispatches to the right function.
# Add real tools here (web search, DB lookup, code executor, etc.)
# ---------------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """
    Safe arithmetic evaluator.
    Supports: integers, floats, +  -  *  /  //  **  %  ( )
    """
    allowed = re.compile(r"^[\d\s\.\+\-\*\/\%\(\)\*\*]+$")
    expr = expression.strip()
    if not allowed.match(expr):
        return f"[calculator] Rejected unsafe expression: {expr!r}"
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307
        return f"[calculator] {expr} = {result}"
    except Exception as exc:
        return f"[calculator] Error evaluating {expr!r}: {exc}"


def tool_web_search(query: str) -> str:
    """
    Web search stub.  Replace the body with a real search API call
    (e.g. SerpAPI, Brave Search, Tavily) when you have a key.
    """
    # --- stub ---
    log.info("[TOOL:web_search] Query: %s", query)
    return (
        f"[web_search] (stub) No live results for: {query!r}. "
        "Plug in a real search API in tool_web_search() to enable this."
    )


TOOL_REGISTRY = {
    "calculator": tool_calculator,
    "web_search": tool_web_search,
}


def dispatch_tools(llm_output: str) -> List[str]:
    """
    Parse all TOOL: lines from llm_output and execute them.

    Expected format the LLM should use:
        TOOL:calculator 123 * 45
        TOOL:web_search latest Python 3.13 features

    Returns a list of result strings.
    """
    results = []
    for line in llm_output.splitlines():
        line = line.strip()
        if not line.upper().startswith("TOOL:"):
            continue
        rest = line[5:].strip()  # strip 'TOOL:'
        # First token is the tool name, rest is the argument
        parts = rest.split(None, 1)
        if not parts:
            continue
        tool_name = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if tool_name in TOOL_REGISTRY:
            log.info("[TOOL] Dispatching %s(%r)", tool_name, argument)
            result = TOOL_REGISTRY[tool_name](argument)
            results.append(result)
        else:
            results.append(f"[tool] Unknown tool: {tool_name!r}")
    return results


# ---------------------------------------------------------------------------
# 4.  LANGGRAPH NODES
# ---------------------------------------------------------------------------
# Node contract: receive AgentState, return updated AgentState.
# No node ever calls another node directly.
# ---------------------------------------------------------------------------

MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))

SYSTEM_PROMPT = """
You are a meticulous AI agent working step-by-step toward a user's goal.

Rules:
- Think carefully before each step.
- If you need to do arithmetic, emit a TOOL line:
    TOOL:calculator <expression>
- If you need to look something up, emit a TOOL line:
    TOOL:web_search <query>
- After tool results are available, use them in your reasoning.
- When you believe the goal is fully achieved, start your response with
  DONE: followed by your final answer.
- Otherwise start with CONTINUE: and explain the next step.
""".strip()


# ---- Node 1: Planner -------------------------------------------------------

def planner_node(state: AgentState) -> AgentState:
    """
    Planner node.

    Reads:  user_goal, scratchpad (compressed summary), step_counter
    Writes: scratchpad (appended), current_step, tool_calls (reset)

    Asks the LLM: given the goal and what we know so far, what is the
    single best next action?  Handles TOOL: lines for calculator/search.
    """
    step = state.get("step_counter", 0)
    goal = state.get("user_goal", "")
    scratch = state.get("scratchpad", "")

    prompt = f"""{SYSTEM_PROMPT}

===  USER GOAL  ===
{goal}

===  WORK SO FAR (compressed)  ===
{scratch if scratch else '(none yet)'}

===  INSTRUCTION  ===
This is step {step + 1}.  What is the single best next action?
Remember to use TOOL: lines if you need calculations or searches.
"""

    log.info("[PLANNER] Calling LLM for step %d", step + 1)
    llm_output = call_llm(prompt)
    log.debug("[PLANNER] LLM output:\n%s", llm_output)

    # Run any tool calls embedded in the planner output
    tool_results = dispatch_tools(llm_output)
    tool_block = ""
    if tool_results:
        tool_block = "\n\n[TOOL RESULTS]\n" + "\n".join(tool_results)

    new_scratch = (
        scratch
        + f"\n\n--- Step {step + 1} [planner] ---\n"
        + llm_output
        + tool_block
    )

    return {
        **state,
        "current_step": f"step_{step + 1}_planned",
        "scratchpad": new_scratch,
        "tool_calls": tool_results,
        "step_counter": step + 1,
    }


# ---- Node 2: Executor ------------------------------------------------------

def executor_node(state: AgentState) -> AgentState:
    """
    Executor node.

    Reads:  scratchpad, user_goal
    Writes: scratchpad (appended with execution result)

    Asks the LLM to synthesize the planner reasoning + tool results into
    a concrete partial answer or next mini-step.  Also dispatches any
    additional TOOL: lines the executor model adds.
    """
    goal = state.get("user_goal", "")
    scratch = state.get("scratchpad", "")
    step = state.get("step_counter", 0)

    prompt = f"""{SYSTEM_PROMPT}

===  USER GOAL  ===
{goal}

===  WORK SO FAR (including planner output + tool results)  ===
{scratch}

===  INSTRUCTION  ===
Review the planner output and any tool results above.
Synthesize them into a clear partial answer or confirm the next step.
If more tools are needed now, use TOOL: lines.
If the goal is now fully achieved, start with DONE:.
Otherwise start with CONTINUE:.
"""

    log.info("[EXECUTOR] Calling LLM for step %d execution", step)
    llm_output = call_llm(prompt)
    log.debug("[EXECUTOR] LLM output:\n%s", llm_output)

    extra_tools = dispatch_tools(llm_output)
    tool_block = ""
    if extra_tools:
        tool_block = "\n\n[EXECUTOR TOOL RESULTS]\n" + "\n".join(extra_tools)

    new_scratch = (
        scratch
        + f"\n\n--- Step {step} [executor] ---\n"
        + llm_output
        + tool_block
    )

    return {
        **state,
        "current_step": f"step_{step}_executed",
        "scratchpad": new_scratch,
    }


# ---- Node 3: Summarizer / Context Compressor -------------------------------

def summarizer_node(state: AgentState) -> AgentState:
    """
    Context Compressor node.   <-- CONTEXT COMPRESSION HAPPENS HERE

    Problem it solves:
        After each planner+executor iteration, the scratchpad grows.
        If left uncompressed, by iteration 3+ the prompt will exceed
        the context window of free-tier models (typically 8k-32k tokens)
        causing a cascading failure where EVERY subsequent model fails too.

    Solution:
        After each loop iteration, this node calls call_llm() with a
        summarisation prompt.  The resulting short bullet summary REPLACES
        the full scratchpad.  The next planner iteration starts with a
        compact ~300-word brief instead of an ever-growing log.

    Reads:  scratchpad, user_goal, step_counter
    Writes: scratchpad (REPLACED with short summary), final_output
    """
    scratch = state.get("scratchpad", "")
    goal = state.get("user_goal", "")
    step = state.get("step_counter", 0)

    if not scratch.strip():
        return state

    prompt = f"""You are a summarizer inside an agentic loop.

The agent is working toward this goal:
{goal}

Here is the full scratchpad from step {step} (reasoning + tool outputs):

{scratch}

Your task:
1. Distil the scratchpad into a concise bullet-point summary (max 8 bullets).
2. Include all key facts, numbers, and tool results that will still be
   needed in future steps.
3. Keep it under 350 words total.
4. Do NOT include meta-commentary; just the facts and findings.
"""

    log.info("[SUMMARIZER] Compressing scratchpad at step %d", step)
    summary = call_llm(prompt, max_tokens=450)
    log.debug("[SUMMARIZER] Summary:\n%s", summary)

    return {
        **state,
        # ---- OVERWRITE bulky scratchpad with compact summary ----
        "scratchpad": f"[Summary after step {step}]\n{summary}",
        "final_output": summary,
        "current_step": f"step_{step}_summarized",
    }


# ---- Node 4: Loop Router ---------------------------------------------------

def loop_router_node(state: AgentState) -> AgentState:
    """
    Decides whether to loop back to the planner or terminate.

    Logic (in priority order):
    1. If any executor output started with DONE: -> mark done=True.
    2. If step_counter >= MAX_STEPS -> mark done=True (hard cap).
    3. Otherwise -> done=False, continue loop.
    """
    scratch = state.get("scratchpad", "")
    step = state.get("step_counter", 0)

    # Check if executor declared DONE
    executor_done = bool(re.search(r"(?i)^DONE:", scratch, re.MULTILINE))

    if executor_done:
        log.info("[LOOP ROUTER] Executor declared DONE at step %d", step)
        done = True
    elif step >= MAX_STEPS:
        log.info("[LOOP ROUTER] Max steps (%d) reached - stopping", MAX_STEPS)
        done = True
    else:
        log.info("[LOOP ROUTER] Continuing to step %d", step + 1)
        done = False

    return {**state, "done": done, "current_step": "finished" if done else "continue"}


# Conditional edge function (used by LangGraph add_conditional_edges)
def _route(state: AgentState) -> str:
    return "END" if state.get("done", False) else "planner"


# ---------------------------------------------------------------------------
# 5.  GRAPH WIRING  (LangGraph StateGraph)
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble the LangGraph workflow.

    Topology:

        START
          |
          v
       [planner]  <---------+
          |                  |
          v                  |
       [executor]            |  (loop back if not done)
          |                  |
          v                  |
       [summarizer]          |
          |                  |
          v                  |
       [loop_router] --------+
          |
          v (done=True)
         END

    Key design rules enforced here:
    - Models share state only through AgentState (TypedDict).
    - No node holds a reference to another node.
    - The conditional edge is the ONLY place that decides loop vs stop.
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("loop_router", loop_router_node)

    # Linear flow within each iteration
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", "loop_router")

    # Conditional: loop back or terminate
    graph.add_conditional_edges(
        "loop_router",
        _route,
        {
            "planner": "planner",
            "END": END,
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# 6.  PUBLIC API
# ---------------------------------------------------------------------------

def run_agent(user_goal: str) -> AgentState:
    """
    Run the full agentic loop for a given user_goal.

    Returns the final AgentState.  The most useful fields:
      result["final_output"]  - latest compressed answer
      result["step_counter"]  - how many iterations were needed
      result["done"]          - True if agent completed or hit MAX_STEPS
    """
    app = build_graph()

    initial: AgentState = {
        "user_goal": user_goal,
        "current_step": "start",
        "scratchpad": "",
        "final_output": "",
        "step_counter": 0,
        "tool_calls": [],
        "done": False,
    }

    console.print(
        Panel(
            f"[bold cyan]Goal:[/bold cyan] {user_goal}",
            title="[bold green]free-tier-agent starting[/bold green]",
            border_style="green",
        )
    )

    final_state: AgentState = app.invoke(initial)

    console.print(
        Panel(
            final_state.get("final_output", "(no output)"),
            title=f"[bold green]Done after {final_state.get('step_counter', 0)} step(s)[/bold green]",
            border_style="green",
        )
    )

    return final_state


# ---------------------------------------------------------------------------
# 7.  CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        # Default demo goal that exercises the calculator tool
        goal = (
            "I run a small SaaS. I have 123 paying users at $45/month. "
            "Calculate my monthly revenue, annual revenue, and tell me "
            "what % of annual revenue I keep if my costs are $18,000/year."
        )

    result = run_agent(goal)

    # Print raw final state for debugging
    console.rule("[bold]Final State[/bold]")
    console.print(f"[yellow]step_counter:[/yellow] {result.get('step_counter')}")
    console.print(f"[yellow]done:[/yellow]         {result.get('done')}")
    console.print(f"[yellow]current_step:[/yellow] {result.get('current_step')}")
    console.rule()
