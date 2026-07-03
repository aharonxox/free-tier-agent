"""
app.py  -  Flask web server for free-tier-agent
================================================
Provides:
  GET  /          -> index.html (chat UI)
  POST /run       -> start agent loop, stream SSE events back to browser
  GET  /health    -> simple health check

Streaming works via Server-Sent Events (SSE):
  - Browser opens EventSource('/run')
  - Server sends 'data: ...' lines for each step / token
  - Browser appends them to the UI in real-time
"""

import os
import json
import queue
import threading
import logging

from flask import Flask, Response, request, render_template, jsonify
from dotenv import load_dotenv

# Import the core agent runner and state type
from agent_system import run_agent, AgentState, log as agent_log

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health-check endpoint for uptime monitors."""
    return jsonify({"status": "ok"})


@app.route("/run", methods=["GET", "POST"])
def run():
    """
    Start the agent loop and stream progress back as SSE.

    Accepts:
      - GET  ?goal=<your goal>
      - POST JSON {"goal": "<your goal>"}

    Streams events:
      step    -> {step: int, type: 'planner'|'executor'|'summarizer', text: str}
      tool    -> {name: str, result: str}
      done    -> {final_output: str, steps: int}
      error   -> {message: str}
    """
    # Parse goal
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        goal = body.get("goal", "").strip()
    else:
        goal = request.args.get("goal", "").strip()

    if not goal:
        return jsonify({"error": "No goal provided"}), 400

    # We run the agent in a background thread and feed events into a queue
    # the SSE generator reads from.
    event_queue: queue.Queue = queue.Queue()

    def _run_agent_thread():
        """
        Background thread: runs the agent loop and puts SSE events onto
        event_queue so the SSE generator can forward them to the browser.
        """
        try:
            # Monkey-patch agent log handler to also push to queue
            class QueueHandler(logging.Handler):
                def emit(self, record):
                    msg = self.format(record)
                    # Only surface [PLANNER], [EXECUTOR], [SUMMARIZER], [TOOL], [ROUTER]
                    for tag in ("[PLANNER]", "[EXECUTOR]", "[SUMMARIZER]", "[TOOL]", "[ROUTER]"):
                        if tag in msg:
                            node = tag.strip("[]").lower()
                            event_queue.put(("step", {
                                "type": node,
                                "text": msg,
                            }))
                            break

            qh = QueueHandler()
            qh.setLevel(logging.INFO)
            agent_log.addHandler(qh)

            final_state: AgentState = run_agent(goal)

            agent_log.removeHandler(qh)

            event_queue.put(("done", {
                "final_output": final_state.get("final_output", ""),
                "steps": final_state.get("step_counter", 0),
            }))
        except Exception as exc:
            log.exception("Agent error")
            event_queue.put(("error", {"message": str(exc)}))
        finally:
            event_queue.put(None)  # sentinel

    thread = threading.Thread(target=_run_agent_thread, daemon=True)
    thread.start()

    def _stream():
        """SSE generator: reads from event_queue and yields SSE strings."""
        # Send a 'start' event immediately so the browser knows the stream began
        yield _sse("start", {"goal": goal})

        while True:
            item = event_queue.get()
            if item is None:  # sentinel = agent finished
                break
            event_name, data = item
            yield _sse(event_name, data)

    return Response(
        _stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # host=0.0.0.0 required for Replit / cloud environments
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
