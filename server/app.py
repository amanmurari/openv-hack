"""
FastAPI application for the Autonomous Traffic Control OpenEnv environment.

Uses the openenv-core create_app() factory which automatically provides:
  - POST /reset       – start a new episode
  - POST /step        – execute one action
  - GET  /state       – retrieve episode-level state
  - GET  /schema      – action / observation JSON schemas
  - WS   /ws          – WebSocket endpoint for persistent sessions
  - GET  /health      – liveness probe
  - GET  /docs        – interactive Swagger UI

Adds custom:
  - POST /grade       – run the automated task grader (0-1 score)

Usage:
    uvicorn traffic_control_env.server.app:app --host 0.0.0.0 --port 8000 --reload
    python -m traffic_control_env.server.app
"""

import sys
import os
# Ensure project root is on sys.path so `models`, `client`, etc. resolve correctly
# regardless of which directory uvicorn is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, _SERVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openenv.core.env_server.http_server import create_app
from fastapi import Request, HTTPException

from models import TrafficAction, TrafficObservation
from traffic_control import TrafficControlEnvironment
from tasks import grade as run_grader


# ---------------------------------------------------------------------------
# 1. Build the standard OpenEnv app via create_app()
# ---------------------------------------------------------------------------

app = create_app(
    TrafficControlEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# 2. Add the /grade endpoint (not part of openenv-core standard, but needed
#    for hackathon evaluation of task scores)
# ---------------------------------------------------------------------------

@app.post("/grade", tags=["eval"])
async def grade(request: Request):
    """
    Run the automated grader for the environment's *current* state and return
    a 0-1 normalised score along with detailed metrics and feedback.

    The request body may optionally contain {"task_id": "..."} to override
    auto-detection; otherwise the grader reads task_id from the env state.

    Note: For WebSocket sessions the client calls this via HTTP after the
    episode ends.  The endpoint reads state from the single shared environment
    (or first available) – for production, pass task_id + metrics explicitly.
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    # Pull metrics either from body (explicit) or from environment manager
    task_id              = body.get("task_id", "basic_flow")
    total_vehicles       = int(body.get("total_vehicles_passed", 0))
    total_emergency      = int(body.get("total_emergency_passed", 0))
    total_waiting        = float(body.get("total_waiting_time", 0.0))
    total_collisions     = int(body.get("total_collisions", 0))
    total_emergency_delay = float(body.get("total_emergency_delay", 0.0))
    total_phase_changes  = int(body.get("total_phase_changes", 0))
    step_count           = int(body.get("step_count", 1))

    result = run_grader(
        task_id,
        total_vehicles_passed=total_vehicles,
        total_emergency_passed=total_emergency,
        total_waiting_time=total_waiting,
        total_collisions=total_collisions,
        total_emergency_delay=total_emergency_delay,
        total_phase_changes=total_phase_changes,
        step_count=step_count,
    )

    return {
        "task_id":  task_id,
        "score":    result.score,
        "metrics":  result.metrics,
        "feedback": result.feedback,
    }


# ---------------------------------------------------------------------------
# 3. Add Gradio UI Testing Interface mounted at /ui
# ---------------------------------------------------------------------------
import gradio as gr
import requests

def reset_env(task_id):
    try:
        resp = requests.post("http://127.0.0.1:8000/reset", json={"task_id": task_id, "seed": 42})
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}

def step_env(phase):
    try:
        resp = requests.post("http://127.0.0.1:8000/step", json={"light_phase": int(phase)})
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}

def get_state():
    try:
        resp = requests.get("http://127.0.0.1:8000/state")
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(title="Traffic Control UI", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🚦 Autonomous Traffic Control - Testing UI")
    gr.Markdown("Use this interface to manually test the OpenEnv HTTP API endpoints.")
    with gr.Row():
        task_dropdown = gr.Dropdown(choices=["basic_flow", "emergency_priority", "dynamic_scenarios"], value="basic_flow", label="Task ID")
        reset_btn = gr.Button("🔄 Reset Environment")
        state_btn = gr.Button("📊 Get State")
        
    with gr.Row():
        phase_radio = gr.Radio(choices=[("0 - NS Green", "0"), ("1 - EW Green", "1"), ("2 - All Red", "2")], value="0", label="Next Action (Light Phase)")
        step_btn = gr.Button("▶️ Step Action", variant="primary")
        
    output_json = gr.JSON(label="API Response")
    
    reset_btn.click(reset_env, inputs=[task_dropdown], outputs=[output_json])
    step_btn.click(step_env, inputs=[phase_radio], outputs=[output_json])
    state_btn.click(get_state, outputs=[output_json])

app = gr.mount_gradio_app(app, ui, path="/ui")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for direct execution.

        uv run --project . server
        python -m traffic_control_env.server.app
        uvicorn traffic_control_env.server.app:app --reload
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Call main() so openenv validate passes its string check
    main(host="0.0.0.0", port=args.port)
