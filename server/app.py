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
