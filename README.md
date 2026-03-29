# Autonomous Traffic Control – OpenEnv Environment

An OpenEnv-compliant Reinforcement Learning environment that simulates a **4-way intersection** where an AI agent controls traffic lights to maximise vehicle throughput and prioritise emergency vehicles.

---

## Overview

| Property | Value |
|---|---|
| **Environment ID** | `traffic-control-env` |
| **Version** | 1.0.0 |
| **API** | OpenEnv `reset / step / state` |
| **Action space** | Discrete – 3 light phases |
| **Observation space** | Structured object (queues, phases, emergency status) |
| **Tasks** | 3 (Easy → Hard) |

---

## Observation Space

Each call to `reset()` or `step()` returns a `TrafficObservation` with these fields:

| Field | Type | Description |
|---|---|---|
| `current_phase` | int (0-4) | Active light phase (see table below) |
| `time_in_phase` | int | Steps elapsed in current phase |
| `queue_lengths` | List[int] × 4 | Regular vehicle queue per approach `[N, S, E, W]` |
| `emergency_queue` | List[int] × 4 | Emergency vehicle count per approach |
| `emergency_urgency` | List[int] × 4 | Max urgency (0-10) of queued emergency vehicles |
| `vehicles_passed` | int | Regular vehicles cleared this step |
| `emergency_passed` | int | Emergency vehicles cleared this step |
| `total_waiting_time` | float | Sum of per-vehicle waiting increments this step |
| `collision` | bool | Gridlock-induced collision flag |
| `reward` | float | Step reward |
| `done` | bool | Episode termination flag |
| `metadata` | dict | `step_count`, `task_id` |

**Phase codes:**

| Code | Name | Description |
|---|---|---|
| 0 | `NS_GREEN` | North + South green, East + West red |
| 1 | `EW_GREEN` | East + West green, North + South red |
| 2 | `ALL_RED` | All approaches red (emergency clearance) |
| 3 | `NS_YELLOW` | N/S transitioning (internal – read-only) |
| 4 | `EW_YELLOW` | E/W transitioning (internal – read-only) |

---

## Action Space

A single integer field `light_phase`:

| Value | Effect |
|---|---|
| `0` | Request NS_GREEN |
| `1` | Request EW_GREEN |
| `2` | Request ALL_RED |

Yellow-light transitions (2 steps) are handled automatically by the environment when switching between NS_GREEN and EW_GREEN.

---

## Reward Function

| Event | Reward |
|---|---|
| Regular vehicle clears intersection | +0.20 |
| Emergency vehicle clears intersection | +10.00 |
| Per-vehicle waiting increment | −0.05 |
| Emergency vehicle waiting (per step, urgency-weighted) | −0.4 × urgency |
| Collision (terminal) | −200.00 |
| Unnecessary phase change (target lane empty) | −0.50 |

---

## Tasks

### Task 1 – Basic Traffic Flow `basic_flow` (Easy)

- Moderate Poisson arrivals (λ = 0.4/direction/step)
- No emergency vehicles
- Episode length: 200 steps
- **Grading (0–1):** 60 % throughput + 40 % efficiency

### Task 2 – Emergency Vehicle Prioritisation `emergency_priority` (Medium)

- Poisson arrivals (λ = 0.5) + 1.5 % emergency probability per direction
- Emergency urgency range: 7–10
- Episode length: 300 steps
- **Grading (0–1):** 35 % throughput + 45 % emergency priority + 20 % efficiency

### Task 3 – Dynamic Scenarios `dynamic_scenarios` (Hard)

- High Poisson arrivals (λ = 0.7) + traffic-surge events + 3.5 % emergency probability
- Emergency urgency range: 8–10
- Episode length: 400 steps
- **Grading (0–1):** 30 % throughput + 40 % emergency priority + 30 % efficiency

All scores are multiplied by `(1 − collision_penalty)`.

---

## Setup

### Local (Python)

```bash
# Clone / enter directory
cd traffic_control_env

# Install
pip install -e .

# Start server
uvicorn traffic_control_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build
docker build -t traffic-control-env .

# Run
docker run -d -p 8000:8000 traffic-control-env

# Health check
curl http://localhost:8000/health
```

### Hugging Face Spaces

Push via the OpenEnv CLI:

```bash
openenv push --repo-id <username>/traffic-control-env
```

The environment will be available at:
- **API**: `https://<username>-traffic-control-env.hf.space`
- **Docs**: `https://<username>-traffic-control-env.hf.space/docs`
- **Docker image**: `registry.hf.space/<username>-traffic-control-env:latest`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state/{session_id}` | Episode-level cumulative state |
| `POST` | `/grade/{session_id}` | Run automated grader (returns 0–1 score) |
| `DELETE` | `/session/{session_id}` | Close a session |

Interactive docs: `http://localhost:8000/docs`

---

## Python Client

```python
from traffic_control_env.client import TrafficControlClient
from traffic_control_env.models import TrafficAction

client = TrafficControlClient("http://localhost:8000")

# Task 2 – emergency priority
obs = client.reset(task_id="emergency_priority", seed=42)

while not obs.done:
    # Your agent logic here – example: always NS green
    action = TrafficAction(light_phase=0)
    obs = client.step(action)

result = client.grade()
print(f"Score: {result['score']:.4f}")
print(f"Feedback: {result['feedback']}")
```

---

## Baseline Agent

A rule-based baseline (fixed-time + emergency override) is provided for benchmarking:

```bash
# Run on all three tasks
python -m traffic_control_env.baseline_agent

# Run on a specific task
python -m traffic_control_env.baseline_agent --task emergency_priority --seed 7

# Suppress step-by-step output
python -m traffic_control_env.baseline_agent --quiet
```

Expected baseline scores (seed=42):

| Task | Approx. Score |
|---|---|
| basic_flow | 0.55 – 0.70 |
| emergency_priority | 0.45 – 0.60 |
| dynamic_scenarios | 0.30 – 0.50 |

RL agents are expected to significantly outperform these baselines, especially on Tasks 2 and 3.

---

## Project Structure

```
traffic_control_env/
├── openenv.yaml          # Environment manifest
├── __init__.py           # Package entry-point
├── models.py             # TrafficAction / TrafficObservation / TrafficState
├── client.py             # HTTP client (type-safe)
├── baseline_agent.py     # Rule-based reference agent
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI application
    ├── traffic_control.py # Core simulation logic
    └── tasks.py          # Task graders (basic_flow, emergency_priority, dynamic_scenarios)
pyproject.toml
Dockerfile
README.md
```
