"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
===================================
MANDATORY env variables before submitting:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face API key.

Run:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

import os
import json
import textwrap
from typing import Optional

# Load .env file automatically so users don't need to set env vars manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars

from openai import OpenAI

from client import TrafficControlEnv
from models import TrafficAction, TrafficObservation

# ---------------------------------------------------------------------------
# Config — required env variables per hackathon rules
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN:     str = os.getenv("HF_TOKEN", "")
OPENAI_KEY:   str = os.getenv("OPENAI_API_KEY", "")

# Pick the right API key based on the endpoint being used
# HuggingFace router accepts HF_TOKEN; OpenAI accepts OPENAI_API_KEY
if "huggingface" in API_BASE_URL.lower() or "hf.co" in API_BASE_URL.lower():
    API_KEY = HF_TOKEN or OPENAI_KEY
else:
    API_KEY = OPENAI_KEY or HF_TOKEN

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
SEED       = 42
MAX_TOKENS = 30
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an Autonomous Traffic Control AI managing a 4-way intersection.

    OBJECTIVE: Maximise vehicle throughput and prioritise emergency vehicles.

    PHASES:
      0 = North-South Green   (vehicles from N/S may pass)
      1 = East-West Green     (vehicles from E/W may pass)
      2 = All Red             (no vehicles pass — use only when necessary)

    STRATEGY:
      1. If any emergency vehicles are waiting, immediately switch to their direction.
      2. Otherwise, switch to the direction with the most queued vehicles.
      3. Do not change phases too quickly — enforce at least 4 steps in a phase.

    OUTPUT: Reply with a single JSON object — no markdown, no explanation:
      {"light_phase": <0, 1, or 2>}
""").strip()


def build_user_prompt(obs: TrafficObservation) -> str:
    q    = obs.queue_lengths
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency
    return textwrap.dedent(f"""
        CURRENT STATE:
          Active phase     : {obs.current_phase}
          Steps in phase   : {obs.time_in_phase}
          Regular queue    : N={q[0]}, S={q[1]}, E={q[2]}, W={q[3]}
          Emergency queue  : N={em_q[0]}, S={em_q[1]}, E={em_q[2]}, W={em_q[3]}
          Emergency urgency: N={em_u[0]}, S={em_u[1]}, E={em_u[2]}, W={em_u[3]}

        Output one JSON object like: {{"light_phase": 0}}
    """).strip()


def get_llm_action(client: OpenAI, obs: TrafficObservation) -> TrafficAction:
    """Ask the LLM for the next traffic phase. Falls back gracefully."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        data  = json.loads(response.choices[0].message.content or "{}")
        phase = int(data.get("light_phase", obs.current_phase))
        phase = max(0, min(2, phase))   # clamp to valid range
    except Exception as exc:
        print(f"    [LLM error] {exc} — keeping current phase")
        phase = obs.current_phase
    return TrafficAction(light_phase=phase)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> Optional[float]:
    """Run one full episode and return a 0–1 grade score."""
    print(f"\n{'='*55}")
    print(f" Task: {task_id}")
    print(f"{'='*55}")

    with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
        step_result = env.reset(task_id=task_id, seed=SEED)
        step = 0

        while not step_result.done:
            obs    = step_result.observation
            action = get_llm_action(client, obs)
            step_result = env.step(action)
            step  += 1

            if step % 20 == 0:
                print(
                    f"  Step {step:3d} | phase {action.light_phase}"
                    f" | reward {step_result.reward:+.2f}"
                )

        state = env.state()
        print(f"\n  [Done after {step} steps]")
        print(f"  Vehicles passed    : {state.total_vehicles_passed}")
        print(f"  Emergency passed   : {state.total_emergency_passed}")
        print(f"  Total waiting time : {state.total_waiting_time:.1f}")
        print(f"  Collisions         : {state.total_collisions}")
        print(f"  Phase changes      : {state.total_phase_changes}")
        return state


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Autonomous Traffic Control – LLM Inference")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API base   : {API_BASE_URL}")
    print(f"  Environment: {SERVER_URL}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    tasks  = ["basic_flow", "emergency_priority", "dynamic_scenarios"]

    for task in tasks:
        run_task(client, task)

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
