"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
===================================
MANDATORY env variables before submitting:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://api.openai.com/v1)
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face API key.

Run:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

import os
import json
import textwrap
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from client import TrafficControlEnv
from models import TrafficAction, TrafficObservation

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN:     str = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
SEED       = 42
MAX_TOKENS = 30
TEMPERATURE = 0.0

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

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

def get_llm_action(obs: TrafficObservation) -> TrafficAction:
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
        phase = max(0, min(2, phase))
    except Exception:
        phase = obs.current_phase
    return TrafficAction(light_phase=phase)

def run_task(task_id: str):
    print(f"[START] task={task_id} env=traffic-control model={MODEL_NAME}")
    
    rewards = []
    success = False
    
    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(task_id=task_id, seed=SEED)
            step = 1
            
            while not step_result.done:
                obs = step_result.observation
                error_msg = "null"
                
                try:
                    action = get_llm_action(obs)
                except Exception as e:
                    error_msg = str(e).replace('"', "'").replace('\\', '')
                    action = TrafficAction(light_phase=0)
                
                action_str = f"TrafficAction(light_phase={action.light_phase})"
                
                try:
                    step_result = env.step(action)
                    done_str = "true" if step_result.done else "false"
                    reward_val = step_result.reward if step_result.reward is not None else 0.0
                    rewards.append(reward_val)
                    print(f"[STEP] step={step} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}")
                except Exception as e:
                    env_error = str(e).replace('"', "'").replace('\\', '')
                    print(f"[STEP] step={step} action={action_str} reward=0.00 done=true error={env_error}")
                    break
                
                step += 1
            
            success = True
    except Exception:
        success = False
        
    success_str = "true" if success else "false"
    num_steps = len(rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={success_str} steps={num_steps} rewards={rewards_str}")

def main() -> None:
    tasks = ["basic_flow", "emergency_priority", "dynamic_scenarios"]
    for task in tasks:
        run_task(task)

if __name__ == "__main__":
    main()
