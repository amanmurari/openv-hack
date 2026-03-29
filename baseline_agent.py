"""
Rule-based baseline agent for the Autonomous Traffic Control environment.

Uses the openenv-core EnvClient (via TrafficControlEnv) for HTTP interaction.
Demonstrates the correct sync API pattern.

Run:
    # 1. Start the server (in another terminal):
    uvicorn traffic_control_env.server.app:app --port 8000

    # 2. Run this agent:
    python -m traffic_control_env.baseline_agent
    python -m traffic_control_env.baseline_agent --task emergency_priority
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from client import TrafficControlEnv
from models import TrafficAction, TrafficObservation
from models import (
    PHASE_NS_GREEN, PHASE_EW_GREEN, PHASE_ALL_RED,
    PHASE_NS_YELLOW, PHASE_EW_YELLOW,
)


# ---------------------------------------------------------------------------
# Rule-based policy
# ---------------------------------------------------------------------------

class RuleBasedAgent:
    """
    Simple heuristic agent:
      1. If any emergency vehicle is waiting, switch to their direction.
      2. Otherwise, switch to the direction with more queued vehicles.
      3. Enforce a minimum green time to avoid flicker.
    """

    MIN_GREEN_STEPS = 4

    def __init__(self) -> None:
        self._steps_in_green = 0
        self._last_phase     = PHASE_NS_GREEN

    def act(self, obs: TrafficObservation) -> TrafficAction:
        current = obs.current_phase

        # During yellow/transition, keep the current phase request
        if current in (PHASE_NS_YELLOW, PHASE_EW_YELLOW):
            return TrafficAction(light_phase=self._last_phase)

        self._steps_in_green += 1

        # 1. Emergency override
        em_q   = obs.emergency_queue
        em_urg = obs.emergency_urgency
        ns_em  = em_q[0] + em_q[1]        # North + South
        ew_em  = em_q[2] + em_q[3]        # East  + West
        ns_urg = max(em_urg[0], em_urg[1]) if ns_em else 0
        ew_urg = max(em_urg[2], em_urg[3]) if ew_em else 0

        if ns_em > 0 or ew_em > 0:
            if ns_urg >= ew_urg and current != PHASE_NS_GREEN:
                self._last_phase     = PHASE_NS_GREEN
                self._steps_in_green = 0
                return TrafficAction(light_phase=PHASE_NS_GREEN)
            elif ew_urg > ns_urg and current != PHASE_EW_GREEN:
                self._last_phase     = PHASE_EW_GREEN
                self._steps_in_green = 0
                return TrafficAction(light_phase=PHASE_EW_GREEN)

        # Enforce minimum green time
        if self._steps_in_green < self.MIN_GREEN_STEPS:
            return TrafficAction(light_phase=self._last_phase)

        # 2. Queue-length balancing
        q          = obs.queue_lengths
        ns_q       = q[0] + q[1]
        ew_q       = q[2] + q[3]
        want_phase = PHASE_NS_GREEN if ns_q >= ew_q else PHASE_EW_GREEN

        if want_phase != current:
            self._last_phase     = want_phase
            self._steps_in_green = 0

        return TrafficAction(light_phase=want_phase)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run(task_id: str = "basic_flow", seed: Optional[int] = 42, url: str = "http://localhost:8000"):
    agent = RuleBasedAgent()

    # Use the sync wrapper provided by openenv-core's EnvClient
    with TrafficControlEnv(base_url=url).sync() as env:
        print(f"\n=== Traffic Control Baseline Agent ===")
        print(f"Task: {task_id} | Seed: {seed} | Server: {url}")

        step_result = env.reset(task_id=task_id, seed=seed)
        step = 0

        while not step_result.done:
            obs    = step_result.observation
            action = agent.act(obs)
            step_result = env.step(action)
            obs    = step_result.observation
            step  += 1

            if step % 20 == 0:
                q      = obs.queue_lengths
                em_q   = obs.emergency_queue
                print(
                    f"  Step {step:3d} | Phase {obs.current_phase} | "
                    f"Q [N{q[0]},S{q[1]},E{q[2]},W{q[3]}] | "
                    f"EmQ [N{em_q[0]},S{em_q[1]},E{em_q[2]},W{em_q[3]}] | "
                    f"Reward {step_result.reward:+.2f}"
                )

        state = env.state()
        print(f"\n=== Episode Ended (step {step}) ===")
        print(f"  Total vehicles:   {state.total_vehicles_passed}")
        print(f"  Emergency passed: {state.total_emergency_passed}")
        print(f"  Total waiting:    {state.total_waiting_time:.1f}")
        print(f"  Collisions:       {state.total_collisions}")
        print(f"  Phase changes:    {state.total_phase_changes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based baseline agent")
    parser.add_argument("--task", default="basic_flow",
                        choices=["basic_flow", "emergency_priority", "dynamic_scenarios"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url",  default="http://localhost:8000")
    args = parser.parse_args()

    run(task_id=args.task, seed=args.seed, url=args.url)
