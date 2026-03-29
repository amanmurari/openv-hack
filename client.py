"""
HTTP/WebSocket client for the Autonomous Traffic Control OpenEnv environment.

Extends openenv-core's EnvClient so it works seamlessly with all openenv-core
compatible RL training frameworks (TRL, TorchForge, Unsloth, ART, Oumi, etc.).

Usage (synchronous):
    from traffic_control_env import TrafficControlEnv, TrafficAction

    with TrafficControlEnv(base_url="http://localhost:8000").sync() as client:
        obs = client.reset(task_id="emergency_priority", seed=42)
        while not obs.done:
            action = TrafficAction(light_phase=0)
            obs = client.step(action)
        state = client.state()
        print(state.total_vehicles_passed)

Usage (async):
    import asyncio
    from traffic_control_env import TrafficControlEnv, TrafficAction

    async def main():
        async with TrafficControlEnv(base_url="http://localhost:8000") as client:
            obs = await client.reset(task_id="basic_flow", seed=0)
            while not obs.done:
                obs = await client.step(TrafficAction(light_phase=0))

    asyncio.run(main())
"""

from typing import Any, Dict
from openenv.core.env_client import EnvClient, StepResult

from models import TrafficAction, TrafficObservation, TrafficState


class TrafficControlEnv(EnvClient[TrafficAction, TrafficObservation, TrafficState]):
    """
    Client for the Autonomous Traffic Control environment.

    Inherits all openenv-core EnvClient functionality:
    - async context manager (async with TrafficControlEnv(...) as env: ...)
    - .sync() wrapper for synchronous use
    - reset() / step() / state()
    - from_docker_image() class method for local Docker deployment
    - from_env() class method for HuggingFace Space deployment
    """

    def _step_payload(self, action: TrafficAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrafficObservation]:
        obs = TrafficObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TrafficState:
        return TrafficState(**payload)

