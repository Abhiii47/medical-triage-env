from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel


class TriageAction(BaseModel):
    action_type: str
    patient_id: Optional[str] = None
    target: Optional[str] = None


class TriageObservation(BaseModel):
    episode_id: str = ""
    queue_summary: list = []
    active_beds_summary: dict = {}
    alerts: list = []
    current_step: int = 0
    max_steps: int = 0
    action_feedback: str = ""
    reward: float = 0.0
    done: bool = False


class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class EpisodeState(BaseModel):
    episode_id: str = ""
    step: int = 0
    max_steps: int = 0
    done: bool = False
    difficulty: str = "easy"
    patients_in_queue: int = 0
    patients_in_beds: int = 0
    fatal_errors: int = 0
    alerts: list = []


class MedicalTriageEnvClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("ENV_BASE_URL", "http://localhost:7860")).rstrip("/")
        self._timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MedicalTriageEnvClient":
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    def sync(self) -> "_SyncWrapper":
        return _SyncWrapper(self)

    async def reset(self, difficulty: str = "easy", **kwargs) -> TriageObservation:
        resp = await self._require_client().post("/reset", json={"difficulty": difficulty, **kwargs})
        resp.raise_for_status()
        data = resp.json()
        return TriageObservation(**{k: data.get(k, v) for k, v in TriageObservation().model_dump().items()})

    async def step(self, action: TriageAction) -> StepResult:
        resp = await self._require_client().post("/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        reward = float(data.pop("reward", 0.0))
        done = bool(data.pop("done", False))
        obs = TriageObservation(**{k: data.get(k, v) for k, v in TriageObservation().model_dump().items()})
        obs.reward = reward
        obs.done = done
        return StepResult(observation=obs, reward=reward, done=done, info=data.get("info", {}))

    async def state(self) -> EpisodeState:
        resp = await self._require_client().get("/state")
        resp.raise_for_status()
        return EpisodeState(**resp.json())

    async def health(self) -> Dict[str, Any]:
        resp = await self._require_client().get("/health")
        resp.raise_for_status()
        return resp.json()

    def _require_client(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("Use `async with MedicalTriageEnvClient(...) as env:`")
        return self._http


class _SyncWrapper:
    def __init__(self, async_client: MedicalTriageEnvClient):
        self._async = async_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "_SyncWrapper":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async.__aenter__())
        return self

    def __exit__(self, *args) -> None:
        if self._loop:
            self._loop.run_until_complete(self._async.__aexit__(*args))
            self._loop.close()

    def reset(self, difficulty: str = "easy", **kwargs) -> TriageObservation:
        return self._loop.run_until_complete(self._async.reset(difficulty=difficulty, **kwargs))

    def step(self, action: TriageAction) -> StepResult:
        return self._loop.run_until_complete(self._async.step(action))

    def state(self) -> EpisodeState:
        return self._loop.run_until_complete(self._async.state())

    def health(self) -> Dict[str, Any]:
        return self._loop.run_until_complete(self._async.health())


async def _demo(url: str, difficulty: str) -> None:
    print(f"\nMedical Triage Env — Demo [{url} difficulty={difficulty}]\n")
    async with MedicalTriageEnvClient(base_url=url) as env:
        print(f"Health: {await env.health()}\n")
        obs = await env.reset(difficulty=difficulty)
        print(f"Episode: {obs.episode_id}  Max steps: {obs.max_steps}  Queue: {len(obs.queue_summary)} patient(s)")

        if obs.queue_summary:
            pid = obs.queue_summary[0]["id"]
            actions = [
                TriageAction(action_type="assess", patient_id=pid),
                TriageAction(action_type="order_test", patient_id=pid, target="ECG"),
                TriageAction(action_type="triage", patient_id=pid, target="1"),
                TriageAction(action_type="treat", patient_id=pid, target="Aspirin"),
                TriageAction(action_type="admit", patient_id=pid, target="Cardiology"),
            ]
            for i, action in enumerate(actions, 1):
                result = await env.step(action)
                print(f"  [STEP {i}] {action.action_type}({action.patient_id}, {action.target}) -> reward={result.reward:+.4f} done={result.done}")
                if result.done:
                    break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("ENV_BASE_URL", "http://localhost:7860"))
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    asyncio.run(_demo(args.url, args.difficulty))


if __name__ == "__main__":
    main()
