import os
import sys
import json
import time
from typing import List, Optional

import httpx
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
API_KEY: str      = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK: str     = "medical-triage-env"
TEMPERATURE: float = 0.0
MAX_TOKENS: int    = 512

TASKS = [
    {"id": "easy",   "name": "STEMI Triage",                                               "max_steps": 15, "success_threshold": 0.60},
    {"id": "medium", "name": "Sepsis + Opioid Overdose",                                   "max_steps": 20, "success_threshold": 0.45},
    {"id": "hard",   "name": "Mass Casualty — Hemorrhagic Shock, Stroke, Asthmatic Child", "max_steps": 25, "success_threshold": 0.30},
]

SYSTEM_PROMPT = """You are an expert emergency room triage nurse making rapid clinical decisions.

Available actions:
  assess      - patient_id required
  order_test  - patient_id + target (ECG, Blood Test, CT Scan, X-Ray, Tox Screen)
  triage      - patient_id + target (1=Resuscitation 2=Emergent 3=Urgent 4=Less Urgent 5=Non-Urgent)
  treat       - patient_id + target (drug name: Aspirin, Naloxone, IV Fluids, Albuterol...)
  admit       - patient_id + target (ward: Cardiology, ICU, Neurology, Surgery, General)
  discharge   - patient_id required
  wait        - no parameters

NEVER give Penicillin/Amoxicillin to Penicillin Allergy patients.
NEVER give Morphine/Fentanyl/Oxycodone to Opioid Overdose patients.
NEVER give Aspirin/Heparin/Warfarin to Hemorrhagic Shock patients.

Respond ONLY with valid JSON: {"action_type": "...", "patient_id": "P-XXX", "target": "..."}
For wait: {"action_type": "wait"}"""


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: dict, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = f" error={error}" if error else " error=null"
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={str(done).lower()}{error_str}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={json.dumps([round(r, 4) for r in rewards])}", flush=True)


def build_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    queue    = obs.get("queue_summary", [])
    beds     = obs.get("active_beds_summary", {})
    alerts   = obs.get("alerts", [])
    feedback = obs.get("action_feedback", "")
    hist_str = "\n".join(history[-8:]) if history else "None yet"

    return f"""=== Step {step} ===
Last Feedback: {feedback}
Last Reward: {last_reward:+.4f}

WAITING QUEUE:
{json.dumps(queue, indent=2) if queue else 'Empty'}

ACTIVE BEDS:
{json.dumps(beds, indent=2)}

ALERTS (last 5):
{chr(10).join(alerts) if alerts else 'None'}

RECENT ACTIONS:
{hist_str}

Choose your next action as JSON:"""


def get_action(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> dict:
    prompt = build_prompt(step, obs, last_reward, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
    return {"action_type": "wait"}


def run_task(client: OpenAI, http: httpx.Client, task: dict) -> float:
    task_id    = task["id"]
    task_name  = task["name"]
    max_steps  = task["max_steps"]
    success_th = task["success_threshold"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    last_reward: float       = 0.0

    try:
        resp = http.post(f"{ENV_BASE_URL}/reset", json={"difficulty": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            action    = get_action(client, step, obs, last_reward, history)
            error_msg = None

            try:
                step_resp = http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
                step_resp.raise_for_status()
                obs = step_resp.json()
            except Exception as e:
                error_msg = str(e)
                print(f"[DEBUG] Step {step} HTTP error: {e}", flush=True)
                obs = {"done": True, "reward": 0.0, "current_step": step, "action_feedback": "error"}

            reward      = float(obs.get("reward", 0.0))
            done        = bool(obs.get("done", False))
            last_reward = reward
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error_msg)
            history.append(
                f"Step {step}: {action.get('action_type')}({action.get('patient_id', '')}"
                f" {action.get('target', '')}) → reward={reward:+.4f}"
            )

            if done:
                break

        score   = max(0.0, min(1.0, rewards[-1])) if rewards else 0.0
        success = score >= success_th

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] HF_TOKEN not set.\n"
            "  Get a free token at huggingface.co/settings/tokens\n"
            "  Add it to .env as: HF_TOKEN=hf_your_token_here",
            flush=True,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] HF Router  : {API_BASE_URL}", flush=True)
    print(f"[INFO] Model      : {MODEL_NAME}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Running {len(TASKS)} tasks...\n", flush=True)

    all_scores: List[float] = []

    with httpx.Client() as http:
        try:
            health = http.get(f"{ENV_BASE_URL}/health", timeout=10)
            print(f"[INFO] Health check: {health.json()}", flush=True)
        except Exception as e:
            print(f"[WARN] Health check failed: {e}", flush=True)

        for task in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"[INFO] Task: {task['name']} (difficulty={task['id']})", flush=True)
            print(f"{'='*60}", flush=True)

            score = run_task(client, http, task)
            all_scores.append(score)
            print(f"[INFO] Score: {score:.4f}", flush=True)
            time.sleep(1)

    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] easy={all_scores[0]:.4f}  medium={all_scores[1]:.4f}  hard={all_scores[2]:.4f}", flush=True)
    print(f"[INFO] Average: {sum(all_scores)/len(all_scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
