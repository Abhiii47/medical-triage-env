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
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
API_KEY: str      = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK: str     = "medical-triage-env"
TEMPERATURE: float = 0.0
MAX_TOKENS: int    = 512

TASKS = [
    {"id": "easy",    "name": "STEMI Triage",              "max_steps": 15, "success_threshold": 0.60},
    {"id": "medium",  "name": "Sepsis + Opioid Overdose",  "max_steps": 20, "success_threshold": 0.45},
    {"id": "hard",    "name": "Mass Casualty",             "max_steps": 25, "success_threshold": 0.30},

]

SYSTEM_PROMPT = """You are a senior emergency room triage nurse with 20 years of clinical experience. You make fast, accurate decisions that save lives.

## YOUR GOAL
For each patient, complete this clinical protocol IN ORDER to maximize your score:
1. assess    → always assess first to reveal full clinical picture
2. order_test → order the key diagnostic (ECG for chest pain, CT Scan for trauma/stroke, Tox Screen for overdose, Blood Test for sepsis/infection)
3. triage    → assign ESI level (1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less-Urgent, 5=Non-Urgent)
4. treat     → administer the correct drug(s)
5. admit     → admit to correct ward OR discharge if minor

## ESI TRIAGE LEVELS
- Level 1 (Resuscitation): STEMI, Hemorrhagic Shock, Opioid Overdose, Status Asthmaticus — immediate life threat
- Level 2 (Emergent): Sepsis, Stroke — high risk, rapid deterioration
- Level 3-5: Less critical presentations

## CORRECT TREATMENTS & WARDS
- STEMI → Aspirin + admit Cardiology
- Sepsis (NO Penicillin Allergy) → Antibiotics (Ceftriaxone/Vancomycin) + admit ICU
- Sepsis (WITH Penicillin Allergy) → Vancomycin or Ceftriaxone ONLY — NEVER Penicillin or Amoxicillin
- Opioid Overdose → Naloxone + admit ICU — NEVER Morphine/Fentanyl/Oxycodone
- Hemorrhagic Shock → Blood Transfusion + IV Fluids + admit Surgery — NEVER Aspirin/Heparin/Warfarin
- Stroke → CT Scan + tPA or Aspirin + admit Neurology
- Status Asthmaticus → Albuterol or Epinephrine + admit ICU
- Ankle Sprain → X-Ray + triage Level 5 + discharge home

## FATAL DRUG INTERACTIONS (NEVER DO THESE — -0.40 penalty each)
- Penicillin Allergy patient → Penicillin or Amoxicillin
- Opioid Overdose patient → Morphine, Fentanyl, or Oxycodone
- Hemorrhagic Shock patient → Aspirin, Heparin, or Warfarin

## EFFICIENCY RULE
Complete each patient's full protocol as fast as possible. Every unnecessary 'wait' costs points.
If multiple patients are present, prioritize the most critical (Level 1) first.

## OUTPUT FORMAT
Respond ONLY with a single valid JSON object — no explanation, no markdown, no extra text:
{"action_type": "assess", "patient_id": "P-101"}
{"action_type": "order_test", "patient_id": "P-101", "target": "ECG"}
{"action_type": "triage", "patient_id": "P-101", "target": "1"}
{"action_type": "treat", "patient_id": "P-101", "target": "Aspirin"}
{"action_type": "admit", "patient_id": "P-101", "target": "Cardiology"}
{"action_type": "wait"}"""


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: dict, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = f" error={error}" if error else " error=null"
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={str(done).lower()}{error_str}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={json.dumps([round(r, 4) for r in rewards])}", flush=True)


def _pick_priority_patient(obs: dict) -> Optional[str]:
    """Return the patient_id of the most critical untreated patient.
    Priority: lowest O2 → highest HR → first in queue → first in any bed."""
    candidates = []

    # Collect from queue
    for p in obs.get("queue_summary", []):
        hr_str = str(p.get("vitals", {}).get("HR", "80"))
        o2_str = str(p.get("vitals", {}).get("O2", "100%")).replace("%", "")
        try:
            hr = int(hr_str.split("/")[0])
        except ValueError:
            hr = 80
        try:
            o2 = int(o2_str)
        except ValueError:
            o2 = 100
        candidates.append((o2, -hr, p["id"]))

    # Collect from active beds (skip already admitted/treated)
    for bed_name, p in obs.get("active_beds_summary", {}).items():
        if not p or p == "Empty":
            continue
        # Patients in active_beds_summary have not yet been admitted/discharged,
        # so they still need attention even if treated/triaged.
        hr_str = str(p.get("vitals", {}).get("HR", "80"))
        o2_str = str(p.get("vitals", {}).get("O2", "100%")).replace("%", "")
        try:
            hr = int(hr_str.split("/")[0])
        except ValueError:
            hr = 80
        try:
            o2 = int(o2_str)
        except ValueError:
            o2 = 100
        candidates.append((o2, -hr, p["id"]))

    if not candidates:
        return None
    # Sort: lowest O2 first; tie-break by highest HR (stored as negative)
    candidates.sort()
    return candidates[0][2]


def build_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    queue    = obs.get("queue_summary", [])
    beds     = obs.get("active_beds_summary", {})
    alerts   = obs.get("alerts", [])
    feedback = obs.get("action_feedback", "")
    hist_str = "\n".join(history[-8:]) if history else "None yet"

    priority_id = _pick_priority_patient(obs)
    priority_hint = (
        f"⚡ NEXT PRIORITY PATIENT: {priority_id} — act on this patient first unless already completed."
        if priority_id else "All patients appear processed."
    )

    return f"""=== Step {step} ===
Last Feedback: {feedback}
Last Reward: {last_reward:+.4f}

{priority_hint}

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

    history:          List[str]   = []
    rewards:          List[float] = []
    steps_taken:      int         = 0
    score:            float       = 0.0
    success:          bool        = False
    last_reward:      float       = 0.0
    _last_final_score: Optional[float] = None  # grader output from server when done=True

    try:
        resp = http.post(f"{ENV_BASE_URL}/reset", json={"difficulty": task_id, "seed": 42}, timeout=30)
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

            # Capture grader score when the server signals episode end.
            # On done=True, /step returns reward=<grader_score> (env.py line 92).
            # Only store it when there was no HTTP error on this step.
            if done and not error_msg:
                _last_final_score = reward

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error_msg)
            history.append(
                f"Step {step}: {action.get('action_type')}({action.get('patient_id', '')}"
                f" {action.get('target', '')}) → reward={reward:+.4f}"
            )

            if done:
                break

        # When done=True the server returns the deterministic grader score as
        # the reward on that final step (confirmed in env.py step() line 91-92).
        # We also capture it from the `final_score` key in the JSON response if
        # the server exposed it, to be robust against HTTP errors on the last step.
        if rewards:
            score = max(0.0, min(1.0, rewards[-1]))
        else:
            score = 0.0

        # Override with explicit final_score from the last obs if available
        # (the FastAPI /step endpoint embeds reward=final_score when done=True)
        if _last_final_score is not None:
            score = max(0.0, min(1.0, _last_final_score))

        # Always fetch /state to grab the final score as a robust fallback
        try:
            state_resp = http.get(f"{ENV_BASE_URL}/state", timeout=10)
            if state_resp.status_code == 200:
                final_state = state_resp.json()
                fetched_score = final_state.get("score")
                if fetched_score is not None:
                    score = max(0.0, min(1.0, float(fetched_score)))
        except Exception as e:
            print(f"[DEBUG] Could not fetch final score from /state: {e}", flush=True)

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
