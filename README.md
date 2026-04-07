---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
  - agent-environment
  - reinforcement-learning
  - healthcare
---

# 🏥 medical-triage-env

> **Meta PyTorch Hackathon Submission** · An OpenEnv environment where an AI agent acts as a clinical Emergency Room Triage Nurse, making high-stakes decisions under time pressure with real clinical consequences.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-00a393.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Is This?

This environment simulates a real hospital Emergency Department. The agent receives a queue of patients with authentic vital signs, symptoms, and medical histories, and must:

1. **Assess** patients to gather clinical information.
2. **Order** the correct diagnostic tests (ECG, Blood Test, CT Scan, Tox Screen).
3. **Triage** each patient to the correct ESI level (1 = Resuscitation → 5 = Non-Urgent).
4. **Treat** patients with appropriate medications — **avoiding fatal drug interactions**.
5. **Admit** to the correct ward (Cardiology, ICU, Neurology, Surgery) or discharge home.

A **deterministic grader** computes a final score from **0.0 to 1.0** based strictly on clinical accuracy, resource efficiency, and patient safety.

---

## 🩺 Why Medical Triage?

- **Real-world gap**: No existing OpenEnv models clinical decision-making under high-stakes uncertainty.
- **High stakes**: Wrong actions (e.g., administering Penicillin to an allergic patient) have immediate, irreversible consequences.
- **Partial observability**: Hidden conditions (e.g., internal bleeding) must be discovered through diagnostic testing.
- **Time pressure**: Untreated critical patients deteriorate organically step-by-step (vitals worsen).
- **Excellent RL training signal**: Dense reward shaping + deterministic terminal scoring.

### Why Medical Triage Matters for RL
This environment serves as an excellent sandbox for localized Reinforcement Learning. It features strong partial observability, strict time pressure, dense step-wise reward shaping, and severe penalties for unsafe actions (fatal drug interactions). By training an RL policy in this simulation, researchers can develop models that prioritize safe clinical decision-making protocols and learn true generalization over rote token prediction.

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `episode_id` | `string` | Unique episode identifier. |
| `queue_summary` | `list[dict]` | Patients waiting: id, vitals, top 2 symptoms. |
| `active_beds_summary` | `dict` | Map of bed → patient state (vitals, triage, tests, treatments). |
| `alerts` | `list[string]` | Last 5 system alerts (critical vitals, fatal errors). |
| `current_step` | `int` | Step counter since last reset. |
| `max_steps` | `int` | Maximum steps allowed for this difficulty. |
| `action_feedback` | `string` | Natural language result of the last action taken. |

---

## ⚡ Action Space

| Action Type | `patient_id` | `target` | Description |
|---|---|---|---|
| `assess` | **Required** | — | Reveal full symptoms, vitals, history. |
| `order_test` | **Required** | `ECG`, `Blood Test`, `CT Scan`, `X-Ray`, `Tox Screen` | Run a diagnostic test. |
| `triage` | **Required** | `1`–`5` (string) | Assign ESI triage level (1 to 5). |
| `treat` | **Required** | Drug name (e.g., `Aspirin`, `Naloxone`, `IV Fluids`) | Administer treatment. |
| `admit` | **Required** | `Cardiology`, `ICU`, `Neurology`, `Surgery`, `General` | Admit to the respective ward. |
| `discharge` | **Required** | — | Discharge the patient home. |
| `wait` | — | — | Pass this turn (−0.01 step penalty). |

---

## 🎯 Clinical Tasks

### Task 1 — 🟢 Easy: STEMI Triage
- **Patient**: P-101 (65M) — Crushing chest pain, diaphoresis, left arm radiation.
- **Hidden condition**: STEMI (ST-elevation myocardial infarction).
- **Max steps**: 15
- **Ideal pathway**: `assess` → `order_test` (ECG) → `triage` (1) → `treat` (Aspirin) → `admit` (Cardiology).
- **Expected score range**: 0.70 – 1.0

### Task 2 — 🟡 Medium: Sepsis + Opioid Overdose
- **Patients**: P-102 (78F, Sepsis, **Penicillin Allergy**) + P-108 (28M, Opioid Overdose).
- **Trap**: Giving Penicillin/Amoxicillin to P-102 is a **fatal interaction** (−0.40 penalty).
- **Max steps**: 20
- **Ideal pathway**: Vancomycin/Ceftriaxone for P-102 (ICU) + Naloxone for P-108 (ICU).
- **Expected score range**: 0.40 – 0.80

### Task 3 — 🔴 Hard: Mass Casualty
- **Patients**: P-104 (Hemorrhagic Shock), P-107 (Stroke), P-105 (9yo Status Asthmaticus).
- **Resource constraint**: 3 critical patients, 2 beds — **prioritization is absolutely required**.
- **Traps**: Blood thinners (Aspirin/Heparin) on P-104 are immediately fatal.
- **Time pressure**: Untreated critical patients deteriorate dynamically.
- **Max steps**: 25
- **Ideal pathway**: Prioritize P-104 (Level 1, Surgery) → P-107 (Level 2, CT Scan, Neurology) → P-105 (Level 1, ICU).
- **Expected score range**: 0.20 – 0.65

---

## 🏆 Reward Function

### Step-Level Signals (Partial Progress for RL):
| Action | Reward |
|---|---|
| `assess`, `triage` | **+0.03** |
| `order_test` | **+0.01** |
| `admit` / `discharge` | **+0.05** |
| `treat` (fatal contraindication) | **−0.15** |
| `wait` | **−0.01** |

### Terminal Reward (Deterministic Grader):
| Component | Weight |
|---|---|
| Correct triage level | +0.25 |
| Required diagnostic test | +0.25 |
| Appropriate treatment | +0.25 |
| Correct ward disposition | +0.25 |
| **Fatal drug interaction** | **−0.40 per event** |
| Unnecessary tests | −0.10 each |
| Efficiency bonus (finish early) | Up to +0.15 max |

## 📊 Baseline Scores

Run with `meta-llama/Llama-3.2-3B-Instruct` via Hugging Face Inference API (temperature=0):

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| STEMI Triage | 🟢 Easy | **0.8500** | Correct triage + ECG + Aspirin + Cardiology, but minor step inefficiency. |
| Sepsis + OD | 🟡 Medium | **0.5250** | Correct antibiotics for Sepsis; Naloxone for OD; occasional extra test penalty. |
| Mass Casualty | 🔴 Hard | **0.3200** | Correct prioritization of P-104; partial ward accuracy; avoided fatal drug errors. |

> Scores use the deterministic grader (0.0–1.0). A perfect run on Easy scores **1.0**; these baselines reflect realistic LLM reasoning with some suboptimal steps.

---

## 🤖 Client Library (for RL Frameworks)

`client.py` provides a canonical `MedicalTriageEnvClient` that RL training frameworks can use natively:

```python
from client import MedicalTriageEnvClient, TriageAction

# Async (Recommended — compatible with TRL, torchforge, SkyRL, ART, Oumi)
async with MedicalTriageEnvClient(base_url="http://localhost:7860") as env:
    obs = await env.reset(difficulty="easy")
    result = await env.step(TriageAction(action_type="assess", patient_id="P-101"))
    print(result.reward, result.done)

# Synchronous (via .sync() wrapper)
with MedicalTriageEnvClient(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(difficulty="medium")
```

**Quick Demo:**
```bash
python client.py --url http://localhost:7860 --difficulty easy
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
curl http://localhost:7860/health
```

### Option 2: Local (Without Docker)
```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🧠 Running the Inference Script

> **No OpenAI key needed.** The script uses your Hugging Face token via the [HF Router](https://huggingface.co/docs/inference-providers) to automatically route to the fastest available inference provider.

**1. Configuration**
```bash
cp .env.example .env
# Edit .env and paste your Hugging Face Token (with Inference Provider permissions)
```

**2. Example Configuration (`.env`)**
```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_BASE_URL=http://localhost:7860
ENV_DIFFICULTY=easy
```

**3. Execution**
```bash
python inference.py
```

The script will emit structured tracking logs compliant with OpenEnv formatting:
```text
[INFO] Model      : Qwen/Qwen2.5-72B-Instruct
[INFO] Environment: http://localhost:7860
[START] task=STEMI Triage env=medical-triage-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type": "assess", "patient_id": "P-101"} reward=0.0300 done=false error=null
[STEP] step=2 action={"action_type": "order_test", "patient_id": "P-101", "target": "ECG"} reward=0.0100 done=false error=null
...
[END] success=true steps=8 score=0.8750 rewards=[0.03, 0.01, ...]
```

---

## 🧪 Testing

The environment includes a comprehensive test suite covering the simulation loops, vital signs degradation, drug interactions, and strict OpenEnv contract validation.

```bash
python tests/test_env.py
```

---

## 🌐 API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode `{"difficulty": "easy"\|"medium"\|"hard"}` |
| `POST` | `/step` | Execute an action `{"action_type": "...", "patient_id": "...", "target": "..."}` |
| `GET` | `/state` | Retrieve current episode state (`OpenEnv state()` contract) |
| `GET` | `/tasks` | List all 3 available meta-tasks and success thresholds |
| `GET` | `/health` | Core service health check |
| `GET` | `/ui` | Web-based Real-time Clinical Dashboard |

---

## 📁 Project Structure

```text
medical-triage-env/
├── client.py                 ← OpenEnv EnvClient subclass (for RL frameworks)
├── inference.py              ← Baseline inference script (spec-compliant)
├── models.py                 ← Pydantic models (Action, Observation, TriageState)
├── tasks.py                  ← Scenario definitions & clinical guidelines
├── simulator.py              ← Core state transition engine (vitals decay)
├── grader.py                 ← Deterministic 0.0–1.0 scoring logic
├── openenv.yaml              ← OpenEnv configuration manifest
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← HuggingFace Spaces compatible Dockerfile
├── outputs/                  ← Automated log output directory
├── tests/
│   └── test_env.py           ← Pytest-compliant unit testing suite
└── server/
    ├── env.py                ← MedicalTriageEnv Controller
    └── app.py                ← FastAPI server + Live WebSocket Dashboard
```

---

## 🏥 Clinical Drug Interaction Database

The environment actively monitors for critical contraindicated actions:

| Condition | **Contraindicated Drugs** | Penalty |
|---|---|---|
| Penicillin Allergy | Penicillin, Amoxicillin | **−0.40** per event |
| Opioid Overdose | Morphine, Fentanyl, Oxycodone | **−0.40** per event |
| Hemorrhagic Shock | Aspirin, Heparin, Warfarin | **−0.40** per event |

> *Note: Administering contraindicated drugs immediately destabilizes the patient and logs a severe fault in the final OpenEnv evaluation grade.*

---

## 📜 License

MIT License — See [LICENSE](LICENSE)
