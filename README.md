# Medical Triage OpenEnv

**Live Demo:** [https://huggingface.co/spaces/abhiii01/medical-triage-env](https://huggingface.co/spaces/abhiii01/medical-triage-env)

## Overview
Medical Triage OpenEnv is a reinforcement learning environment simulating an emergency room triage nurse. Agents must prioritize patients, order diagnostics, administer treatments, and admit patients to appropriate wards—all under time pressure with strict safety protocols.

## Environment

### Observation Space
- `queue_summary`: Patients waiting for beds with vitals and deterioration trends
- `active_beds_summary`: Currently occupied beds with patient status
- `alerts`: Critical warnings (vitals warnings, patient surges)
- `current_step` / `max_steps`: Episode progress
- `action_feedback`: Result of the previous action

### Action Space
- `assess` — Evaluate patient condition
- `triage` — Assign ESI level (1-5)
- `order_test` — Order diagnostic (ECG, Blood Test, CT Scan, X-Ray, Tox Screen)
- `treat` — Administer treatment
- `admit` / `discharge` — Patient disposition
- `wait` — Skip action

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| STEMI Triage | Easy | Single chest pain patient — admit to Cardiology |
| Sepsis + Opioid Overdose | Medium | Two patients — avoid Penicillin allergy, use Naloxone |
| Mass Casualty | Hard | Three critical patients — prioritize Level 1, avoid blood thinners |

## Reward Mechanics
- **Step rewards**: +0.03 assess, +0.01 order_test, +0.03 triage, +0.04 correct treat, +0.05 admit
- **Penalties**: -0.01 wait, -0.15 fatal drug error
- **Terminal score**: Deterministic grader output [0.0–1.0]

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Or with Docker:
```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

## Running Inference

```bash
export HF_TOKEN="your_token_here"
python inference.py
```

## Performance (meta-llama/Llama-3.1-8B-Instruct)

| Task | Score | Threshold |
|------|-------|-----------|
| Easy | 0.9900 | 0.60 |
| Medium | 0.9900 | 0.45 |
| Hard | 0.9100 | 0.30 |
| **Average** | **0.9633** | |

All tasks pass their success thresholds with deterministic, reproducible scores.
