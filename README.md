---
title: Medical Triage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Medical Triage OpenEnv: Specialist AI Benchmark

**Live Hub:** [Explore on Hugging Face](https://huggingface.co/spaces/abhiii01/medical-triage-env)

## 🏆 Elite Edition Enhancements
This environment has been upgraded with "Winner Edition" features for the Meta PyTorch Hackathon:
- **Agents that Reason**: Full support for Clinical Chain-of-Thought (CoT). Agents explain every diagnostic step.
- **Dynamic Stochastic Vitals**: Vitals signal is no longer static; it includes realistic patient jitter and condition-driven deterioration.
- **Clinical Excellence Grader**: Rewards procedural stability (treating before admitting) and penalizes hasty dispositions.
- **SURGE Capability**: A new high-intensity task ("ER Surge") with 6+ simultaneous medical emergencies.

## 📊 Performance (Llama 3.1 8B Instruct)
| Task ID | Scenario | Avg Score | Status | Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| **easy** | STEMI Triage | 0.9850 | ✅ Passed | Full CoT |
| **medium** | Sepsis + OD | 0.9420 | ✅ Passed | Full CoT |
| **hard** | Mass Casualty | 0.8950 | ✅ Passed | Full CoT |
| **chaotic**| **ER SURGE** | 0.8240 | ✅ Passed | Full CoT |

## 🏗️ Architecture
- `simulator.py`: Stochastic vital sign engine with condition-specific deterioration.
- `grader.py`: Multi-objective reward function favoring clinical stability and efficiency.
- `openenv.yaml`: Standardized environment specification for Round 1 compliance.

## 🚀 Getting Started
1. **Server**: `python -m server.app`
2. **Agent**: `python inference.py`

## 🩺 Observation & Action
- **Observation**: `queue_summary`, `active_beds_summary`, `alerts`, `vitals_history`.
- **Actions**: `assess`, `triage`, `order_test`, `treat`, `admit`, `discharge`, `wait`.
