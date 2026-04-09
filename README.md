# 🏥 Medical Triage AI: Winner Edition 🏆

Welcome to the **Medical Triage AI Environment**, a high-fidelity Reinforcement Learning benchmark developed for the **Meta PyTorch OpenEnv Hackathon**. This environment simulates an Emergency Department surge where an AI agent must prioritize, diagnose, and treat patients under intense pressure.

## 🌟 Elite Features (San Francisco Parity)

This environment has been architected to meet the highest standards of the OpenEnv framework, drawing inspiration from winning projects like *CARLA* and *Reasoning Gym*.

- **🧠 Clinical Chain-of-Thought**: Every decision made by the agent includes a `clinical_logic` trace, providing full transparency into the AI's medical reasoning.
- **📈 Stochastic Hemodynamics**: Patient vitals are not static. They deteriorate pathologically based on the underlying condition and include 0.5% physiological "jitter" for realistic simulation.
- **🛡️ Formal Triage Rubric**: A professional-grade audit system that scores based on Triage Accuracy (20%), Diagnostics (20%), Stabilization (30%), and Disposition (30%).
- **📟 Structured Telemetry**: Uses nested Pydantic models for vitals, allowing for precise tracking of Heart Rate, Blood Pressure, and Oxygen saturation.

## 📁 Repository Structure

```text
├── models.py         # Structured Pydantic telemetry & state models
├── simulator.py     # Physio-engine with pathological deterioration logic
├── tasks.py         # Clinical scenarios (STEMI, Sepsis, Stroke, ER Surge)
├── grader.py        # TriageRubric clinical audit framework
├── inference.py     # Llama-3.1 expert agent with clinical protocol fallback
└── server/          # FastAPI production environment wrapper
```

## 🚑 Scenarios & Challenges

| Task | Complexity | Key Conditions | Clinical Goal |
| :--- | :--- | :--- | :--- |
| **Easy** | Single Patient | STEMI | Rapid ECG & Cardiology Admit |
| **Medium** | Multi-Patient | Sepsis, Overdose | Contraindication avoidance (Penicillin) |
| **Hard** | Trauma Surge | Hemorrhagic Shock | Stabilization before disposition |
| **Chaotic** | ER Overload | Mixed Criticality | Dynamic prioritization of Level 1 ESI |

## 🚀 Deployment & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
python server/app.py  # Starts the hospital simulation server
```

### 2. Expert Inference
The agent utilizes **Llama 3.1 8B** via the Hugging Face Inference Router. Ensure your `HF_TOKEN` is set in the `.env` file.

```bash
python inference.py
```

## ⚖️ Clinical Audit Logic
Our `grader.py` doesn't just check if a patient was discharged. It audits the **entire clinical pathway**:
1. **Was the correct ESI Level assigned?**
2. **Was the indicated diagnostic test ordered?** (e.g., ECG for chest pain).
3. **Was the patient stabilized?** (e.g., Naloxone for overdose).
4. **Is the disposition appropriate?** (e.g., ICU for Sepsis).

---
*Created for the Meta PyTorch Hackathon x Scaler School of Technology.* 🩺✨
