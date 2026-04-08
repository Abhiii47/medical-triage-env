# Medical Triage OpenEnv

## Environment Overview & Motivation
**Medical Triage OpenEnv** is an intensive, real-world task simulation built for the Meta PyTorch Hackathon. 

In real-world emergency rooms, nurses must make split-second prioritization and diagnostic decisions. They must balance bed occupancy limits, track deteriorating vitals dynamically, and administer treatments adhering to strict safety protocols (such as allergy cross-checking). 

This reinforcement learning environment encapsulates these exact pressures. It moves beyond standard "toy problems" by placing the agent in a highly dynamic, time-sensitive system where poor sequencing or inaction leads to fatal outcomes and massive grading penalties.

## Space Definitions

### Observation Space
The observation space is a heavily typed JSON schema (`IncidentObservation`) mapping instantaneous ER metrics.
- `queue_summary`: A list of patients waiting for beds. Tracks `time_in_queue` and recent `vitals_delta`. 
- `active_beds_summary`: A dictionary of available physical beds, showing which patient occupies them, their assigned `triage_level`, ordered tests, and administered treatments.
- `alerts`: System notifications logging critical vital warnings (e.g., "O2 dropped to 82%!") or dynamic surges in the queue.
- `current_step` / `max_steps`: Action cycle constraints.
- `action_feedback`: Textual result of the prior action submitted by the agent.

### Action Space
The agent submits a discrete `IncidentAction` JSON payload containing:
- `action_type`: Action verb (`assess`, `triage`, `order_test`, `treat`, `admit`, `discharge`, `wait`).
- `patient_id`: Target of the action.
- `target`: Action variant parameter (e.g., specifying `IV Fluids` for `treat`, or `Cardiology` for `admit`).

## Task Descriptions
The environment exposes 4 benchmark tasks via the OpenEnv API.

1. **STEMI Triage (Easy)**
   - *Goal*: Process a single patient arriving with chest pain.
   - *Challenge*: Simple pattern matching. Order ECG, prescribe Aspirin, and admit to Cardiology efficiently.

2. **Sepsis + Opioid Overdose (Medium)**
   - *Goal*: Process two concurrent critical patients.
   - *Challenge*: Sepsis patient has a marked Penicillin Allergy natively in their history. The agent must successfully navigate alternative antibiotics without triggering a fatal shock while keeping the overdose patient stabilized.

3. **Mass Casualty (Hard)**
   - *Goal*: Three simultaneous high-acuity patients (Hemorrhagic Shock, Stroke, Status Asthmaticus).
   - *Challenge*: Saturated ER conditions. The agent must flawlessly execute parallel triage pathways, avoiding blood thinners for the bleeding patient while accelerating tPA for the stroke patient.

## Reward Mechanics
* **Step Shaping (+):** The agent gains intermediate rewards for prioritizing fast assessments strings on rapidly deteriorating patients (`+0.05`), correctly executing preliminary tests before drugs (`+0.04`), and standard progressions (`+1` for discharge).
* **Penalties (-):** Wait penalties (`-0.01`) and catastrophic protocol violations result in heavy subtraction (`-0.15` per fatal error).
* **Terminal Evaluation:** The episode closes natively returning the `grader.py` deterministic fractional grade bounded between `0.0` and `1.0`.

## Setup and Usage

**Docker Execution**
The environment perfectly matches Hugging Face Spaces architectures:
```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```
The FastAPI instance will natively start at `http://localhost:7860`.

**Direct Execution**
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline Performance Scores
A complete baseline inference script leveraging `meta-llama/Llama-3.2-3B-Instruct` is included (`inference.py`). Simply export your HF API key to execute:

```bash
export HF_TOKEN="your_token_here"
python inference.py
```

**Baseline Performance Scores (meta-llama/Llama-3.2-3B-Instruct):**
- `easy`: ~0.85 *(Acceptable OpenEnv Range: 0.70 - 1.0)*
- `medium`: ~0.55 *(Acceptable OpenEnv Range: 0.40 - 0.80)*
- `hard`: ~0.35 *(Acceptable OpenEnv Range: 0.20 - 0.65)*
