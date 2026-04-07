from typing import Dict, Any, List
from models import IncidentState, Patient, IncidentAction, IncidentObservation
from tasks import EVALUATIONS_DB, INTERACTIONS_DB

class Simulator:
    def __init__(self, state: IncidentState):
        self.state = state
        self.action_feedback = ""
        self.completed: dict = {}  

    def step(self, action: IncidentAction) -> None:
        self.state.current_step += 1
        self.action_feedback = f"Action {action.action_type} executed."
        
        if action.action_type == "wait":
            self.action_feedback = "Waited for 1 step."
            self._update_time()
            return
            
        if not action.patient_id:
            self.action_feedback = "Error: target patient_id required."
            return
            
        patient = self._get_patient(action.patient_id)
        if not patient:
            self.action_feedback = f"Error: Patient {action.patient_id} not found."
            return

        p_id = patient.id
        
        if action.action_type == "assess":
            self.action_feedback = f"Assessed Patient {p_id}. Symptoms: {patient.symptoms}. Vitals: {patient.vitals}. History: {patient.history}."
            
        elif action.action_type == "order_test":
            if action.target in EVALUATIONS_DB:
                patient.tests_ordered.append(action.target)
                result = EVALUATIONS_DB[action.target].get(patient.hidden_condition, "Normal.")
                patient.test_results[action.target] = result
                self.action_feedback = f"Ordered {action.target} for {p_id}. Result: {result}"
            else:
                self.action_feedback = f"Unknown test: {action.target}"
                
        elif action.action_type == "treat":
            treatment = action.target or "Unknown"
            patient.treatments_given.append(treatment)

            # Check for fatal drug interactions
            for condition, bad_drugs in INTERACTIONS_DB.items():
                if condition in patient.history or condition == patient.hidden_condition:
                    if treatment in bad_drugs:
                        self.state.fatal_errors.append(f"Fatal Interaction triggered for {p_id}: {treatment} contraindicated for {condition}.")
                        patient.is_stable = False

            self.action_feedback = f"Administered {treatment} to {p_id}."
            
        elif action.action_type == "triage":
            try:
                level = int(action.target)
                patient.triage_level = level
                self.action_feedback = f"Triaged {p_id} as Level {level}."
            except (ValueError, TypeError):
                self.action_feedback = f"Invalid triage level: {action.target}"

        elif action.action_type == "admit":
            ward = action.target or "General"
            patient.admitted_ward = ward
            patient.discharged = True
            self.action_feedback = f"Admitted {p_id} to {ward}."
            
        elif action.action_type == "discharge":
            patient.discharged = True
            self.action_feedback = f"Discharged {p_id} home."
            
        else:
            self.action_feedback = f"Unknown action type: {action.action_type}"

        if patient and patient.discharged:
            self._move_to_completed(patient)

        self._update_time()

    def _move_to_completed(self, patient: Patient):
        self.completed[patient.id] = patient
        if patient in self.state.queue:
            self.state.queue.remove(patient)
        for bed, p in self.state.active_beds.items():
            if p and p.id == patient.id:
                self.state.active_beds[bed] = None

    def _get_patient(self, pid: str) -> Patient:
        for p in self.state.queue:
            if p.id == pid:
                return p
        for bed, p in self.state.active_beds.items():
            if p and p.id == pid:
                return p
        
        if pid in self.completed:
            return self.completed[pid]
        return None

    def _update_time(self):
        
        CRITICAL_CONDITIONS = {
            "STEMI": {"O2": -2, "HR": +5},            # O2 drops, HR climbs
            "Sepsis": {"HR": +8, "Temp_delta": +0.3},  # tachycardia + fever worsens
            "Hemorrhagic Shock": {"HR": +10, "BP_sys": -10},  # pressure crashes fast
            "Status Asthmaticus": {"O2": -3, "HR": +6},
            "Opioid Overdose": {"O2": -4},
        }

        all_patients = list(self.state.queue)
        for p in self.state.active_beds.values():
            if p:
                all_patients.append(p)

        for patient in all_patients:
            cond = patient.hidden_condition
            if cond not in CRITICAL_CONDITIONS:
                continue
            
            if patient.treatments_given or patient.admitted_ward:
                continue

            deltas = CRITICAL_CONDITIONS[cond]
            new_vitals = dict(patient.vitals)

            if "O2" in deltas:
                try:
                    o2 = int(new_vitals.get("O2", "95%").replace("%", ""))
                    o2 = max(60, o2 + deltas["O2"])
                    new_vitals["O2"] = f"{o2}%"
                    if o2 < 80:
                        self.state.alerts.append(f"⚠️ CRITICAL: {patient.id} O2 is {o2}%!")
                except ValueError:
                    pass

            if "HR" in deltas:
                try:
                    hr = int(new_vitals.get("HR", "80").split("/")[0])
                    hr = min(200, hr + deltas["HR"])
                    new_vitals["HR"] = str(hr)
                    if hr > 150:
                        self.state.alerts.append(f"⚠️ CRITICAL: {patient.id} HR is {hr} bpm!")
                except ValueError:
                    pass

            if "BP_sys" in deltas:
                try:
                    bp = new_vitals.get("BP", "120/80").split("/")
                    sys = max(40, int(bp[0]) + deltas["BP_sys"])
                    new_vitals["BP"] = f"{sys}/{bp[1]}"
                    if sys < 60:
                        self.state.alerts.append(f"⚠️ CRITICAL: {patient.id} BP crashing: {sys}mmHg!")
                except ValueError:
                    pass

            
            patient.vitals_history.append(dict(patient.vitals))
            patient.vitals = new_vitals

        
        empty_beds = [b for b, p in self.state.active_beds.items() if p is None]
        for b in empty_beds:
            if self.state.queue:
                self.state.active_beds[b] = self.state.queue.pop(0)

        if self.state.current_step >= self.state.max_steps:
            self.state.is_done = True
            self.state.alerts.append("TIME LIMIT REACHED. Shift summary required.")

       
        active_count = len(self.state.queue) + sum(1 for p in self.state.active_beds.values() if p)
        if active_count == 0:
            self.state.is_done = True
            self.state.alerts.append("All patients processed. Shift ended.")
            
    def get_observation(self) -> IncidentObservation:
        q_sum = [{"id": p.id, "vitals": p.vitals, "symptoms": p.symptoms[:2]} for p in self.state.queue]
        bed_sum = {}
        for b, p in self.state.active_beds.items():
            if p:
                bed_sum[b] = {
                    "id": p.id,
                    "vitals": p.vitals,
                    "stable": p.is_stable,
                    "triage_level": p.triage_level,
                    "tests_done": p.tests_ordered,
                    "treatments": p.treatments_given
                }
            else:
                bed_sum[b] = "Empty"

        return IncidentObservation(
            episode_id=self.state.episode_id,
            queue_summary=q_sum,
            active_beds_summary=bed_sum,
            alerts=self.state.alerts[-5:],  
            current_step=self.state.current_step,
            max_steps=self.state.max_steps,
            action_feedback=self.action_feedback
        )
