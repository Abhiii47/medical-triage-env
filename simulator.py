import random
from typing import Dict, Any, List
from models import IncidentState, Patient, IncidentAction, IncidentObservation, VitalsTelemetry
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
            for condition, bad_drugs in INTERACTIONS_DB.items():
                if condition in patient.history or condition == patient.hidden_condition:
                    if treatment in bad_drugs:
                        self.state.fatal_errors.append(f"Fatal Interaction triggered for {p_id}: {treatment} contraindicated for {condition}.")
                        patient.is_stable = False
            self.action_feedback = f"Administered {treatment} to {p_id}."
        elif action.action_type == "triage":
            try:
                patient.triage_level = int(action.target)
                self.action_feedback = f"Triaged {p_id} as Level {action.target}."
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
        for p in self.state.active_beds.values():
            if p and p.id == pid:
                return p
        if pid in self.completed:
            return self.completed[pid]
        return None

    def _update_time(self):
        CRITICAL_CONDITIONS = {
            "STEMI": {"O2": -2, "HR": +5},
            "Sepsis": {"HR": +8, "Temp_delta": +0.3},
            "Hemorrhagic Shock": {"HR": +10, "BP_sys": -10},
            "Status Asthmaticus": {"O2": -3, "HR": +6},
            "Opioid Overdose": {"O2": -4},
        }

        all_patients = list(self.state.queue) + [p for p in self.state.active_beds.values() if p]

        for patient in all_patients:
            cond = patient.hidden_condition
            if cond not in CRITICAL_CONDITIONS or patient.treatments_given or patient.admitted_ward:
                pass
            else:
                deltas = CRITICAL_CONDITIONS[cond]
                v: VitalsTelemetry = patient.vitals
                
                if "O2" in deltas:
                    v.o2 = max(60, v.o2 + deltas["O2"])
                    if v.o2 < 80:
                        self.state.alerts.append(f"CRITICAL: {patient.id} O2 is {v.o2}%!")
                
                if "HR" in deltas:
                    v.hr = min(200, v.hr + deltas["HR"])
                    if v.hr > 150:
                        self.state.alerts.append(f"CRITICAL: {patient.id} HR is {v.hr} bpm!")
                
                if "BP_sys" in deltas:
                    v.bp_sys = max(40, v.bp_sys + deltas["BP_sys"])
                    if v.bp_sys < 60:
                        self.state.alerts.append(f"CRITICAL: {patient.id} BP crashing: {v.bp_sys}mmHg!")

            patient.vitals_history.append(patient.vitals.model_copy(deep=True))
            
            # STOCHASTIC NOISE: Add minor random fluctuations (0.5%) to keep environment feeling dynamic
            noise = random.uniform(0.995, 1.005)
            patient.vitals.hr = int(patient.vitals.hr * noise)
            patient.vitals.o2 = min(100, int(patient.vitals.o2 * noise))
            patient.vitals.temp = round(patient.vitals.temp * random.uniform(0.998, 1.002), 1)
            
            # BP Jitter
            patient.vitals.bp_sys = int(patient.vitals.bp_sys * random.uniform(0.99, 1.01))
            patient.vitals.bp_dia = int(patient.vitals.bp_dia * random.uniform(0.99, 1.01))

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

    def _enrich_patient_obs(self, p: Patient) -> dict:
        obs = {
            "id": p.id,
            "vitals": p.vitals.to_dict(),  # Keep strings for the Agent LLM
            "time_in_queue": self.state.current_step - p.arrival_step,
            "deterioration_trend": "stable",
            "vitals_delta": {}
        }
        if len(p.vitals_history) >= 1:
            prev: VitalsTelemetry = p.vitals_history[-1]
            curr: VitalsTelemetry = p.vitals
            delta = {}
            
            d_hr = curr.hr - prev.hr
            if d_hr != 0: delta["HR"] = d_hr
            
            d_o2 = curr.o2 - prev.o2
            if d_o2 != 0: delta["O2"] = d_o2
            
            obs["vitals_delta"] = delta
            if d_hr >= 5 or d_o2 <= -2:
                obs["deterioration_trend"] = "worsening"
            elif d_hr <= -5 or d_o2 >= 2:
                obs["deterioration_trend"] = "improving"

        return obs

    def get_observation(self) -> IncidentObservation:
        q_sum = []
        for p in self.state.queue:
            p_obs = self._enrich_patient_obs(p)
            p_obs["symptoms"] = p.symptoms[:2]
            q_sum.append(p_obs)

        bed_sum = {}
        for b, p in self.state.active_beds.items():
            if p:
                p_obs = self._enrich_patient_obs(p)
                p_obs.update({
                    "stable": p.is_stable,
                    "triage_level": p.triage_level,
                    "tests_done": p.tests_ordered,
                    "treatments": p.treatments_given
                })
                bed_sum[b] = p_obs
            else:
                bed_sum[b] = "Empty"

        # SF BEST PRACTICE: Provide both summary and structured telemetry
        all_patients = list(self.state.queue) + [p for p in self.state.active_beds.values() if p]
        telemetry = {p.id: p.vitals.model_copy(deep=True) for p in all_patients}

        return IncidentObservation(
            episode_id=self.state.episode_id,
            queue_summary=q_sum,
            active_beds_summary=bed_sum,
            alerts=self.state.alerts[-5:],
            current_step=self.state.current_step,
            max_steps=self.state.max_steps,
            action_feedback=self.action_feedback,
            telemetry=telemetry
        )
