"""
Medical Triage Simulator - Core Engine
Handles the physiological progression of patients, critical alerts,
and the execution of clinical actions.
"""
import random
from typing import Dict, Any, List
from models import IncidentState, Patient, IncidentAction, IncidentObservation, VitalsTelemetry
from tasks import EVALUATIONS_DB, INTERACTIONS_DB


class Simulator:
    """
    Main simulation engine for the Medical Triage environment.
    Manages the patient lifecycle from arrival to disposition.
    """
    def __init__(self, state: IncidentState):
        self.state = state
        self.action_feedback = ""
        self.completed: dict = {}

    def step(self, action: IncidentAction) -> None:
        """Executes a single clinical action and advances the global state."""
        self.state.current_step += 1
        self.action_feedback = f"Action {action.action_type} executed."

        if action.action_type == "wait":
            self.action_feedback = "Observation period. No immediate action taken."
            self._process_time_step()
            return

        if not action.patient_id:
            self.action_feedback = "Error: Clinical action requires a target patient_id."
            return

        patient = self._get_patient(action.patient_id)
        if not patient:
            self.action_feedback = f"Error: Patient {action.patient_id} not found in census."
            return

        p_id = patient.id

        if action.action_type == "assess":
            self.action_feedback = (
                f"Full assessment completed for Patient {p_id}. "
                f"Symptoms: {patient.symptoms}. Vitals: {patient.vitals}. History: {patient.history}."
            )
        elif action.action_type == "order_test":
            if action.target in EVALUATIONS_DB:
                patient.tests_ordered.append(action.target)
                result = EVALUATIONS_DB[action.target].get(patient.hidden_condition, "Normal.")
                patient.test_results[action.target] = result
                self.action_feedback = f"Diagnostic {action.target} completed for {p_id}. Result: {result}"
            else:
                self.action_feedback = f"Error: {action.target} is not a valid diagnostic in this protocol."
        elif action.action_type == "treat":
            treatment = action.target or "Supportive Care"
            patient.treatments_given.append(treatment)
            # Check for life-threatening drug interactions
            for condition, contraindications in INTERACTIONS_DB.items():
                if condition in patient.history or condition == patient.hidden_condition:
                    if treatment in contraindications:
                        self.state.fatal_errors.append(
                            f"FATAL ERROR for {p_id}: {treatment} was administered despite {condition} contraindication."
                        )
                        patient.is_stable = False
            self.action_feedback = f"Treatment '{treatment}' administered to {p_id}."
        elif action.action_type == "triage":
            try:
                patient.triage_level = int(action.target)
                self.action_feedback = f"ED Triage Complete: {p_id} assigned Level {action.target}."
            except (ValueError, TypeError):
                self.action_feedback = f"Error: Invalid Triage Level: {action.target}"
        elif action.action_type == "admit":
            ward = action.target or "General Medicine"
            patient.admitted_ward = ward
            patient.discharged = True
            self.action_feedback = f"Patient {p_id} admitted to {ward} for definitive care."
        elif action.action_type == "discharge":
            patient.discharged = True
            self.action_feedback = f"Patient {p_id} stabilized and discharged home."
        else:
            self.action_feedback = f"Error: Action '{action.action_type}' not recognized in standard ED workflow."

        if patient and patient.discharged:
            self._finalize_disposition(patient)

        self._process_time_step()

    def _finalize_disposition(self, patient: Patient):
        """Moves a patient from active beds/queue to the shift summary census."""
        self.completed[patient.id] = patient
        if patient in self.state.queue:
            self.state.queue.remove(patient)
        for bed, p in self.state.active_beds.items():
            if p and p.id == patient.id:
                self.state.active_beds[bed] = None

    def _get_patient(self, pid: str) -> Patient:
        """Retrieves a patient by ID from the queue, active beds, or completed list."""
        for p in self.state.queue:
            if p.id == pid:
                return p
        for p in self.state.active_beds.values():
            if p and p.id == pid:
                return p
        return self.completed.get(pid)

    def _process_time_step(self):
        """Simulates the passage of time, triggering clinical deterioration and arrivals."""
        CRITICAL_LEVELS = {
            "STEMI": {"O2": -2, "HR": +5},
            "Sepsis": {"HR": +8, "BP_sys": -2},
            "Hemorrhagic Shock": {"HR": +10, "BP_sys": -10},
            "Status Asthmaticus": {"O2": -3, "HR": +6},
            "Opioid Overdose": {"O2": -4},
        }

        all_patients = list(self.state.queue) + [p for p in self.state.active_beds.values() if p]

        for patient in all_patients:
            cond = patient.hidden_condition
            # Skip progression if already treated or stable
            if cond not in CRITICAL_LEVELS or patient.treatments_given or patient.admitted_ward:
                pass
            else:
                self._apply_pathological_deterioration(patient, CRITICAL_LEVELS[cond])

            # Always record vitals history and apply physiological jitter
            patient.vitals_history.append(patient.vitals.model_copy(deep=True))
            self._apply_physiological_noise(patient.vitals)

        # Handle bed management
        empty_beds = [b for b, p in self.state.active_beds.items() if p is None]
        for b in empty_beds:
            if self.state.queue:
                self.state.active_beds[b] = self.state.queue.pop(0)

        # Check for shift completion
        if self.state.current_step >= self.state.max_steps:
            self.state.is_done = True
            self.state.alerts.append("SHIFT END: Time limit reached. Finalizing reports.")

        active_count = len(self.state.queue) + sum(1 for p in self.state.active_beds.values() if p)
        if active_count == 0:
            self.state.is_done = True
            self.state.alerts.append("SHIFT END: All patients processed and dispositioned.")

    def _apply_pathological_deterioration(self, patient: Patient, deltas: Dict[str, int]):
        """Simulates clinical decline for critical untreated conditions."""
        v = patient.vitals
        if "O2" in deltas:
            v.o2 = max(60, v.o2 + deltas["O2"])
            if v.o2 < 82:
                self.state.alerts.append(f"CRITICAL HYPOXIA: Patient {patient.id} O2 is {v.o2}%!")
        
        if "HR" in deltas:
            v.hr = min(210, v.hr + deltas["HR"])
            if v.hr > 155:
                self.state.alerts.append(f"SEVERE TACHYCARDIA: Patient {patient.id} HR is {v.hr} bpm!")
        
        if "BP_sys" in deltas:
            v.bp_sys = max(40, v.bp_sys + deltas["BP_sys"])
            if v.bp_sys < 65:
                self.state.alerts.append(f"HYPOTENSIVE CRASH: Patient {patient.id} Sys BP is {v.bp_sys} mmHg!")

    def _apply_physiological_noise(self, vitals: VitalsTelemetry):
        """Adds natural stochastic fluctuations to patient vital signs."""
        noise = random.uniform(0.995, 1.005)
        vitals.hr = int(vitals.hr * noise)
        vitals.o2 = min(100, int(vitals.o2 * noise))
        vitals.temp = round(vitals.temp * random.uniform(0.998, 1.002), 1)
        
        # Hemodynamic jitter
        vitals.bp_sys = int(vitals.bp_sys * random.uniform(0.99, 1.01))
        vitals.bp_dia = int(vitals.bp_dia * random.uniform(0.99, 1.01))
        
        # Absolute safety clamps for biological realism
        vitals.hr = max(30, min(220, vitals.hr))
        vitals.o2 = max(50, vitals.o2)
        vitals.temp = max(34.0, min(43.0, vitals.temp))

    def _enrich_patient_obs(self, p: Patient) -> dict:
        """Adds clinical trend analysis to a patient's observation data."""
        obs = {
            "id": p.id,
            "vitals": p.vitals.to_dict(),  # String mappings for Agent compatibility
            "time_in_wait": self.state.current_step - p.arrival_step,
            "clinical_trend": "stable",
            "hemodynamic_delta": {}
        }
        if len(p.vitals_history) >= 1:
            prev: VitalsTelemetry = p.vitals_history[-1]
            curr: VitalsTelemetry = p.vitals
            delta = {}
            
            d_hr = curr.hr - prev.hr
            if d_hr != 0: delta["HR"] = d_hr
            
            d_o2 = curr.o2 - prev.o2
            if d_o2 != 0: delta["O2"] = d_o2
            
            obs["hemodynamic_delta"] = delta
            if d_hr >= 5 or d_o2 <= -2:
                obs["clinical_trend"] = "worsening"
            elif d_hr <= -5 or d_o2 >= 2:
                obs["clinical_trend"] = "improving"

        return obs

    def get_observation(self) -> IncidentObservation:
        """Constructs the final observation payload for the RL agent."""
        q_sum = []
        for p in self.state.queue:
            p_obs = self._enrich_patient_obs(p)
            p_obs["chief_complaint"] = p.symptoms[:2]
            q_sum.append(p_obs)

        bed_sum = {}
        for b, p in self.state.active_beds.items():
            if p:
                p_obs = self._enrich_patient_obs(p)
                p_obs.update({
                    "clinical_stability": p.is_stable,
                    "esi_level": p.triage_level,
                    "diagnostics_completed": p.tests_ordered,
                    "medications_administered": p.treatments_given
                })
                bed_sum[b] = p_obs
            else:
                bed_sum[b] = "Bed Available"

        # Provide strict numeric telemetry for advanced analytic models
        all_patients = list(self.state.queue) + [p for p in self.state.active_beds.values() if p]
        telemetry = {p.id: p.vitals.model_copy(deep=True) for p in all_patients}

        return IncidentObservation(
            episode_id=self.state.episode_id,
            queue_summary=q_sum,
            active_beds_summary=bed_sum,
            alerts=self.state.alerts[-8:],
            current_step=self.state.current_step,
            max_steps=self.state.max_steps,
            action_feedback=self.action_feedback,
            telemetry=telemetry
        )
