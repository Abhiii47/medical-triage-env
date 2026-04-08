import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import IncidentState, IncidentObservation, IncidentAction, Patient, TriageState
from tasks import get_scenario
from simulator import Simulator
from grader import grade, EXPECTED


class MedicalTriageEnv:
    def __init__(self):
        self._state: IncidentState = None
        self.simulator: Simulator = None
        self.all_patients_history: list = []
        self.difficulty: str = "easy"

    def reset(self, **kwargs) -> IncidentObservation:
        import random
        if kwargs.get("seed") is not None:
            random.seed(kwargs["seed"])

        env_diff = os.getenv("ENV_DIFFICULTY", "easy")
        self.difficulty = kwargs.get("difficulty", env_diff)
        scenario = get_scenario(self.difficulty)

        patients = [Patient(**p) for p in scenario["patients"]]
        if len(patients) > 1:
            random.shuffle(patients)
        self.all_patients_history = [p.model_copy(deep=True) for p in patients]

        arrival_sched = {}
        if "arrival_schedule" in scenario:
            for step_key, patient_list in scenario["arrival_schedule"].items():
                arrival_sched[int(step_key)] = [Patient(**p) for p in patient_list]

        self._state = IncidentState(
            queue=patients,
            active_beds={b: None for b in scenario["beds"]},
            current_step=0,
            max_steps=scenario["max_steps"],
            arrival_schedule=arrival_sched,
            difficulty=self.difficulty,
        )
        self.simulator = Simulator(self._state)
        self.simulator._update_time()
        return self.simulator.get_observation()

    def step(self, action: IncidentAction):
        if not self.simulator:
            raise RuntimeError("Call reset() before step().")

        prev_fatal_count = len(self._state.fatal_errors)
        self.simulator.step(action)

        if self._state.arrival_schedule and self._state.current_step in self._state.arrival_schedule:
            new_patients = self._state.arrival_schedule.pop(self._state.current_step)
            for new_p in new_patients:
                new_p.arrival_step = self._state.current_step
                self._state.queue.append(new_p)
                self.all_patients_history.append(new_p.model_copy(deep=True))
                self._state.alerts.append(f"SURGE: New patient {new_p.id} arrived!")

            empty_beds = [b for b, p in self._state.active_beds.items() if p is None]
            for b in empty_beds:
                if self._state.queue:
                    self._state.active_beds[b] = self._state.queue.pop(0)

        obs = self.simulator.get_observation()

        for p in self.all_patients_history:
            current = self.simulator._get_patient(p.id)
            if current:
                p.triage_level = current.triage_level
                p.tests_ordered = list(current.tests_ordered)
                p.test_results = dict(current.test_results)
                p.treatments_given = list(current.treatments_given)
                p.admitted_ward = current.admitted_ward
                p.discharged = current.discharged
                p.is_stable = current.is_stable

        step_reward = 0.0
        at = action.action_type

        if at == "assess":
            step_reward += 0.03
            if action.patient_id:
                p = self.simulator._get_patient(action.patient_id)
                if p and p.triage_level is None:
                    try:
                        o2 = int(p.vitals.get("O2", "100").replace("%", ""))
                        hr = int(p.vitals.get("HR", "80").split("/")[0])
                        if o2 < 85 or hr > 140:
                            step_reward += 0.05
                    except Exception:
                        pass
        elif at == "order_test" and action.target:
            step_reward += 0.01
        elif at == "triage" and action.patient_id:
            step_reward += 0.03
        elif at == "treat" and action.patient_id:
            new_fatals = len(self._state.fatal_errors) - prev_fatal_count
            if new_fatals > 0:
                step_reward -= 0.15 * new_fatals
            else:
                p = self.simulator._get_patient(action.patient_id)
                if p and p.hidden_condition in EXPECTED:
                    req_test = EXPECTED[p.hidden_condition].get("tests")
                    if req_test and req_test[0] in p.tests_ordered:
                        step_reward += 0.04
        elif at in ("admit", "discharge") and action.patient_id:
            step_reward += 0.05
        elif at == "wait":
            step_reward -= 0.01

        done = self._state.is_done
        if done:
            final_score = grade(self._state, self.all_patients_history)
            return obs, final_score, done, {"final_score": final_score}

        return obs, round(max(0.01, min(0.99, float(step_reward))), 4), done, {}

    def state(self) -> TriageState:
        if self._state is None:
            return TriageState(
                episode_id="", step=0, max_steps=0, done=False,
                difficulty=self.difficulty, patients_in_queue=0,
                patients_in_beds=0, fatal_errors=0, alerts=[], score=0.01
            )
        s = self._state
        occupied = sum(1 for p in s.active_beds.values() if p is not None)
        return TriageState(
            episode_id=s.episode_id,
            step=s.current_step,
            max_steps=s.max_steps,
            done=s.is_done,
            difficulty=self.difficulty,
            patients_in_queue=len(s.queue),
            patients_in_beds=occupied,
            fatal_errors=len(s.fatal_errors),
            alerts=s.alerts[-5:],
            score=grade(s, self.all_patients_history) if s.is_done else 0.01,
        )

    def get_state(self) -> IncidentState:
        return self._state

    def render(self) -> str:
        s = self._state
        if not s:
            return "Environment not initialized."
        lines = [f"=== ER Status | Step {s.current_step}/{s.max_steps} ==="]
        lines.append(f"Queue ({len(s.queue)}): " + ", ".join(p.id for p in s.queue))
        for bed, p in s.active_beds.items():
            status = f"{p.id} L{p.triage_level or '?'}" if p else "Empty"
            lines.append(f"  {bed}: {status}")
        if s.fatal_errors:
            lines.append(f"FATAL ERRORS: {len(s.fatal_errors)}")
        return "\n".join(lines)