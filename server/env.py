import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import IncidentState, IncidentObservation, IncidentAction, Patient, TriageState
from tasks import get_scenario
from simulator import Simulator
from grader import grade


class MedicalTriageEnv:
    """
    OpenEnv-compatible Medical ER Triage environment.
    Implements reset() / step() / state() — Gymnasium-style interface.
    """

    def __init__(self):
        self._state: IncidentState = None
        self.simulator: Simulator = None
        self.all_patients_history: list = []
        self.difficulty: str = "easy"

    def reset(self, **kwargs) -> IncidentObservation:
        """Reset to a new episode. kwargs: difficulty='easy'|'medium'|'hard'"""
        import random
        env_diff = os.getenv("ENV_DIFFICULTY", "easy")
        self.difficulty = kwargs.get("difficulty", env_diff)
        scenario = get_scenario(self.difficulty)

        patients = [Patient(**p) for p in scenario["patients"]]
        random.shuffle(patients)
        self.all_patients_history = [p.model_copy(deep=True) for p in patients]

        self._state = IncidentState(
            queue=patients,
            active_beds={b: None for b in scenario["beds"]},
            current_step=0,
            max_steps=scenario["max_steps"],
            difficulty=self.difficulty,
        )
        self.simulator = Simulator(self._state)
        self.simulator._update_time()
        return self.simulator.get_observation()

    def step(self, action: IncidentAction):
        """
        Execute one action. Returns (observation, reward, done, info).

        Step rewards:  +0.03 assess  +0.01 order_test  +0.03 triage
                       +0.05 admit/discharge  -0.15 fatal treat  -0.01 wait
        Terminal reward: deterministic grade score [0.0–1.0]
        """
        if not self.simulator:
            raise RuntimeError("Call reset() before step().")

        prev_fatal_count = len(self._state.fatal_errors)
        self.simulator.step(action)
        obs = self.simulator.get_observation()

        for p in self.all_patients_history:
            current = self.simulator._get_patient(p.id)
            if current:
                p.triage_level    = current.triage_level
                p.tests_ordered   = list(current.tests_ordered)
                p.test_results    = dict(current.test_results)
                p.treatments_given = list(current.treatments_given)
                p.admitted_ward   = current.admitted_ward
                p.discharged      = current.discharged
                p.is_stable       = current.is_stable

        step_reward = 0.0
        at = action.action_type

        if at == "assess":
            step_reward += 0.03
        elif at == "order_test" and action.target:
            step_reward += 0.01
        elif at == "triage" and action.patient_id:
            step_reward += 0.03
        elif at == "treat" and action.patient_id:
            new_fatals = len(self._state.fatal_errors) - prev_fatal_count
            if new_fatals > 0:
                step_reward -= 0.15 * new_fatals
        elif at in ("admit", "discharge") and action.patient_id:
            step_reward += 0.05
        elif at == "wait":
            step_reward -= 0.01

        done = self._state.is_done
        if done:
            final_score = grade(self._state, self.all_patients_history)
            return obs, final_score, done, {"final_score": final_score}

        return obs, round(step_reward, 4), done, {}

    def state(self) -> TriageState:
        """Current episode state — OpenEnv state() contract."""
        if self._state is None:
            return TriageState(
                episode_id="", step=0, max_steps=0, done=False,
                difficulty=self.difficulty, patients_in_queue=0,
                patients_in_beds=0, fatal_errors=0, alerts=[],
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
        )

    def get_state(self) -> IncidentState:
        """Return raw internal state (for testing)."""
        return self._state
