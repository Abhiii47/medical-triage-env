from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid


class Patient(BaseModel):
    """A patient in the ER queue or occupying a bed."""
    id: str
    age: int
    vitals: Dict[str, str]
    symptoms: List[str]
    history: List[str]
    tests_ordered: List[str] = Field(default_factory=list)
    test_results: Dict[str, str] = Field(default_factory=dict)
    treatments_given: List[str] = Field(default_factory=list)
    triage_level: Optional[int] = None  
    admitted_ward: Optional[str] = None
    discharged: bool = False
    is_stable: bool = True
    hidden_condition: Optional[str] = None
    vitals_history: List[Dict[str, str]] = Field(default_factory=list)


class IncidentState(BaseModel):
    """Full internal state of the triage environment."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    queue: List[Patient]
    active_beds: Dict[str, Optional[Patient]]
    current_step: int
    max_steps: int
    alerts: List[str] = Field(default_factory=list)
    fatal_errors: List[str] = Field(default_factory=list)
    score_components: Dict[str, float] = Field(default_factory=dict)
    is_done: bool = False
    difficulty: str = "easy"


class IncidentObservation(BaseModel):
    """
    Observation returned to the agent at each step.

    Fields:
        episode_id: Unique identifier for the current episode.
        queue_summary: List of patients waiting (id, vitals, top symptoms).
        active_beds_summary: Map of bed name -> patient info (or 'Empty').
        alerts: Last 5 system alerts (vitals warnings, fatal events).
        current_step: Step counter since last reset.
        max_steps: Maximum steps allowed in this episode.
        action_feedback: Human-readable result of the last action taken.
    """
    episode_id: str = ""
    queue_summary: List[Dict[str, Any]]
    active_beds_summary: Dict[str, Any]
    alerts: List[str]
    current_step: int
    max_steps: int = 0
    action_feedback: str


class IncidentAction(BaseModel):
    """
    Action submitted by the agent.

    Fields:
        action_type: One of: assess, order_test, treat, triage, admit, discharge, wait
        patient_id: Target patient ID (required for all actions except 'wait').
        target: Dependent on action_type:
            - order_test: test name (ECG, Blood Test, CT Scan, X-Ray, Tox Screen)
            - treat: drug/treatment name (e.g., Aspirin, Naloxone, Fluids)
            - triage: integer 1–5 as string
            - admit: ward name (Cardiology, ICU, Neurology, Surgery, General)
    """
    action_type: str
    patient_id: Optional[str] = None
    target: Optional[str] = None


class TriageState(BaseModel):
    """
    Concise episode state summary exposed via the state() endpoint.
    Compatible with the OpenEnv State base contract.
    """
    episode_id: str
    step: int
    max_steps: int
    done: bool
    difficulty: str
    patients_in_queue: int
    patients_in_beds: int
    fatal_errors: int
    alerts: List[str]
