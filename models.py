"""
Medical Triage Environment - Data Models
This module defines the core data structures for the hospital simulation,
adhering to OpenEnv and Gymnasium standards.
"""
import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class VitalsTelemetry(BaseModel):
    """
    Structured telemetry for patient vital signs.
    Used for objective clinical tracking and deterioration simulation.
    """
    hr: int = 80
    bp_sys: int = 120
    bp_dia: int = 80
    o2: int = 100
    temp: float = 37.0

    def to_dict(self) -> Dict[str, str]:
        """Convert to the legacy string-dictionary format for backward compatibility."""
        return {
            "HR": str(self.hr),
            "BP": f"{self.bp_sys}/{self.bp_dia}",
            "O2": f"{self.o2}%",
            "Temp": f"{self.temp:.1f}"
        }

    def __str__(self) -> str:
        return f"HR {self.hr}, BP {self.bp_sys}/{self.bp_dia}, O2 {self.o2}%, Temp {self.temp:.1f}"


class Patient(BaseModel):
    """
    Represents a patient in the ED (Emergency Department).
    Tracks clinical state, treatment history, and disposition.
    """
    id: str
    age: int
    vitals: VitalsTelemetry
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
    vitals_history: List[VitalsTelemetry] = Field(default_factory=list)
    arrival_step: int = 0


class IncidentState(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    queue: List[Patient]
    active_beds: Dict[str, Optional[Patient]]
    current_step: int
    max_steps: int
    arrival_schedule: Dict[int, List[Patient]] = Field(default_factory=dict)
    alerts: List[str] = Field(default_factory=list)
    fatal_errors: List[str] = Field(default_factory=list)
    score_components: Dict[str, float] = Field(default_factory=dict)
    is_done: bool = False
    difficulty: str = "easy"


class IncidentObservation(BaseModel):
    episode_id: str = ""
    queue_summary: List[Dict[str, Any]]
    active_beds_summary: Dict[str, Any]
    alerts: List[str]
    current_step: int
    max_steps: int = 0
    action_feedback: str
    telemetry: Optional[Dict[str, VitalsTelemetry]] = None  # New structured field for SF parity


class IncidentAction(BaseModel):
    action_type: str
    patient_id: Optional[str] = None
    target: Optional[str] = None


class TriageState(BaseModel):
    episode_id: str
    step: int
    max_steps: int
    done: bool
    difficulty: str
    patients_in_queue: int
    patients_in_beds: int
    fatal_errors: int
    alerts: List[str]
    score: float = 0.01