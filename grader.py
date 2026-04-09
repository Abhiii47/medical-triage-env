"""
Medical Triage Clinical Auditor (Grader)
Determines the performance score based on clinical accuracy, safety, 
and disposition efficiency. Adheres to strict Meta OpenEnv Phase 2 specs.
"""
from models import VitalsTelemetry


EXPECTED = {
    "STEMI": {
        "level": 1, 
        "tests": ["ECG"], 
        "treat": ["Aspirin", "Nitroglycerin", "Heparin"], 
        "ward": "Cardiology"
    },
    "Sepsis": {
        "level": 2, 
        "tests": ["Blood Test"], 
        "treat": ["Vancomycin", "Ceftriaxone", "Meropenem", "Piperacillin", "Fluids", "IV Fluids"], 
        "ward": "ICU"
    },
    "Ankle Sprain": {
        "level": 5, 
        "tests": ["X-Ray"], 
        "treat": [], 
        "ward": None
    },
    "Hemorrhagic Shock": {
        "level": 1, 
        "tests": ["CT Scan"], 
        "treat": ["Blood Transfusion", "IV Fluids", "Fluids", "Transfusion"], 
        "ward": "Surgery"
    },
    "Status Asthmaticus": {
        "level": 1, 
        "tests": [], 
        "treat": ["Albuterol", "Salbutamol", "Epinephrine", "Steroids", "Oxygen"], 
        "ward": "ICU"
    },
    "Stroke": {
        "level": 2, 
        "tests": ["CT Scan"], 
        "treat": ["tPA", "Alteplase", "Aspirin"], 
        "ward": "Neurology"
    },
    "Opioid Overdose": {
        "level": 1, 
        "tests": ["Tox Screen"], 
        "treat": ["Naloxone"], 
        "ward": "ICU"
    },
}

class TriageRubric:
    """
    Formally evaluates the quality of care provided to a single patient.
    Weights are distributed based on clinical significance (20% Triage, 30% Treatment, etc.).
    """
    
    @staticmethod
    def evaluate_patient_outcome(patient_record: dict) -> float:
        score = 0.0
        condition = patient_record.get("hidden_condition")
        standards = EXPECTED.get(condition, {})
        
        if not standards:
            return 0.0

        
        if patient_record.get("triage_level") == standards.get("level"):
            score += 0.20

        
        required_tests = standards.get("tests", [])
        tests_ordered = patient_record.get("tests_ordered", [])
        if not required_tests:
            score += 0.20
        elif any(t in tests_ordered for t in required_tests):
            score += 0.20

    
        accepted_treats = standards.get("treat", [])
        treatments_given = patient_record.get("treatments_given", [])
        is_emergent = standards.get("level") in (1, 2)
        has_stabilized = any(t in treatments_given for t in accepted_treats) if accepted_treats else True

        if has_stabilized:
            score += 0.30
        elif is_emergent and patient_record.get("admitted_ward"):
            
            score -= 0.20

        
        target_ward = standards.get("ward")
        actual_ward = patient_record.get("admitted_ward")
        discharged = patient_record.get("discharged")

        if target_ward:
            if actual_ward == target_ward:
                score += 0.30
        elif discharged and not actual_ward:
            score += 0.30

        return score

def grade(state, all_patients_history) -> float:
    """
    Global grading function. 
    Calculates the mean score across all patients and applies system-level penalties.
    """
    try:
        total_score = 0.0
        max_possible = 0.0
        diagnostic_waste_penalty = 0.0
        
        
        if hasattr(state, "model_dump"): 
            state_data = state.model_dump()
        else: 
            state_data = state if isinstance(state, dict) else {}

        fatal_errors = state_data.get("fatal_errors", [])
        if all_patients_history is None: 
            all_patients_history = []

        for patient in all_patients_history:
            max_possible += 1.0
            if hasattr(patient, "model_dump"): 
                p_record = patient.model_dump()
            else: 
                p_record = patient if isinstance(patient, dict) else {}
            
            
            total_score += TriageRubric.evaluate_patient_outcome(p_record)
            
            
            standards = EXPECTED.get(p_record.get("hidden_condition"), {})
            for test in p_record.get("tests_ordered", []):
                if test not in standards.get("tests", []):
                    diagnostic_waste_penalty += 0.02

        total_score -= 0.50 * len(fatal_errors) 
        total_score -= diagnostic_waste_penalty

        if max_possible > 0:
            final_score = total_score / max_possible
        else:
            final_score = 0.0

        
        return round(max(0.01, min(0.99, final_score)), 4)
    except Exception:
        
        return 0.01

def grade_task(task_id: str, state, all_patients_history) -> float:
    """Entry point for many Evaluation Frameworks."""
    return grade(state, all_patients_history)
