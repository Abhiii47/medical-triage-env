from models import VitalsTelemetry

EXPECTED = {
    "STEMI": {"level": 1, "tests": ["ECG"], "treat": ["Aspirin", "Nitroglycerin", "Heparin"], "ward": "Cardiology"},
    "Sepsis": {"level": 2, "tests": ["Blood Test"], "treat": ["Vancomycin", "Ceftriaxone", "Meropenem", "Piperacillin", "Fluids", "IV Fluids"], "ward": "ICU"},
    "Ankle Sprain": {"level": 5, "tests": ["X-Ray"], "treat": [], "ward": None},
    "Hemorrhagic Shock": {"level": 1, "tests": ["CT Scan"], "treat": ["Blood Transfusion", "IV Fluids", "Fluids", "Transfusion"], "ward": "Surgery"},
    "Status Asthmaticus": {"level": 1, "tests": [], "treat": ["Albuterol", "Salbutamol", "Epinephrine", "Steroids", "Oxygen"], "ward": "ICU"},
    "Stroke": {"level": 2, "tests": ["CT Scan"], "treat": ["tPA", "Alteplase", "Aspirin"], "ward": "Neurology"},
    "Opioid Overdose": {"level": 1, "tests": ["Tox Screen"], "treat": ["Naloxone"], "ward": "ICU"},
}

class TriageRubric:
    """A formal scoring rubric for Medical Triage, inspired by San Francisco winning projects."""
    
    @staticmethod
    def evaluate_patient(p_dict) -> float:
        p_score = 0.0
        hidden_condition = p_dict.get("hidden_condition")
        exp = EXPECTED.get(hidden_condition, {})
        if not exp: return 0.0

        # 1. Triage Accuracy (20%)
        if p_dict.get("triage_level") == exp.get("level"):
            p_score += 0.20

        # 2. Diagnostic Accuracy (20%)
        required_tests = exp.get("tests", [])
        tests_ordered = p_dict.get("tests_ordered", [])
        if not required_tests:
            p_score += 0.20
        elif any(t in tests_ordered for t in required_tests):
            p_score += 0.20

        # 3. Treatment Accuracy & Stabilization (30%)
        accepted_treats = exp.get("treat", [])
        treatments_given = p_dict.get("treatments_given", [])
        is_critical = exp.get("level") in (1, 2)
        has_treatment = any(t in treatments_given for t in accepted_treats) if accepted_treats else True

        if has_treatment:
            p_score += 0.30
        elif is_critical and p_dict.get("admitted_ward"):
            # Penalty for admitting without stabilization
            p_score -= 0.20

        # 4. Correct Disposition (30%)
        target_ward = exp.get("ward")
        admitted_ward = p_dict.get("admitted_ward")
        discharged = p_dict.get("discharged")

        if target_ward:
            if admitted_ward == target_ward:
                p_score += 0.30
        elif discharged and not admitted_ward:
            p_score += 0.30

        return p_score

def grade(state, all_patients_history) -> float:
    try:
        score = 0.0
        max_score = 0.0
        unnecessary_penalty = 0.0
        
        # Robust dictionary conversion
        if hasattr(state, "model_dump"): sd = state.model_dump()
        else: sd = state if isinstance(state, dict) else {}

        fatal_errors = sd.get("fatal_errors", [])
        
        if all_patients_history is None: all_patients_history = []

        for p in all_patients_history:
            max_score += 1.0
            if hasattr(p, "model_dump"): pd = p.model_dump()
            else: pd = p if isinstance(p, dict) else {}
            
            # Use the Rubric for logic
            score += TriageRubric.evaluate_patient(pd)
            
            # Penalize unnecessary tests
            exp = EXPECTED.get(pd.get("hidden_condition"), {})
            for t in pd.get("tests_ordered", []):
                if t not in exp.get("tests", []):
                    unnecessary_penalty += 0.05

        score -= 0.50 * len(fatal_errors)
        score -= unnecessary_penalty

        if max_score > 0:
            final = score / max_score
        else:
            final = 0.0

        # Deterministic range clamping for Phase 2 compliance
        return round(max(0.01, min(0.99, final)), 4)
    except Exception:
        return 0.01

def grade_task(task_id: str, state, all_patients_history) -> float:
    return grade(state, all_patients_history)
