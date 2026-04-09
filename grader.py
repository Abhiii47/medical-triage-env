from models import IncidentState

EXPECTED = {
    "STEMI": {"level": 1, "tests": ["ECG"], "treat": ["Aspirin", "Nitroglycerin", "Heparin"], "ward": "Cardiology"},
    "Sepsis": {"level": 2, "tests": ["Blood Test"], "treat": ["Vancomycin", "Ceftriaxone", "Meropenem", "Piperacillin", "Fluids", "IV Fluids"], "ward": "ICU"},
    "Ankle Sprain": {"level": 5, "tests": ["X-Ray"], "treat": [], "ward": None},
    "Hemorrhagic Shock": {"level": 1, "tests": ["CT Scan"], "treat": ["Blood Transfusion", "IV Fluids", "Fluids", "Transfusion"], "ward": "Surgery"},
    "Status Asthmaticus": {"level": 1, "tests": [], "treat": ["Albuterol", "Salbutamol", "Epinephrine", "Steroids", "Oxygen"], "ward": "ICU"},
    "Stroke": {"level": 2, "tests": ["CT Scan"], "treat": ["tPA", "Alteplase", "Aspirin"], "ward": "Neurology"},
    "Opioid Overdose": {"level": 1, "tests": ["Tox Screen"], "treat": ["Naloxone"], "ward": "ICU"},
}


def grade(state, all_patients_history) -> float:
    try:
        score = 0.0
        max_score = 0.0
        unnecessary_penalty = 0.0
        outcomes_achieved = 0

        if hasattr(state, "model_dump"):
            state_dict = state.model_dump()
        elif hasattr(state, "__dict__"):
            state_dict = state.__dict__
        else:
            state_dict = state if isinstance(state, dict) else {}

        fatal_errors = state_dict.get("fatal_errors", [])
        current_step = state_dict.get("current_step", 0)
        max_steps = state_dict.get("max_steps", 100)
        
        if all_patients_history is None:
            all_patients_history = []

        for p in all_patients_history:
            max_score += 1.0
            p_score = 0.0
            
            if hasattr(p, "model_dump"):
                p_dict = p.model_dump()
            elif hasattr(p, "__dict__"):
                p_dict = p.__dict__
            else:
                p_dict = p if isinstance(p, dict) else {}

            hidden_condition = p_dict.get("hidden_condition")
            exp = EXPECTED.get(hidden_condition, {})

            if p_dict.get("triage_level") == exp.get("level"):
                p_score += 0.25

            required_tests = exp.get("tests", [])
            tests_ordered = p_dict.get("tests_ordered", [])
            if not required_tests:
                p_score += 0.25
            elif any(t in tests_ordered for t in required_tests):
                p_score += 0.25

            for t in tests_ordered:
                if t not in required_tests:
                    unnecessary_penalty += 0.10

            accepted_treats = exp.get("treat", [])
            treatments_given = p_dict.get("treatments_given", [])
            if not accepted_treats:
                p_score += 0.25
            elif any(t in treatments_given for t in accepted_treats):
                p_score += 0.25

            target_ward = exp.get("ward")
            admitted_ward = p_dict.get("admitted_ward")
            discharged = p_dict.get("discharged")

            if target_ward:
                if admitted_ward == target_ward:
                    p_score += 0.25
                    outcomes_achieved += 1
            elif discharged and not admitted_ward:
                p_score += 0.25
                outcomes_achieved += 1

            score += p_score

        score -= 0.40 * len(fatal_errors)
        score -= unnecessary_penalty

        n_patients = len(all_patients_history)
        if max_score > 0 and outcomes_achieved == n_patients and not fatal_errors:
            steps_used = current_step
            efficiency = max(0.0, (max_steps - steps_used) / max_steps)
            score += efficiency * 0.15 * max_score

        if max_score > 0:
            final = score / max_score
        else:
            final = 0.0

        return round(max(0.01, min(0.99, final)), 4)
    except Exception:
        return 0.01


def grade_task(task_id: str, state, all_patients_history) -> float:
    return grade(state, all_patients_history)
