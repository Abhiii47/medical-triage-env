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


def grade(state: IncidentState, all_patients_history: list) -> float:
    score = 0.0
    max_score = 0.0
    unnecessary_penalty = 0.0
    outcomes_achieved = 0

    for p in all_patients_history:
        max_score += 1.0
        p_score = 0.0
        exp = EXPECTED.get(p.hidden_condition, {})

        if p.triage_level == exp.get("level"):
            p_score += 0.25

        required_tests = exp.get("tests", [])
        if not required_tests:
            p_score += 0.25
        elif any(t in p.tests_ordered for t in required_tests):
            p_score += 0.25

        for t in p.tests_ordered:
            if t not in required_tests:
                unnecessary_penalty += 0.10

        accepted_treats = exp.get("treat", [])
        if not accepted_treats:
            p_score += 0.25
        elif any(t in p.treatments_given for t in accepted_treats):
            p_score += 0.25

        target_ward = exp.get("ward")
        if target_ward:
            if p.admitted_ward == target_ward:
                p_score += 0.25
                outcomes_achieved += 1
        elif p.discharged and not p.admitted_ward:
            p_score += 0.25
            outcomes_achieved += 1

        score += p_score

    score -= 0.40 * len(state.fatal_errors)
    score -= unnecessary_penalty

    n_patients = len(all_patients_history)
    if max_score > 0 and outcomes_achieved == n_patients and not state.fatal_errors:
        steps_used = state.current_step
        efficiency = max(0.0, (state.max_steps - steps_used) / state.max_steps)
        score += efficiency * 0.15 * max_score

    if max_score > 0:
        final = score / max_score
    else:
        final = 0.0

    return round(max(0.01, min(0.99, final)), 4)


def grade_task(task_id: str, state: IncidentState, all_patients_history: list) -> float:
    return grade(state, all_patients_history)
