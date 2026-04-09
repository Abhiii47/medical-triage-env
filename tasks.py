"""
Triage Scenarios & Clinical Database
Defines archetypes, deterioration paths, and test results
based on standard emergency medicine protocols.
"""
import copy
import random
from typing import Dict, Any
from models import VitalsTelemetry

EVALUATIONS_DB = {
    "ECG": {"STEMI": "ST elevation in leads II, III, aVF (Inferior STEMI).", "Sepsis": "Normal Sinus Rhythm.", "Ankle Sprain": "Sinus Tachycardia."},
    "Blood Test": {"STEMI": "Troponin highly elevated.", "Sepsis": "WBC 18.0 (High), Lactate 4.5 (High).", "Ankle Sprain": "Normal limits."},
    "X-Ray": {"Ankle Sprain": "No fracture, significant soft tissue swelling."},
    "CT Scan": {"Stroke": "Acute ischemic stroke in territory of MCA.", "Hemorrhagic Shock": "Massive internal bleeding detected."},
    "Tox Screen": {"Opioid Overdose": "Positive for Opioids and Benzodiazepines."}
}

INTERACTIONS_DB = {
    "Penicillin Allergy": ["Penicillin", "Amoxicillin"],
    "Opioid Overdose": ["Morphine", "Fentanyl", "Oxycodone"],
    "Hemorrhagic Shock": ["Aspirin", "Heparin", "Warfarin"]
}

SCENARIOS: Dict[str, Any] = {
    "easy": {
        "max_steps": 15,
        "patients": [
            {
                "id": "P-101", "age": 65,
                "vitals": VitalsTelemetry(hr=110, bp_sys=150, bp_dia=90, o2=94, temp=37.1),
                "symptoms": ["Crushing chest pain", "Diaphoresis", "Left arm radiation"],
                "history": ["Hypertension", "Smoker"],
                "hidden_condition": "STEMI"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "easy_ankle_sprain": {
        "max_steps": 10,
        "patients": [
            {
                "id": "P-109", "age": 22,
                "vitals": VitalsTelemetry(hr=85, bp_sys=120, bp_dia=80, o2=99, temp=36.8),
                "symptoms": ["Ankle pain", "Swelling", "Difficulty walking"],
                "history": ["None"],
                "hidden_condition": "Ankle Sprain"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "medium": {
        "max_steps": 20,
        "patients": [
            {
                "id": "P-102", "age": 78,
                "vitals": VitalsTelemetry(hr=125, bp_sys=85, bp_dia=50, o2=92, temp=39.2),
                "symptoms": ["Confusion", "Fever", "Chills", "Decreased urination"],
                "history": ["UTI recurrences", "Penicillin Allergy"],
                "hidden_condition": "Sepsis"
            },
            {
                "id": "P-108", "age": 28,
                "vitals": VitalsTelemetry(hr=40, bp_sys=90, bp_dia=50, o2=82, temp=36.2),
                "symptoms": ["Pinpoint pupils", "Unresponsive", "Respiratory depression"],
                "history": ["Substance Abuse"],
                "hidden_condition": "Opioid Overdose"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "hard": {
        "max_steps": 25,
        "patients": [
            {
                "id": "P-104", "age": 45,
                "vitals": VitalsTelemetry(hr=140, bp_sys=70, bp_dia=40, o2=88, temp=36.5),
                "symptoms": ["Unresponsive", "Massive trauma from MVA"],
                "history": ["Unknown"],
                "hidden_condition": "Hemorrhagic Shock"
            },
            {
                "id": "P-107", "age": 62,
                "vitals": VitalsTelemetry(hr=85, bp_sys=190, bp_dia=110, o2=96, temp=37.4),
                "symptoms": ["Facial droop", "Slurred speech", "Left arm weakness"],
                "history": ["Hypertension"],
                "hidden_condition": "Stroke"
            },
            {
                "id": "P-105", "age": 9,
                "vitals": VitalsTelemetry(hr=130, bp_sys=100, bp_dia=60, o2=90, temp=37.5),
                "symptoms": ["Severe wheezing", "Accessory muscle use", "Can't speak in full sentences"],
                "history": ["Asthma"],
                "hidden_condition": "Status Asthmaticus"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "chaotic": {
        "max_steps": 35,
        "patients": [
            {
                "id": "P-201", "age": 55,
                "vitals": VitalsTelemetry(hr=118, bp_sys=155, bp_dia=95, o2=93, temp=37.2),
                "symptoms": ["Chest tightness", "Shortness of breath", "Jaw pain"],
                "history": ["Diabetes", "Hypertension"],
                "hidden_condition": "STEMI"
            },
            {
                "id": "P-202", "age": 34,
                "vitals": VitalsTelemetry(hr=38, bp_sys=88, bp_dia=52, o2=80, temp=36.1),
                "symptoms": ["Unresponsive", "Slow shallow breathing", "Pinpoint pupils"],
                "history": ["Heroin use"],
                "hidden_condition": "Opioid Overdose"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None, "Bed_3": None},
        "arrival_schedule": {
            "8": [
                {
                    "id": "P-203", "age": 71,
                    "vitals": VitalsTelemetry(hr=128, bp_sys=82, bp_dia=48, o2=91, temp=39.5),
                    "symptoms": ["High fever", "Confusion", "Rigors", "Dark urine"],
                    "history": ["Diabetes", "Penicillin Allergy"],
                    "hidden_condition": "Sepsis"
                }
            ],
            "16": [
                {
                    "id": "P-204", "age": 19,
                    "vitals": VitalsTelemetry(hr=88, bp_sys=118, bp_dia=76, o2=99, temp=36.9),
                    "symptoms": ["Twisted ankle", "Pain on weight-bearing", "Mild swelling"],
                    "history": ["None"],
                    "hidden_condition": "Ankle Sprain"
                },
                {
                    "id": "P-205", "age": 67,
                    "vitals": VitalsTelemetry(hr=92, bp_sys=195, bp_dia=115, o2=95, temp=37.3),
                    "symptoms": ["Sudden right-sided weakness", "Slurred speech", "Confusion"],
                    "history": ["Atrial fibrillation", "Hypertension"],
                    "hidden_condition": "Stroke"
                }
            ],
            "24": [
                {
                    "id": "P-206", "age": 7,
                    "vitals": VitalsTelemetry(hr=135, bp_sys=98, bp_dia=58, o2=89, temp=37.6),
                    "symptoms": ["Severe wheezing", "Retractions", "Unable to complete sentences"],
                    "history": ["Asthma", "Multiple prior intubations"],
                    "hidden_condition": "Status Asthmaticus"
                }
            ]
        }
    },
}


def get_scenario(difficulty: str) -> Dict[str, Any]:
    scenario = copy.deepcopy(SCENARIOS.get(difficulty.lower(), SCENARIOS["easy"]))
    if difficulty.lower() in ("easy", "easy_ankle_sprain"):
        return scenario

    all_patients_to_jitter = list(scenario["patients"])
    for ps in scenario.get("arrival_schedule", {}).values():
        all_patients_to_jitter.extend(ps)

    for patient in all_patients_to_jitter:
        v: VitalsTelemetry = patient["vitals"]
        jitter = random.uniform(0.95, 1.05)
        
        v.hr = max(30, min(200, int(v.hr * jitter)))
        v.o2 = max(60, min(100, int(v.o2 * jitter)))
        v.bp_sys = max(40, int(v.bp_sys * random.uniform(0.95, 1.05)))
        v.bp_dia = max(30, int(v.bp_dia * random.uniform(0.95, 1.05)))
        v.temp = round(max(34.0, min(42.0, v.temp * random.uniform(0.99, 1.01))), 1)

    return scenario