import copy
import random
from typing import Dict, Any

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
                "vitals": {"HR": "110", "BP": "150/90", "O2": "94%", "Temp": "37.1"},
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
                "vitals": {"HR": "85", "BP": "120/80", "O2": "99%", "Temp": "36.8"},
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
                "vitals": {"HR": "125", "BP": "85/50", "O2": "92%", "Temp": "39.2"},
                "symptoms": ["Confusion", "Fever", "Chills", "Decreased urination"],
                "history": ["UTI recurrences", "Penicillin Allergy"],
                "hidden_condition": "Sepsis"
            },
            {
                "id": "P-108", "age": 28,
                "vitals": {"HR": "40", "BP": "90/50", "O2": "82%", "Temp": "36.2"},
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
                "vitals": {"HR": "140", "BP": "70/40", "O2": "88%", "Temp": "36.5"},
                "symptoms": ["Unresponsive", "Massive trauma from MVA"],
                "history": ["Unknown"],
                "hidden_condition": "Hemorrhagic Shock"
            },
            {
                "id": "P-107", "age": 62,
                "vitals": {"HR": "85", "BP": "190/110", "O2": "96%", "Temp": "37.4"},
                "symptoms": ["Facial droop", "Slurred speech", "Left arm weakness"],
                "history": ["Hypertension"],
                "hidden_condition": "Stroke"
            },
            {
                "id": "P-105", "age": 9,
                "vitals": {"HR": "130", "BP": "100/60", "O2": "90%", "Temp": "37.5"},
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
                "vitals": {"HR": "118", "BP": "155/95", "O2": "93%", "Temp": "37.2"},
                "symptoms": ["Chest tightness", "Shortness of breath", "Jaw pain"],
                "history": ["Diabetes", "Hypertension"],
                "hidden_condition": "STEMI"
            },
            {
                "id": "P-202", "age": 34,
                "vitals": {"HR": "38", "BP": "88/52", "O2": "80%", "Temp": "36.1"},
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
                    "vitals": {"HR": "128", "BP": "82/48", "O2": "91%", "Temp": "39.5"},
                    "symptoms": ["High fever", "Confusion", "Rigors", "Dark urine"],
                    "history": ["Diabetes", "Penicillin Allergy"],
                    "hidden_condition": "Sepsis"
                }
            ],
            "16": [
                {
                    "id": "P-204", "age": 19,
                    "vitals": {"HR": "88", "BP": "118/76", "O2": "99%", "Temp": "36.9"},
                    "symptoms": ["Twisted ankle", "Pain on weight-bearing", "Mild swelling"],
                    "history": ["None"],
                    "hidden_condition": "Ankle Sprain"
                },
                {
                    "id": "P-205", "age": 67,
                    "vitals": {"HR": "92", "BP": "195/115", "O2": "95%", "Temp": "37.3"},
                    "symptoms": ["Sudden right-sided weakness", "Slurred speech", "Confusion"],
                    "history": ["Atrial fibrillation", "Hypertension"],
                    "hidden_condition": "Stroke"
                }
            ],
            "24": [
                {
                    "id": "P-206", "age": 7,
                    "vitals": {"HR": "135", "BP": "98/58", "O2": "89%", "Temp": "37.6"},
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
        v = patient["vitals"]
        if "HR" in v:
            try:
                hr = int(v["HR"].split("/")[0])
                jitter = random.uniform(0.95, 1.05)
                v["HR"] = str(max(30, min(200, int(hr * jitter))))
            except ValueError:
                pass
        if "O2" in v:
            try:
                o2 = int(v["O2"].replace("%", ""))
                jitter = random.uniform(0.95, 1.05)
                v["O2"] = f"{max(60, min(100, int(o2 * jitter)))}%"
            except ValueError:
                pass
        if "BP" in v:
            try:
                sys_p, dia = map(int, v["BP"].split("/"))
                v["BP"] = f"{max(40, int(sys_p * random.uniform(0.95, 1.05)))}/{max(30, int(dia * random.uniform(0.95, 1.05)))}"
            except ValueError:
                pass
        if "Temp" in v:
            try:
                temp = float(v["Temp"])
                jitter = random.uniform(0.95, 1.05)
                v["Temp"] = f"{max(34.0, min(42.0, temp * jitter)):.1f}"
            except ValueError:
                pass

    return scenario