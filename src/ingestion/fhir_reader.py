import json
from datetime import date
from pathlib import Path
from src.models.clinical_state import ClinicalState, PatientFact


def calculate_age(birth_date_str: str) -> int:
    """Converts yyyy-mm-dd birth date into age in years"""
    birth_year, birth_month, birth_day = map(int, birth_date_str.split("-"))
    today = date.today()
    age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
    return age

def parse_fhir_bundle(file_path: str, case_id: str = "CASE-001") -> ClinicalState:
    """Reads a FHIR R4 bundle json and returns a de-identified clinicalstate"""
    raw_data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    
    if raw_data.get("resourceType") != "Bundle":
        raise ValueError("Expected a FHIR bundle resource")
    
    state = ClinicalState(case_id=case_id)
    fact_counter = 1
    
    for entry in raw_data.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")
        
        #demographics stripped
        if rtype == "Patient":
            state.patient_gender = resource.get("gender")
            if "birthDate" in resource:
                state.patient_age = calculate_age(resource["birthDate"])
            state.phi_scrubbed = True
        
        #diagnoses/conditions
        elif rtype == "Condition":
            coding = resource.get("code", {}).get("coding", [{}])[0]
            status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown")
            state.patient_facts.append(
                PatientFact(
                    fact_id=f"F-{fact_counter:02d}",
                    category="condition",
                    label=coding.get("display", "Unknown Condition"),
                    value=f"Status: {status}",
                    code=coding.get("code"),
                )
            )
            fact_counter += 1
        
        #medications
        elif rtype == "MedicationRequest":
            coding = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            state.patient_facts.append(
                PatientFact(
                    fact_id=f"F-{fact_counter:02d}",
                    category="medication",
                    label=coding.get("display", "Unknown Medication"),
                    value=f"Status: {resource.get('status', 'unknown')}",
                    code=coding.get("code"),
                    timestamp=resource.get("authoredOn"),
                )
            )
            fact_counter += 1
        
        elif rtype == "Observation":
            coding = resource.get("code", {}).get("coding", [{}])[0]
            val_qty = resource.get("valueQuantity", {})
            value_str = f"{val_qty.get('value')} {val_qty.get('unit', '')}".strip()
            state.patient_facts.append(
                PatientFact(
                    fact_id=f"F-{fact_counter:02d}",
                    category="lab",
                    label=coding.get("display", "Unknown Lab"),
                    value=value_str,
                    code=coding.get("code"),
                    timestamp=resource.get("effectiveDatetime"),
                )
            )
            fact_counter += 1
            
    return state