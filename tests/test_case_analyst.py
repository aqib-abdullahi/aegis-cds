from src.ingestion.fhir_reader import parse_fhir_bundle
from src.agents.case_analyst import run_case_synthesis


if  __name__ == "__main__":
    state = parse_fhir_bundle("data/sample_patient.json", case_id="CASE-001")
    state = run_case_synthesis(state)
    print(state.model_dump_json(indent=2))