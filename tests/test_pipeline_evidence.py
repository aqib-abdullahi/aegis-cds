from src.ingestion.fhir_reader import parse_fhir_bundle
from src.agents.case_analyst import run_case_synthesis
from src.agents.evidence_retriever import run_evidence_retrieval

if __name__ == "__main__":
    state = parse_fhir_bundle("data/sample_patient.json", case_id="CASE-001")
    state = run_case_synthesis(state)
    state = run_evidence_retrieval(state)
    print(state.model_dump_json(indent=2))