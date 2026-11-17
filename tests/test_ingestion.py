from src.ingestion.fhir_reader import parse_fhir_bundle

if __name__ == "__main__":
    clinical_state = parse_fhir_bundle("data/sample_patient.json", case_id="CASE-001")
    print(clinical_state.model_dump_json(indent=2))