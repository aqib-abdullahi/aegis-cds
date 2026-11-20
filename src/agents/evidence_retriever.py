import json
import os
from pathlib import Path
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from dotenv import load_dotenv
from src.models.clinical_state import ClinicalState, RetrievedEvidence


class EvidenceMatchList(BaseModel):
    """wrapper schema for structured LLM output"""
    matches: list[RetrievedEvidence]
    

SYSTEM_PROMPT = """
You are a clinical evidence retrieval agent inside a safety-first decision support system.
Given a patient's structured facts, initial hypotheses, and a vetted library of clinical guidelines,
select only the guidelines that directly apply to the case

STRIC RULES:
1. You may ONLY use 'evidence_id', 'source', and 'excerpt' values verbatim from the provided library
2. Never invent external citations, DOIs, or guideline numbers.
3. Link each selected guideline to exact 'hypothesis_id' values (e.g., H-01, H-02).
4. Include any guideline that warns about diagnostic pitfalls or lab artifacts relevant to these values.
"""

FREE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash"
]

def run_evidence_retrieval(state: ClinicalState,
                           guidelines_path: str = "data/clinical_guidelines.json") -> ClinicalState:    
    """matches vetted clinical guidelines to initial hypotheses with strict ID validation"""
    if not state.phi_scrubbed:
        raise RuntimeError("SAFETY BLOCK: Patient data is not de-identified.")
    if not state.initial_hypotheses:
        raise RuntimeError("PIPELINE ERROR: Cannot retrieve evidence before initial hypotheses")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY env variable")
    
    raw_guidelines = json.loads(Path(guidelines_path).read_text(encoding="utf-8"))
    valid_ids = {item["evidence_id"]: item for item in raw_guidelines}
    
    prompt_payload = (
        f"Patient facts:\n{[f.model_dump() for f in state.patient_facts]}\n\n"
        f"Initial Hypotheses:\n{[h.model_dump() for h in state.initial_hypotheses]}\n\n"
        f"Vetted Guideline Library:\n{json.dumps(raw_guidelines, indent=2)}"
    )
    
    client = genai.Client()
    last_error = None
    
    for model_name in FREE_MODELS:
        try:
            print(f"[Evidence retriever] trying model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_payload,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=EvidenceMatchList,
                    temperature=0.0 #strict source matching
                ),
            ) 
            parsed = EvidenceMatchList.model_validate_json(response.text)
            
            #guardrail: verify every evidence_id exists in our vetted store
            verified_matches = []
            for match in parsed.matches:
                if match.evidence_id not in valid_ids:
                    raise ValueError(f"HALLUCINATION BLOCKED: Unknown evidence_id '{match.evidence_id}'")
                #enforce terms from source file
                canonical = valid_ids[match.evidence_id]
                match.source = canonical["source"]
                match.excerpt = canonical["excerpt"]
                verified_matches.append(match)
            
            state.retrieved_evidence = verified_matches
            return state

        except errors.APIError as err:
            print(f"[Evidence Retriever] {model_name} unavailable ({err.code}), trying fallback...")
    
    raise RuntimeError(f"All free-tier models failed. Last error: {last_error}")
                