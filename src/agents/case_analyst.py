import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from src.models.clinical_state import (
    ClinicalHypothesis,
    ClinicalState
)

load_dotenv()

class HypothesisList(BaseModel):
    """wrapper schema so LLM returns a list of hypotheses"""
    hypotheses: list[ClinicalHypothesis]
    

SYSTEM_PROMPT = """
You are a Clinical Case Synthesis Agent inside a safety-first decision support system.
Your job is to review anonymized patient facts and identify primary clinical problems,
acute lab abnormalities, or medication-related risks.

STRICT RULES:
1. Base your assessment ONLY on the provided patient facts. Never invent history or labs.
2. Every hypothesis MUST cite exact 'fact_id' values (e.g., F-01, F-03) in 'supporting_fact_ids'.
3. Assign sequential IDs starting with H-01, H-02, etc.
4. Keep reasoning concise, objective, and focused on clinical mechanisms.
"""

def run_case_synthesis(state: ClinicalState, model_name: str = "gemini-3.8-flash") -> ClinicalState:
    """analyze patient facts in clinicalstate and populates initial hypothesis"""
    if not state.phi_scrubbed:
        raise RuntimeError("SAFETY BLOCK: cannot send un_scrubberd patient data to LLM")
    
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")
    
    client = genai.Client()
    
    patient_summary = (
        f"Patient: {state.patient_age}-year-old {state.patient_gender}\n"
        f"Ftructured facts:\n"
        f"{[fact.model_dump() for fact in state.patient_facts]}"
    )
    
    response = client.models.generate_content(
        model=model_name,
        contents=patient_summary,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=HypothesisList,
            temperature=0.1 #low for deterministic clinical reasoning
        ),
    )
    
    parsed = HypothesisList.model_validate_json(response.text)
    state.initial_hypotheses = parsed.hypotheses
    return state