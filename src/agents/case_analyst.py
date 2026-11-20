import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
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

#FALLBACK
FREE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]

def run_case_synthesis(state: ClinicalState) -> ClinicalState:
    """analyze patient facts in clinicalstate and populates initial hypothesis"""
    if not state.phi_scrubbed:
        raise RuntimeError("SAFETY BLOCK: cannot send un-scrubberd patient data to LLM")
    
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")
    
    client = genai.Client()
    
    patient_summary = (
        f"Patient: {state.patient_age}-year-old {state.patient_gender}\n"
        f"Ftructured facts:\n"
        f"{[fact.model_dump() for fact in state.patient_facts]}"
    )
    
    
    last_error = None
    for model_name in FREE_MODELS:
        try:
            print(f"[Case Analyst] Trying model: {model_name}...")
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
        
        except errors.APIError as err:
            print("[Case Analystt] {model_name} unavailable ({err.code}), trying fallback")
            last_error = err
    
    raise RuntimeError(f"All free-tier models failed. Last error: {last_error}")