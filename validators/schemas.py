from typing import List
from pydantic import BaseModel


# ==========================================================
# DIAGNOSIS
# ==========================================================

class Diagnosis(BaseModel):
    diagnosis_code: str
    diagnosis_type: str


# ==========================================================
# ACTIVITY
# ==========================================================

class Activity(BaseModel):

    activity_code: str
    activity_quantity: float
    activity_gross: float

    diagnoses: List[Diagnosis]


# ==========================================================
# CLAIM
# ==========================================================

class ClaimRequest(BaseModel):

    claim_id: str

    patient_age: int
    gender: str
    nationality: str

    encounter_type: str
    length_of_stay: int

    claim_gross: float
    claim_net: float

    clinician_profession: str

    facility_type: str
    payer_classification: str

    primary_diagnosis_code: str

    activities: List[Activity]