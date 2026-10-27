from typing import List, Optional
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
    activity_code:     str
    activity_quantity: float
    activity_gross:    float        # kept for UI risk-amount calc — not fed to models
    diagnoses:         List[Diagnosis]


# ==========================================================
# CLAIM
# ==========================================================

class ClaimRequest(BaseModel):
    claim_id:   str

    # Patient
    patient_age: int
    gender:      str

    # Encounter
    encounter_type:  str
    length_of_stay:  int

    # Financial — kept for UI display / at-risk amount, not fed to models
    claim_gross: Optional[float] = None
    claim_net:   Optional[float] = None

    # Provider
    clinician_profession: str
    facility_type:        str

    # Payer
    payer_classification: str
    payer_name:           str
    payer_id:             str
    receiver_id:          str
    network:              str

    # Clinical
    primary_diagnosis_code: str

    activities: List[Activity]


# ==========================================================
# KSA — SERVICE + CLAIM SCHEMA
# ==========================================================
#
# KSA (Nphies/CCHI) uses "services" not "activities".
# Key differences from UAE:
#   - Line item: service_code / service_quantity / service_gross
#   - Claim-level: department, claim_type, claim_subtype, policy_name
#   - facility (name string) is a separate field from facility_type
#   - clinician_category replaces clinician_profession
#   - No network / payer_name fields
#
# Score model input  : 38 columns (feature_columns.json)
# Reason model input : 38 columns (feature_names_in_)
# Both models are sklearn Pipelines — encoder is baked in.
# Call: pipeline.predict_proba(df) directly, no separate encode step.
# ==========================================================

class KSAService(BaseModel):
    service_code:     str
    service_quantity: float
    service_gross:    float        # UI / at-risk calc only — not in model features
    diagnoses:        List[Diagnosis]


class KSAClaimRequest(BaseModel):
    claim_id: str

    # Patient
    patient_age: int
    gender:      str

    # Facility
    facility:      str             # facility name  (score model: yes, reason model: no)
    facility_type: str

    # Encounter
    encounter_type: str
    length_of_stay: int

    # Clinical
    clinician_category:    str
    department:            str
    claim_type:            str
    claim_subtype:         str
    primary_diagnosis_code: str    # mapped → "primary_diagnosis" column

    # Payer
    payer_classification: str
    payer_id:             str
    receiver_id:          str
    policy_name:          str      # reason model: yes, score model: no

    # Financial — UI only, not fed to models
    claim_gross: Optional[float] = None
    claim_net:   Optional[float] = None

    services: List[KSAService]
