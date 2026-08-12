# # validation/schemas.py
# import re
# from pydantic import BaseModel, Field, field_validator, model_validator

# class ClaimInput(BaseModel):
#     primary_diagnosis_code: str = Field(..., example="E11.22")
#     activity_code: str = Field(..., example="84432")
#     medical_necessity_score: float = Field(0.50, example=0.6813)
#     pa_risk_score: float = Field(0.00, example=0.0124)
#     coverage_score: float = Field(0.50, example=0.5164)
#     clinician_success_score: float = Field(0.50, example=0.6259)
#     facility_success_score: float = Field(0.50, example=0.6482)
#     activity_gross: float = Field(..., example=1200.00)
#     activity_quantity: int = Field(1, example=1)
#     gross_amount: float = Field(..., example=1200.00)
#     net_amount: float = Field(..., example=1000.00)
#     patient_age: int = Field(..., example=45)
#     gender: str = Field("UNKNOWN", example="MALE")
#     nationality: str = Field("UNKNOWN", example="EMIRATI")
#     payer_id: str = Field("UNKNOWN", example="E001")
#     insurance_plan_tier: str = Field("Standard", example="Standard")
#     profession: str = Field("UNKNOWN", example="Internal Medicine")
#     category: str = Field("UNKNOWN", example="Pulmonology")
#     facility_type_id: str = Field("UNKNOWN", example="Hospital")
#     billing_lag_days: int = Field(0, example=2)
#     length_of_stay: int = Field(0, example=0)
#     encounter_type: str = Field("OP", example="OP")


# class UserClaimInput(BaseModel):
#     primary_diagnosis_code: str = Field(..., example="I10")
#     activity_code: str = Field(..., example="99213")
#     activity_gross: float = Field(..., gt=0, example=250.00)
#     activity_quantity: int = Field(1, ge=1, le=15, example=1)
#     gross_amount: float = Field(..., gt=0, example=250.00)
#     net_amount: float = Field(..., gt=0, example=250.00)
#     patient_age: int = Field(..., ge=0, le=120, example=52)
#     gender: str = Field("MALE", example="MALE")
#     nationality: str = Field("EMIRATI", example="EMIRATI")
#     payer_id: str = Field("E001", example="E001")
#     insurance_plan_tier: str = Field("Standard", example="Standard")
#     profession: str = Field("Internal Medicine", example="Internal Medicine")
#     category: str = Field("Internal Medicine", example="Internal Medicine")
#     facility_type_id: str = Field("Clinic", example="Clinic")
#     billing_lag_days: int = Field(1, ge=0, le=365, example=1)
#     length_of_stay: int = Field(0, ge=0, example=0)
#     encounter_type: str = Field("OP", example="OP")

#     # Guardrail 1: Strict ICD-10 Format Validation
#     @field_validator("primary_diagnosis_code")
#     @classmethod
#     def validate_icd_code(cls, v: str) -> str:
#         clean_code = v.strip().upper()
#         icd_pattern = r"^[A-Z][0-9]{2}(\.[0-9]{1,4})?$"
#         if not re.match(icd_pattern, clean_code):
#             raise ValueError(
#                 f"Invalid ICD-10 Diagnosis Code format: '{v}'. Must follow standard medical coding format (e.g., 'I10', 'E11.22')."
#             )
#         return clean_code

#     # Guardrail 2: Relational Cross-Field Validation Rules
#     @model_validator(mode="after")
#     def validate_business_rules(self):
#         if self.net_amount > self.gross_amount:
#             raise ValueError("net_amount cannot exceed gross_amount.")
#         if self.encounter_type == "OP" and self.length_of_stay > 0:
#             raise ValueError("Outpatient (OP) encounters must have a length_of_stay of 0 days.")
#         return self


# class PredictionResponse(BaseModel):
#     denial_probability_pct: float
#     is_high_risk: bool
#     risk_level: str
#     action_recommendation: str


# validation/schemas.py
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class UserClaimInput(BaseModel):
    # 1. Clinical Features
    diagnosis_code: str = Field(..., example="I10")
    diagnosis_type: str = Field("Principal", example="Principal")
    activity_code: str = Field(..., example="99213")
    
    # 2. Financial Features
    activity_gross: float = Field(..., gt=0, example=250.00)
    activity_quantity: int = Field(1, ge=1, le=15, example=1)
    claim_gross: float = Field(..., gt=0, example=250.00)
    claim_net: float = Field(..., ge=0, example=250.00)
    
    # 3. Patient Features
    patient_age: int = Field(..., ge=0, le=120, example=52)
    gender: str = Field("MALE", example="MALE")
    nationality: str = Field("EMIRATI", example="EMIRATI")
    
    # 4. Insurance & Provider Features
    payer_id: Optional[str] = Field("E001", example="E001") # Kept for optional passthrough
    insurance_plan_tier: str = Field("Insurance", example="Insurance")
    clinician_profession: str = Field("Internal Medicine", example="Internal Medicine")
    clinician_category: str = Field("Internal Medicine", example="Internal Medicine")
    facility_type: str = Field("Clinic", example="Clinic")
    
    # 5. Operational Features
    billing_lag_days: int = Field(1, ge=0, le=365, example=1)
    length_of_stay: int = Field(0, ge=0, example=0)
    encounter_type: str = Field("OP", example="OP")

    # Guardrail 1: Strict ICD-10 Format Validation
    @field_validator("diagnosis_code")
    @classmethod
    def validate_icd_code(cls, v: str) -> str:
        clean_code = v.strip().upper()
        icd_pattern = r"^[A-Z][0-9]{2}(\.[0-9]{1,4})?$"
        if not re.match(icd_pattern, clean_code):
            raise ValueError(
                f"Invalid ICD-10 Diagnosis Code format: '{v}'. Must follow standard medical coding format (e.g., 'I10', 'E11.22')."
            )
        return clean_code

    # Guardrail 2: Relational Cross-Field Validation Rules
    @model_validator(mode="after")
    def validate_business_rules(self):
        if self.claim_net > self.claim_gross:
            raise ValueError("claim_net cannot exceed claim_gross.")
        if self.encounter_type == "OP" and self.length_of_stay > 0:
            raise ValueError("Outpatient (OP) encounters must have a length_of_stay of 0 days.")
        return self


class PredictionResponse(BaseModel):
    denial_probability_pct: float
    is_high_risk: bool
    risk_level: str
    action_recommendation: str