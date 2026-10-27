"""
feature_engineering.py
----------------------
Builds a wide dataframe from a ClaimRequest.

Financial fields (activity_gross, claim_gross, claim_net) are included
in the dataframe so the API can use them for UI calculations (at-risk
amount etc.), but both models slice their own FEATURE_ORDER columns
and never see the financial fields.
"""

import pandas as pd

from preprocessing.categorial_conversion import (
    convert_cpt_category,
    generate_diagnosis_features,
)


def build_activity_dataframe(request):
    rows = []

    for activity in request.activities:

        secondary_dx_codes = [
            d.diagnosis_code
            for d in activity.diagnoses
            if d.diagnosis_type.lower() == "secondary"
        ]

        diagnosis_features = generate_diagnosis_features(
            primary_diagnosis_code=request.primary_diagnosis_code,
            secondary_diagnosis_codes=secondary_dx_codes,
        )

        row = {
            # ── Activity ──────────────────────────────────
            "activity_code":      activity.activity_code,
            "activity_quantity":  activity.activity_quantity,
            "activity_gross":     activity.activity_gross,   # UI only

            # ── Patient ───────────────────────────────────
            "patient_age":  request.patient_age,
            "gender":       request.gender,

            # ── Encounter ─────────────────────────────────
            "encounter_type":  request.encounter_type,
            "length_of_stay":  request.length_of_stay,

            # ── Provider ──────────────────────────────────
            "clinician_profession": request.clinician_profession,
            "facility_type":        request.facility_type,

            # ── Payer ─────────────────────────────────────
            "payer_classification": request.payer_classification,
            "payer_name":           request.payer_name,
            "payer_id":             request.payer_id,
            "receiver_id":          request.receiver_id,
            "network":              request.network,

            # ── CPT ───────────────────────────────────────
            "cpt_category": convert_cpt_category(activity.activity_code),
        }

        row.update(diagnosis_features)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# KSA — SERVICE DATAFRAME BUILDER
# ============================================================
#
# Mirrors build_activity_dataframe() for UAE but maps to KSA schema.
#
# Key differences:
#   - Iterates request.services (not request.activities)
#   - Line-item column names: service_code, quantity (not activity_code)
#   - No CPT category — KSA models have no cpt_category feature
#   - Adds claim-level fields: department, claim_type, claim_subtype
#   - facility (name) is a separate field from facility_type
#   - policy_name replaces payer_name
#   - primary_diagnosis_code mapped to column "primary_diagnosis"
#   - Uses generate_ksa_diagnosis_features() for secondary dx flags
#
# Both KSA pipelines include their own encoder internally.
# The returned dataframe is passed directly to pipeline.predict_proba().
# Two separate slices are made in main.py:
#   df[KSA_FEATURE_ORDER]        → score pipeline
#   df[KSA_REASON_FEATURE_ORDER] → reason pipeline
# ============================================================

from preprocessing.categorial_conversion import generate_ksa_diagnosis_features


def build_service_dataframe(request) -> "pd.DataFrame":
    """
    Build a wide dataframe from a KSAClaimRequest.

    One row per service. Financial fields (service_gross, claim_gross,
    claim_net) are included for UI at-risk calculations but are not in
    either model's feature order and are never passed to the pipelines.
    """
    rows = []

    for service in request.services:

        secondary_dx_codes = [
            d.diagnosis_code
            for d in service.diagnoses
            if d.diagnosis_type.lower() == "secondary"
        ]

        diagnosis_features = generate_ksa_diagnosis_features(
            primary_diagnosis_code=request.primary_diagnosis_code,
            secondary_diagnosis_codes=secondary_dx_codes,
        )

        row = {
            # ── Service ───────────────────────────────────
            "service_code":  service.service_code,
            "quantity":      service.service_quantity,
            "service_gross": service.service_gross,     # UI only

            # ── Patient ───────────────────────────────────
            "patient_age": request.patient_age,
            "gender":      request.gender,

            # ── Facility ──────────────────────────────────
            "facility":      request.facility,          # facility name
            "facility_type": request.facility_type,

            # ── Encounter ─────────────────────────────────
            "encounter_type": request.encounter_type,
            "length_of_stay": request.length_of_stay,

            # ── Clinical ──────────────────────────────────
            "department":   request.department,
            "claim_type":   request.claim_type,
            "claim_subtype": request.claim_subtype,

            # ── Payer ─────────────────────────────────────
            "payer_classification": request.payer_classification,
            "payer_id":             request.payer_id,
            "receiver_id":          request.receiver_id,
            "policy_name":          request.policy_name,

            # ── Financial (UI only) ───────────────────────
            "claim_gross": request.claim_gross,
            "claim_net":   request.claim_net,
        }

        # Adds primary_diagnosis, primary_diagnosis_category,
        # and all secondary_dx_* flag columns
        row.update(diagnosis_features)
        rows.append(row)

    return pd.DataFrame(rows)
