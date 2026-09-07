"""
feature_engineering.py
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
            # =====================================================
            # ACTIVITY FEATURES
            # =====================================================
            "activity_code": activity.activity_code,
            "activity_quantity": activity.activity_quantity,
            "activity_gross": activity.activity_gross,

            # =====================================================
            # PATIENT FEATURES
            # =====================================================
            "patient_age": request.patient_age,
            "gender": request.gender,
            "nationality": request.nationality,

            # =====================================================
            # CLAIM FEATURES
            # =====================================================
            "claim_gross": request.claim_gross,
            "claim_net": request.claim_net,

            "encounter_type": request.encounter_type,
            "length_of_stay": request.length_of_stay,

            # =====================================================
            # PROVIDER FEATURES
            # =====================================================
            "clinician_profession": request.clinician_profession,

            "facility_type": request.facility_type,
            "payer_classification": request.payer_classification,

            # =====================================================
            # CPT
            # =====================================================
            "cpt_category": convert_cpt_category(
                activity.activity_code
            ),
        }

        row.update(diagnosis_features)

        rows.append(row)

    return pd.DataFrame(rows)