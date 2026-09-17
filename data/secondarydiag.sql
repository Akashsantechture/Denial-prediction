WITH secondary_features AS (

    SELECT
        haad_claim_id,
        activity_code,

        COUNT(*) FILTER (
            WHERE diagnosis_type = 'Secondary'
        ) AS secondary_dx_count,

        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1) IN ('A','B') THEN 1 ELSE 0 END) AS secondary_infectious,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='C' THEN 1 ELSE 0 END) AS secondary_oncology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='D' THEN 1 ELSE 0 END) AS secondary_oncology_hematology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='E' THEN 1 ELSE 0 END) AS secondary_endocrinology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='F' THEN 1 ELSE 0 END) AS secondary_psychiatry,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='G' THEN 1 ELSE 0 END) AS secondary_neurology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='H' THEN 1 ELSE 0 END) AS secondary_eye_ear,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='I' THEN 1 ELSE 0 END) AS secondary_cardiology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='J' THEN 1 ELSE 0 END) AS secondary_pulmonology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='K' THEN 1 ELSE 0 END) AS secondary_gastroenterology_dental,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='L' THEN 1 ELSE 0 END) AS secondary_dermatology,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='M' THEN 1 ELSE 0 END) AS secondary_musculoskeletal,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='N' THEN 1 ELSE 0 END) AS secondary_genitourinary,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='O' THEN 1 ELSE 0 END) AS secondary_obgyn,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='P' THEN 1 ELSE 0 END) AS secondary_pediatrics,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='Q' THEN 1 ELSE 0 END) AS secondary_congenital,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='R' THEN 1 ELSE 0 END) AS secondary_general_symptoms,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1) IN ('S','T') THEN 1 ELSE 0 END) AS secondary_trauma_burns_poisoning,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1) IN ('V','W','X','Y') THEN 1 ELSE 0 END) AS secondary_external_causes,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1)='Z' THEN 1 ELSE 0 END) AS secondary_factors_influencing_health_status,
        MAX(CASE WHEN LEFT(UPPER(diagnosis_code),1) NOT IN
            ('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','V','W','X','Y','Z')
            THEN 1 ELSE 0 END
        ) AS secondary_unknown_icd_category

    FROM claim_denial_reason_model_v1
    WHERE diagnosis_type = 'Secondary'
    GROUP BY haad_claim_id, activity_code
)

UPDATE claim_denial_reason_model_v1 t
SET
    secondary_dx_count = s.secondary_dx_count,
    secondary_infectious = s.secondary_infectious,
    secondary_oncology = s.secondary_oncology,
    secondary_oncology_hematology = s.secondary_oncology_hematology,
    secondary_endocrinology = s.secondary_endocrinology,
    secondary_psychiatry = s.secondary_psychiatry,
    secondary_neurology = s.secondary_neurology,
    secondary_eye_ear = s.secondary_eye_ear,
    secondary_cardiology = s.secondary_cardiology,
    secondary_pulmonology = s.secondary_pulmonology,
    secondary_gastroenterology_dental = s.secondary_gastroenterology_dental,
    secondary_dermatology = s.secondary_dermatology,
    secondary_musculoskeletal = s.secondary_musculoskeletal,
    secondary_genitourinary = s.secondary_genitourinary,
    secondary_obgyn = s.secondary_obgyn,
    secondary_pediatrics = s.secondary_pediatrics,
    secondary_congenital = s.secondary_congenital,
    secondary_general_symptoms = s.secondary_general_symptoms,
    secondary_trauma_burns_poisoning = s.secondary_trauma_burns_poisoning,
    secondary_external_causes = s.secondary_external_causes,
    secondary_factors_influencing_health_status = s.secondary_factors_influencing_health_status,
    secondary_unknown_icd_category = s.secondary_unknown_icd_category
FROM secondary_features s
WHERE
    t.haad_claim_id = s.haad_claim_id
    AND t.activity_code = s.activity_code;