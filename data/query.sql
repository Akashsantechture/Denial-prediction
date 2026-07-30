----phase 1 cleaning------

CREATE TABLE public.denial_prediction_features AS		
SELECT 		
    -- 1. Diagnosis & Procedure Features		
    COALESCE(ca.diagnosis_code, 'UNKNOWN') AS primary_diagnosis_code,		
    'NONE' AS secondary_diagnosis_code, -- Simplification unless mapped from separate diagnosis table		
    ca.activity_code,		
    		
    -- 2. Financials		
    ca.activity_gross::NUMERIC(12,2) AS activity_gross,		
    ca.activity_quantity::INTEGER AS activity_quantity,		
    ca.claim_gross::NUMERIC(12,2) AS gross_amount,		
    ca.claim_net::NUMERIC(12,2) AS net_amount,		
    		
    -- 3. Demographics		
    ca.patient_age::INTEGER AS patient_age,		
    COALESCE(NULLIF(ca.gender, ''), 'UNKNOWN') AS gender,		
    COALESCE(NULLIF(ca.nationality, ''), 'UNKNOWN') AS nationality,		
    		
    -- 4. Provider & Payer Context		
    ca.payer_id,		
    COALESCE(NULLIF(ca.payer_classification, ''), 'Standard') AS insurance_plan_tier,		
    COALESCE(NULLIF(ca.clinician_profession, ''), 'UNKNOWN') AS profession,		
    COALESCE(NULLIF(ca.clinician_category, ''), 'UNKNOWN') AS category,		
    COALESCE(NULLIF(ca.facility_type, ''), 'UNKNOWN') AS facility_type_id,		
    		
    -- 5. Temporal & Operational Features		
    GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted - ca.datestamp)))::INTEGER AS billing_lag_days,		
    COALESCE(NULLIF(ca.claim_route, ''), 'Direct') AS network,		
    COALESCE(NULLIF(ca.payer_classification, ''), 'Standard') AS plan_type,		
    		
    -- 6. Prior Authorization Risk Proxy Features		
    CASE WHEN ca.activity_denial_code = 'AUTH-001' THEN 0 ELSE 1 END AS prior_auth_approved_flag,		
    CASE WHEN ca.activity_denial_code = 'AUTH-001' THEN 1 ELSE 0 END AS prior_auth_needed,		
    		
    -- 7. Encounter Info		
    COALESCE(ca.length_of_stay, 0)::INTEGER AS length_of_stay,		
    COALESCE(NULLIF(ca.encounter_type, ''), 'OP') AS encounter_type,		
    		
    -- 8. Activity Classification		
    CASE 		
        WHEN ca.activity_code ~ '^[0-9]' THEN 'CPT_Procedure'		
        WHEN ca.activity_code LIKE 'D%' THEN 'Dental'		
        ELSE 'General'		
    END AS activity_type,		
    		
    -- 9. TARGET VARIABLE (1 = Denied, 0 = Approved)		
    CASE WHEN ca.activity_denied = 't' THEN 1 ELSE 0 END AS claim_status		
		
FROM public.claim_activity ca		
		
-- Intelligent Filters to Drop Noise & Non-Realtime Data:		
WHERE ca.date_submitted IS NOT NULL              -- Drops 416k incomplete/unsubmitted rows		
  AND ca.activity_gross > 0                      -- Drops zero/negative adjustment rows		
  AND ca.activity_quantity > 0                   -- Drops invalid line quantities		
  AND ca.diagnosis_code IS NOT NULL              -- Drops 435 missing diagnosis rows		
  AND (ca.claim_route IS NULL OR ca.claim_route NOT IN ('Resubmission', 'Re-submission'));	



----------------phase 2 with the conversional scores------------------------

DROP TABLE IF EXISTS public.denial_prediction_features;				
				
CREATE TABLE public.denial_prediction_features AS				
				
-- 1. Medical Necessity Score				
WITH med_necessity_stats AS (				
    SELECT 				
        diagnosis_code,				
        activity_code,				
        ROUND(				
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC / 				
            NULLIF(COUNT(*), 0), 4				
        ) AS medical_necessity_score				
    FROM public.claim_activity				
    WHERE activity_denial_code IS NULL 				
       OR activity_denial_code LIKE 'MNEC%' 				
       OR activity_denied::text IN ('f', 'false', '0')				
    GROUP BY diagnosis_code, activity_code				
),				
				
-- 2. PA Risk Score				
pa_risk_stats AS (				
    SELECT 				
        payer_id,				
        diagnosis_code,				
        activity_code,				
        ROUND(				
            SUM(CASE WHEN activity_denial_code = 'AUTH-001' THEN 1 ELSE 0 END)::NUMERIC / 				
            NULLIF(COUNT(*), 0), 4				
        ) AS pa_risk_score				
    FROM public.claim_activity				
    GROUP BY payer_id, diagnosis_code, activity_code				
),				
				
-- 3. Coverage Score				
coverage_stats AS (				
    SELECT 				
        payer_id,				
        activity_code,				
        ROUND(				
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC / 				
            NULLIF(COUNT(*), 0), 4				
        ) AS coverage_score				
    FROM public.claim_activity				
    GROUP BY payer_id, activity_code				
),				
				
-- 4. Clinician Success Score				
clinician_stats AS (				
    SELECT 				
        COALESCE(activity_clinician, clinician_license) AS clinician_id,				
        ROUND(				
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC / 				
            NULLIF(COUNT(*), 0), 4				
        ) AS clinician_success_score				
    FROM public.claim_activity				
    WHERE COALESCE(activity_clinician, clinician_license) IS NOT NULL				
    GROUP BY COALESCE(activity_clinician, clinician_license)				
),				
				
-- 5. Facility Success Score				
facility_stats AS (				
    SELECT 				
        COALESCE(facility_license, facility_name) AS facility_id,				
        ROUND(				
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC / 				
            NULLIF(COUNT(*), 0), 4				
        ) AS facility_success_score				
    FROM public.claim_activity				
    WHERE COALESCE(facility_license, facility_name) IS NOT NULL				
    GROUP BY COALESCE(facility_license, facility_name)				
)				
				
#NAME?				
SELECT 				
    -- Clinical Features				
    COALESCE(ca.diagnosis_code, 'UNKNOWN') AS primary_diagnosis_code,				
    COALESCE(ca.activity_code, 'UNKNOWN') AS activity_code,				
    (COALESCE(ca.activity_code, 'UNK') || '_' || COALESCE(ca.diagnosis_code, 'UNK')) AS cpt_icd_pair,				
    				
    -- Calculated Scores				
    COALESCE(mns.medical_necessity_score, 0.5000) AS medical_necessity_score,				
    COALESCE(prs.pa_risk_score, 0.0000) AS pa_risk_score,				
    COALESCE(cov.coverage_score, 0.5000) AS coverage_score,				
    COALESCE(cls.clinician_success_score, 0.5000) AS clinician_success_score,				
    COALESCE(fas.facility_success_score, 0.5000) AS facility_success_score,				
    				
    -- Financial Features	min_patient_age	max_patient_age	avg_billing_lag	max_billing_lag
    COALESCE(ca.activity_gross, 0)::NUMERIC(12,2) AS activity_gross,	1	103	25.7	326
    COALESCE(ca.activity_quantity, 1)::INTEGER AS activity_quantity,				
    COALESCE(ca.claim_gross, 0)::NUMERIC(12,2) AS gross_amount,				
    COALESCE(ca.claim_net, 0)::NUMERIC(12,2) AS net_amount,				
    				
    -- Business Logic Rule				
    CASE 				
        WHEN COALESCE(ca.activity_gross, 0) >= 1000 OR COALESCE(ca.claim_gross, 0) >= 1000 THEN 1 				
        ELSE 0 				
    END AS pa_exceeds_1000_flag,				
    				
    -- Demographics				
    COALESCE(ca.patient_age, 0)::INTEGER AS patient_age,				
    COALESCE(NULLIF(ca.gender, ''), 'UNKNOWN') AS gender,				
    COALESCE(NULLIF(ca.nationality, ''), 'UNKNOWN') AS nationality,				
    				
    -- Context				
    COALESCE(ca.payer_id, 'UNKNOWN') AS payer_id,				
    COALESCE(NULLIF(ca.payer_classification, ''), 'Standard') AS insurance_plan_tier,				
    COALESCE(NULLIF(ca.clinician_profession, ''), 'UNKNOWN') AS profession,				
    COALESCE(NULLIF(ca.clinician_category, ''), 'UNKNOWN') AS category,				
    COALESCE(NULLIF(ca.facility_type, ''), 'UNKNOWN') AS facility_type_id,				
    				
    -- Operational Lag				
    CASE 				
        WHEN ca.date_submitted IS NOT NULL AND ca.datestamp IS NOT NULL 				
        THEN GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted - ca.datestamp)))::INTEGER				
        ELSE 0 				
    END AS billing_lag_days,				
    COALESCE(ca.length_of_stay, 0)::INTEGER AS length_of_stay,				
    COALESCE(NULLIF(ca.encounter_type, ''), 'OP') AS encounter_type,				
    				
    -- TARGET VARIABLE (1 = Denied, 0 = Approved)				
    CASE 				
        WHEN ca.activity_denied::text IN ('t', 'true', '1') THEN 1 				
        ELSE 0 				
    END AS claim_status				
				
FROM public.claim_activity ca				
				
LEFT JOIN med_necessity_stats mns 				
       ON ca.diagnosis_code = mns.diagnosis_code 				
      AND ca.activity_code = mns.activity_code				
				
LEFT JOIN pa_risk_stats prs 				
       ON ca.payer_id = prs.payer_id 				
      AND ca.diagnosis_code = prs.diagnosis_code 				
      AND ca.activity_code = prs.activity_code				
				
LEFT JOIN coverage_stats cov 				
       ON ca.payer_id = cov.payer_id 				
      AND ca.activity_code = cov.activity_code				
				
LEFT JOIN clinician_stats cls 				
       ON COALESCE(ca.activity_clinician, ca.clinician_license) = cls.clinician_id				
				
LEFT JOIN facility_stats fas 				
       ON COALESCE(ca.facility_license, ca.facility_name) = fas.facility_id;				
		


-----------------------------------cleaned query-----------query 3

DROP TABLE IF EXISTS public.denialclaims_features;

CREATE TABLE public.denialclaims_features AS

-- 0. Clean Base Data (Filters out noise, resubmissions, and invalid rows)
WITH cleaned_claims AS (
    SELECT *
    FROM public.claim_activity
    WHERE date_submitted IS NOT NULL 
      AND activity_gross > 0 
      AND activity_quantity > 0 
      AND diagnosis_code IS NOT NULL 
      AND (claim_route IS NULL OR claim_route NOT IN ('Resubmission', 'Re-submission'))
),

-- 1. Medical Necessity Score
med_necessity_stats AS (
    SELECT
        diagnosis_code,
        activity_code,
        ROUND(
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0), 4
        ) AS medical_necessity_score
    FROM cleaned_claims
    WHERE activity_denial_code IS NULL
       OR activity_denial_code LIKE 'MNEC%'
       OR activity_denied::text IN ('f', 'false', '0')
    GROUP BY diagnosis_code, activity_code
),

-- 2. PA Risk Score
pa_risk_stats AS (
    SELECT
        payer_id,
        diagnosis_code,
        activity_code,
        ROUND(
            SUM(CASE WHEN activity_denial_code = 'AUTH-001' THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0), 4
        ) AS pa_risk_score
    FROM cleaned_claims
    GROUP BY payer_id, diagnosis_code, activity_code
),

-- 3. Coverage Score
coverage_stats AS (
    SELECT
        payer_id,
        activity_code,
        ROUND(
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0), 4
        ) AS coverage_score
    FROM cleaned_claims
    GROUP BY payer_id, activity_code
),

-- 4. Clinician Success Score
clinician_stats AS (
    SELECT
        COALESCE(activity_clinician, clinician_license) AS clinician_id,
        ROUND(
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0), 4
        ) AS clinician_success_score
    FROM cleaned_claims
    WHERE COALESCE(activity_clinician, clinician_license) IS NOT NULL
    GROUP BY COALESCE(activity_clinician, clinician_license)
),

-- 5. Facility Success Score
facility_stats AS (
    SELECT
        COALESCE(facility_license, facility_name) AS facility_id,
        ROUND(
            SUM(CASE WHEN activity_denied::text IN ('f', 'false', '0') THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0), 4
        ) AS facility_success_score
    FROM cleaned_claims
    WHERE COALESCE(facility_license, facility_name) IS NOT NULL
    GROUP BY COALESCE(facility_license, facility_name)
)

-- MAIN OUTPUT
SELECT
    -- Clinical Features
    COALESCE(ca.diagnosis_code, 'UNKNOWN') AS primary_diagnosis_code,
    COALESCE(ca.activity_code, 'UNKNOWN') AS activity_code,
    (COALESCE(ca.activity_code, 'UNK') || '_' || COALESCE(ca.diagnosis_code, 'UNK')) AS cpt_icd_pair,
    
    -- Calculated Scores
    COALESCE(mns.medical_necessity_score, 0.5000) AS medical_necessity_score,
    COALESCE(prs.pa_risk_score, 0.0000) AS pa_risk_score,
    COALESCE(cov.coverage_score, 0.5000) AS coverage_score,
    COALESCE(cls.clinician_success_score, 0.5000) AS clinician_success_score,
    COALESCE(fas.facility_success_score, 0.5000) AS facility_success_score,
    
    -- Financial Features
    COALESCE(ca.activity_gross, 0)::NUMERIC(12,2) AS activity_gross,
    COALESCE(ca.activity_quantity, 1)::INTEGER AS activity_quantity,
    COALESCE(ca.claim_gross, 0)::NUMERIC(12,2) AS gross_amount,
    COALESCE(ca.claim_net, 0)::NUMERIC(12,2) AS net_amount,
    
    -- Business Logic Rule
    CASE
        WHEN COALESCE(ca.activity_gross, 0) >= 1000 OR COALESCE(ca.claim_gross, 0) >= 1000 THEN 1
        ELSE 0
    END AS pa_exceeds_1000_flag,
    
    -- Demographics
    COALESCE(ca.patient_age, 0)::INTEGER AS patient_age,
    COALESCE(NULLIF(ca.gender, ''), 'UNKNOWN') AS gender,
    COALESCE(NULLIF(ca.nationality, ''), 'UNKNOWN') AS nationality,
    
    -- Context
    COALESCE(ca.payer_id, 'UNKNOWN') AS payer_id,
    COALESCE(NULLIF(ca.payer_classification, ''), 'Standard') AS insurance_plan_tier,
    COALESCE(NULLIF(ca.clinician_profession, ''), 'UNKNOWN') AS profession,
    COALESCE(NULLIF(ca.clinician_category, ''), 'UNKNOWN') AS category,
    COALESCE(NULLIF(ca.facility_type, ''), 'UNKNOWN') AS facility_type_id,
    
    -- Operational Lag
    CASE
        WHEN ca.date_submitted IS NOT NULL AND ca.datestamp IS NOT NULL
        THEN GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted - ca.datestamp)))::INTEGER
        ELSE 0
    END AS billing_lag_days,
    COALESCE(ca.length_of_stay, 0)::INTEGER AS length_of_stay,
    COALESCE(NULLIF(ca.encounter_type, ''), 'OP') AS encounter_type,
    
    -- TARGET VARIABLE (1 = Denied, 0 = Approved)
    CASE
        WHEN ca.activity_denied::text IN ('t', 'true', '1') THEN 1
        ELSE 0
    END AS claim_status
    
FROM cleaned_claims ca

LEFT JOIN med_necessity_stats mns
       ON ca.diagnosis_code = mns.diagnosis_code
      AND ca.activity_code = mns.activity_code
      
LEFT JOIN pa_risk_stats prs
       ON ca.payer_id = prs.payer_id
      AND ca.diagnosis_code = prs.diagnosis_code
      AND ca.activity_code = prs.activity_code
      
LEFT JOIN coverage_stats cov
       ON ca.payer_id = cov.payer_id
      AND ca.activity_code = cov.activity_code
      
LEFT JOIN clinician_stats cls
       ON COALESCE(ca.activity_clinician, ca.clinician_license) = cls.clinician_id
       
LEFT JOIN facility_stats fas
       ON COALESCE(ca.facility_license, ca.facility_name) = fas.facility_id;