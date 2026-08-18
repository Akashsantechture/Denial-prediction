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
    		
    -- 6. Prior Authorization risk Proxy Features		
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
				
-- 2. PA risk Score				
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

-- 2. PA risk Score
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



-------------------------------data retrival for test data from db

--------------------approved filter 

SELECT json_build_object(
    'primary_diagnosis_code', diagnosis_code,
    'activity_code', activity_code,
    'activity_gross', activity_gross,
    'activity_quantity', activity_quantity,
    'gross_amount', claim_gross,
    'net_amount', claim_net,
    'patient_age', patient_age,
    'gender', UPPER(gender), -- Converts "Male" to "MALE" to match schema
    'nationality', nationality,
    'payer_id', payer_id,
    'insurance_plan_tier', COALESCE(payer_classification, 'Standard'), -- Fallback if null
    'profession', clinician_profession,
    'category', clinician_category,
    'facility_type_id', facility_type,
    'billing_lag_days', GREATEST(0, EXTRACT(DAY FROM (date_submitted::timestamp - datestamp::timestamp))), -- Calculates days
    'length_of_stay', length_of_stay,
    'encounter_type', encounter_type
) AS api_payload
FROM 
    public.claim_activity
WHERE 
    -- 1. Target Variable: Approved claims (using 'f' based on your sample data)
    activity_denied = 'f' 
    
    -- 2. Completeness: Ensure no missing critical codes or dates
    AND diagnosis_code IS NOT NULL 
    AND date_submitted IS NOT NULL 
    AND datestamp IS NOT NULL
    
    -- 3. Financial Validity: Gross and quantity must be positive
    AND activity_gross > 0 
    AND activity_quantity > 0 
    
    -- 4. Route Isolation: Exclude reworked/resubmitted claims
    AND claim_route NOT IN ('Resubmission', 'Re-submission')
    
ORDER BY 
    RANDOM()
LIMIT 10;



------------------------denied positive

SELECT json_build_object(
    'primary_diagnosis_code', diagnosis_code,
    'activity_code', activity_code,
    'activity_gross', activity_gross,
    'activity_quantity', activity_quantity,
    'gross_amount', claim_gross,
    'net_amount', claim_net,
    'patient_age', patient_age,
    'gender', UPPER(gender), 
    'nationality', nationality,
    'payer_id', payer_id,
    'insurance_plan_tier', COALESCE(payer_classification, 'Standard'), 
    'profession', clinician_profession,
    'category', clinician_category,
    'facility_type_id', facility_type,
    'billing_lag_days', GREATEST(0, EXTRACT(DAY FROM (date_submitted::timestamp - datestamp::timestamp))),
    'length_of_stay', length_of_stay,
    'encounter_type', encounter_type
) AS api_payload
FROM 
    public.claim_activity
WHERE 
    -- 1. Target Variable: Denied claims ('t' based on your provided schema)
    activity_denied = 't' 
    
    -- 2. Completeness: Ensure no missing critical codes or dates
    AND diagnosis_code IS NOT NULL 
    AND date_submitted IS NOT NULL 
    AND datestamp IS NOT NULL
    
    -- 3. Financial Validity: Gross and quantity must be positive
    AND activity_gross > 0 
    AND activity_quantity > 0 
    
    -- 4. Route Isolation: Exclude reworked/resubmitted claims
    AND claim_route NOT IN ('Resubmission', 'Re-submission')
    
ORDER BY 
    RANDOM()
LIMIT 10;




----------------------------------------------------denialmodel1.5-----------------------------

CREATE TABLE public."denialclaims1.5" AS
WITH PayerriskTiers AS (
    -- CTE 1: Calculate the Payer Tiers dynamically without exposing the Payer ID
    SELECT 
        payer_id,
        CASE 
            WHEN ROUND(SUM(CASE WHEN activity_denied = 't' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) >= 25.00 THEN 'Tier 1 (Strict)'
            WHEN ROUND(SUM(CASE WHEN activity_denied = 't' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) >= 10.00 THEN 'Tier 2 (Standard)'
            ELSE 'Tier 3 (Lenient)'
        END AS payer_risk_tier
    FROM 
        public.claim_activity
    GROUP BY 
        payer_id
),
ClaimComplexity AS (
    -- CTE 2: Calculate how many lines (activities) are on each individual claim
    SELECT 
        haad_claim_id,
        COUNT(activity_code) AS claim_line_count
    FROM 
        public.claim_activity
    GROUP BY 
        haad_claim_id
)

SELECT 
    -- 0. Target Variable
    ca.activity_denied,

    -- 1. Generalized Dimensions (NO IDs included)
    prt.payer_risk_tier,
    ca.encounter_type,

    -- 2. Clinical Consistency Behaviors
    CASE 
        -- Maternity/Pregnancy codes billed for a Male
        WHEN (ca.diagnosis_code LIKE 'O%' OR ca.activity_code BETWEEN '59000' AND '59899') AND UPPER(ca.gender) = 'MALE' THEN 0
        -- Prostate/Male Genital codes billed for a Female
        WHEN (ca.activity_code BETWEEN '54000' AND '55899') AND UPPER(ca.gender) = 'FEMALE' THEN 0
        ELSE 1
    END AS gender_appropriate_flag,
    
    CASE 
        -- Maternity codes for patients under 12 or over 60
        WHEN ca.diagnosis_code LIKE 'O%' AND (ca.patient_age < 12 OR ca.patient_age > 60) THEN 0
        ELSE 1
    END AS age_appropriate_flag,
    
    CASE 
        -- Respiratory ICD (J-codes) matching Respiratory CPT (30xxx - 32xxx)
        WHEN ca.diagnosis_code LIKE 'J%' AND ca.activity_code BETWEEN '30000' AND '32999' THEN 1
        -- Cardiovascular ICD (I-codes) matching Cardiovascular CPT (33xxx - 37xxx)
        WHEN ca.diagnosis_code LIKE 'I%' AND ca.activity_code BETWEEN '33000' AND '37799' THEN 1
        -- Routine checks (Z-codes) are universally acceptable
        WHEN ca.diagnosis_code LIKE 'Z%' THEN 1
        ELSE 0 
    END AS icd_cpt_chapter_match,

    -- 3. Financial Behaviors
    ROUND(CAST(ca.activity_gross / NULLIF(ca.activity_quantity, 0) AS numeric), 2) AS unit_cost,
    
    ROUND(CAST((ca.claim_gross - ca.claim_net) / NULLIF(ca.claim_gross, 0) AS numeric), 4) AS discount_ratio,
    
    CASE 
        WHEN ca.encounter_type = 'IP' THEN ROUND(CAST(ca.claim_gross / NULLIF(ca.length_of_stay, 0) AS numeric), 2)
        ELSE 0 
    END AS ip_stay_to_cost_ratio,

    -- 4. Operational Behaviors (Excluding Weekend logic as requested)
    cc.claim_line_count,
    
    CASE 
        WHEN GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted::timestamp - ca.datestamp::timestamp))) = 0 THEN 'Same Day'
        WHEN GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted::timestamp - ca.datestamp::timestamp))) BETWEEN 1 AND 3 THEN 'Standard'
        WHEN GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted::timestamp - ca.datestamp::timestamp))) BETWEEN 4 AND 14 THEN 'Delayed'
        ELSE 'Extreme'
    END AS lag_day_severity

FROM 
    public.claim_activity ca
LEFT JOIN 
    PayerriskTiers prt ON ca.payer_id = prt.payer_id
LEFT JOIN 
    ClaimComplexity cc ON ca.haad_claim_id = cc.haad_claim_id
WHERE 
    -- Base cleanup to ensure valid data rows
    ca.activity_gross > 0 
    AND ca.activity_quantity > 0 
    AND ca.claim_route NOT IN ('Resubmission', 'Re-submission')
    AND ca.diagnosis_code IS NOT NULL;


-------------------------------------------------------------------version1.5 beta-----------------------------------------------------------------------------------



CREATE TABLE public."denialprediction1.5beta" AS
WITH CostThreshold AS (
    -- CTE: Dynamically calculate the 90th percentile for the high_cost_flag
    SELECT 
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY activity_gross) AS threshold_90
    FROM public.claim_activity
    WHERE activity_gross > 0
)

SELECT 
    -- 0. TARGET (Binary Not/f -> 1/0)
    CASE WHEN ca.activity_denied = 't' THEN 1 ELSE 0 END AS activity_denied,

    -- 1. CLINICAL (Categorical & Numeric Raw)
    ca.diagnosis_code,
    ca.diagnosis_type,
    ca.activity_code,
    ca.activity_quantity,

    -- 2. PATIENT (Numeric & Categorical)
    ca.patient_age,
    ca.gender,
    ca.nationality,

    -- 3. FINANCIAL (Raw)
    ca.activity_gross,
    ca.claim_gross,
    ca.claim_net,

    -- 4. FINANCIAL (Engineered Formulas)
    ROUND(CAST(ca.activity_gross / NULLIF(ca.activity_quantity, 0) AS numeric), 2) AS unit_cost,
    ROUND(CAST((ca.claim_gross - ca.claim_net) / NULLIF(ca.claim_gross, 0) AS numeric), 4) AS discount_ratio,
    
    -- log1p logic: LN(value + 1). Wrapped in GREATEST(0, x) to prevent negative log errors
    ROUND(CAST(LN(GREATEST(0, ca.activity_gross) + 1) AS numeric), 4) AS activity_gross_log,
    ROUND(CAST(LN(GREATEST(0, ca.claim_gross) + 1) AS numeric), 4) AS claim_gross_log,
    ROUND(CAST(LN(GREATEST(0, ca.claim_net) + 1) AS numeric), 4) AS claim_net_log,
    ROUND(CAST(LN(GREATEST(0, (ca.activity_gross / NULLIF(ca.activity_quantity, 0))) + 1) AS numeric), 4) AS unit_cost_log,
    
    -- high_cost_flag using 90th percentile from CTE
    CASE WHEN ca.activity_gross > ct.threshold_90 THEN 1 ELSE 0 END AS high_cost_flag,
    
    -- ip_stay_cost_ratio using max(length_of_stay, 1) mapped as GREATEST
    CASE WHEN ca.encounter_type = 'IP' THEN ROUND(CAST(ca.claim_gross / GREATEST(ca.length_of_stay, 1) AS numeric), 2) ELSE 0 END AS ip_stay_cost_ratio,

    -- 5. OPERATIONAL
    ca.encounter_type,
    ca.length_of_stay,
    GREATEST(0, EXTRACT(DAY FROM (ca.date_submitted::timestamp - ca.datestamp::timestamp))) AS billing_lag_days,

    -- 6. PROVIDER
    ca.clinician_profession,
    ca.clinician_category,
    ca.facility_type,

    -- 7. INSURANCE
    'Insurance' AS insurance_plan_tier

FROM 
    public.claim_activity ca
CROSS JOIN 
    CostThreshold ct
WHERE 
    ca.activity_gross > 0 
    AND ca.activity_quantity > 0 
    AND ca.claim_route NOT IN ('Resubmission', 'Re-submission')
    AND ca.diagnosis_code IS NOT NULL;