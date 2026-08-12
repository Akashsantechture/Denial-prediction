-- STEP 1: Add the new columns for our Clinical Consistency Engine
ALTER TABLE denialpredictionbeta
ADD COLUMN icd_category VARCHAR(50),
ADD COLUMN cpt_category VARCHAR(50),
ADD COLUMN icd_cpt_domain_match VARCHAR(100);

-- STEP 2: Populate categories using ICD letters and CPT numeric ranges
UPDATE denialpredictionbeta
SET 
    -- Map Diagnosis Codes (ICD-10) based on the starting letter
    icd_category = CASE
        WHEN LEFT(diagnosis_code, 1) IN ('A', 'B') THEN 'Infectious'
        WHEN LEFT(diagnosis_code, 1) = 'C' THEN 'Oncology'
        WHEN LEFT(diagnosis_code, 1) = 'D' THEN 'Oncology_Hematology'
        WHEN LEFT(diagnosis_code, 1) = 'E' THEN 'Endocrinology'
        WHEN LEFT(diagnosis_code, 1) = 'F' THEN 'Psychiatry'
        WHEN LEFT(diagnosis_code, 1) = 'G' THEN 'Neurology'
        WHEN LEFT(diagnosis_code, 1) = 'H' THEN 'Eye_Ear'
        WHEN LEFT(diagnosis_code, 1) = 'I' THEN 'Cardiology'
        WHEN LEFT(diagnosis_code, 1) = 'J' THEN 'Pulmonology'
        WHEN LEFT(diagnosis_code, 1) = 'K' THEN 'Gastroenterology_Dental'
        WHEN LEFT(diagnosis_code, 1) = 'L' THEN 'Dermatology'
        WHEN LEFT(diagnosis_code, 1) = 'M' THEN 'Musculoskeletal'
        WHEN LEFT(diagnosis_code, 1) = 'N' THEN 'Genitourinary'
        WHEN LEFT(diagnosis_code, 1) = 'O' THEN 'OBGYN'
        WHEN LEFT(diagnosis_code, 1) = 'P' THEN 'Pediatrics'
        WHEN LEFT(diagnosis_code, 1) = 'Q' THEN 'Congenital'
        WHEN LEFT(diagnosis_code, 1) = 'R' THEN 'General_Symptoms'
        WHEN LEFT(diagnosis_code, 1) IN ('S', 'T') THEN 'Trauma_Burns_Poisoning'
        WHEN LEFT(diagnosis_code, 1) IN ('V', 'W', 'X', 'Y') THEN 'External_Causes'
        WHEN LEFT(diagnosis_code, 1) = 'Z' THEN 'Factors_Influencing_Health_Status'
        ELSE 'Unknown_ICD'
    END,

    -- Map Activity Codes (CPT) using regex to check if it's a number, then checking range
    cpt_category = CASE
        WHEN activity_code ~ '^[0-9]+$' THEN
            CASE
                WHEN CAST(activity_code AS INTEGER) BETWEEN 100 AND 1999 THEN 'Anesthesia'
                WHEN CAST(activity_code AS INTEGER) BETWEEN 10000 AND 69999 THEN 'Surgery'
                WHEN CAST(activity_code AS INTEGER) BETWEEN 70000 AND 79999 THEN 'Radiology'
                WHEN CAST(activity_code AS INTEGER) BETWEEN 80000 AND 89999 THEN 'Pathology_Laboratory'
                WHEN CAST(activity_code AS INTEGER) BETWEEN 99202 AND 99499 THEN 'Evaluation_Management'
                WHEN CAST(activity_code AS INTEGER) >= 90000 THEN 'Medicine'
                ELSE 'Unknown_CPT'
            END
        ELSE 'HCPCS_Supplies' -- Captures all alphanumeric codes (e.g., J-codes, EX1)
    END;

-- STEP 3: Concatenate the domains to create the ultimate behavioral feature
UPDATE denialpredictionbeta
SET icd_cpt_domain_match = icd_category || '_' || cpt_category;

-- STEP 4: Drop the obsolete financial ratios & 0-variance plan tier column
ALTER TABLE denialpredictionbeta
DROP COLUMN unit_cost,
DROP COLUMN unit_cost_log,
DROP COLUMN discount_ratio,
DROP COLUMN ip_stay_cost_ratio,
DROP COLUMN insurance_plan_tier;