import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

scenarios = [
    {
        "scenario_name": "1. Routine OP (Clean / Low Risk)",
        "endpoint": "/predict_user_claim",
        "payload": {
            "primary_diagnosis_code": "I10",
            "activity_code": "99213",
            "activity_gross": 180.00,
            "activity_quantity": 1,
            "gross_amount": 180.00,
            "net_amount": 180.00,
            "patient_age": 45,
            "gender": "MALE",
            "nationality": "EMIRATI",
            "payer_id": "E001",
            "insurance_plan_tier": "Standard",
            "profession": "Internal Medicine",
            "category": "Internal Medicine",
            "facility_type_id": "Clinic",
            "billing_lag_days": 1,
            "length_of_stay": 0,
            "encounter_type": "OP"
        }
    },
    {
        "scenario_name": "2. High-Cost Inpatient (Prior Auth Risk)",
        "endpoint": "/predict_user_claim",
        "payload": {
            "primary_diagnosis_code": "E11.22",
            "activity_code": "84432",
            "activity_gross": 4500.00,
            "activity_quantity": 1,
            "gross_amount": 4500.00,
            "net_amount": 4000.00,
            "patient_age": 62,
            "gender": "FEMALE",
            "nationality": "INDIAN",
            "payer_id": "INS026",
            "insurance_plan_tier": "VIP",
            "profession": "Nephrology",
            "category": "Nephrology",
            "facility_type_id": "Hospital",
            "billing_lag_days": 12,
            "length_of_stay": 3,
            "encounter_type": "IP"
        }
    },
    {
        "scenario_name": "3. Delayed Billing (Administrative Lag)",
        "endpoint": "/predict_user_claim",
        "payload": {
            "primary_diagnosis_code": "M54.5",
            "activity_code": "97110",
            "activity_gross": 350.00,
            "activity_quantity": 2,
            "gross_amount": 700.00,
            "net_amount": 700.00,
            "patient_age": 34,
            "gender": "MALE",
            "nationality": "EMIRATI",
            "payer_id": "E001",
            "insurance_plan_tier": "Standard",
            "profession": "Physiotherapy",
            "category": "Physiotherapy And Rehabilitation",
            "facility_type_id": "Clinic",
            "billing_lag_days": 45,
            "length_of_stay": 0,
            "encounter_type": "OP"
        }
    },
    {
        "scenario_name": "4. Emergency Encounter (Uncertain Coverage)",
        "endpoint": "/predict_user_claim",
        "payload": {
            "primary_diagnosis_code": "J45.909",
            "activity_code": "99284",
            "activity_gross": 1250.00,
            "activity_quantity": 1,
            "gross_amount": 1250.00,
            "net_amount": 1250.00,
            "patient_age": 28,
            "gender": "FEMALE",
            "nationality": "OTHERS",
            "payer_id": "INS026",
            "insurance_plan_tier": "Basic",
            "profession": "Pulmonology",
            "category": "Pulmonology",
            "facility_type_id": "Hospital",
            "billing_lag_days": 2,
            "length_of_stay": 1,
            "encounter_type": "EM"
        }
    },
    {
        "scenario_name": "5. Developer Endpoint (Validated High Scores)",
        "endpoint": "/predict",
        "payload": {
            "primary_diagnosis_code": "I10",
            "activity_code": "99213",
            "medical_necessity_score": 0.9850,
            "pa_risk_score": 0.0010,
            "coverage_score": 0.9600,
            "clinician_success_score": 0.9500,
            "facility_success_score": 0.9400,
            "activity_gross": 200.00,
            "activity_quantity": 1,
            "gross_amount": 200.00,
            "net_amount": 200.00,
            "patient_age": 50,
            "gender": "MALE",
            "nationality": "EMIRATI",
            "payer_id": "E001",
            "insurance_plan_tier": "Standard",
            "profession": "Internal Medicine",
            "category": "Internal Medicine",
            "facility_type_id": "Clinic",
            "billing_lag_days": 1,
            "length_of_stay": 0,
            "encounter_type": "OP"
        }
    },
    {
        "scenario_name": "6. Developer Endpoint (Low Necessity / High PA Risk)",
        "endpoint": "/predict",
        "payload": {
            "primary_diagnosis_code": "E11.22",
            "activity_code": "84432",
            "medical_necessity_score": 0.1200,
            "pa_risk_score": 0.8900,
            "coverage_score": 0.2500,
            "clinician_success_score": 0.4000,
            "facility_success_score": 0.4500,
            "activity_gross": 3200.00,
            "activity_quantity": 1,
            "gross_amount": 3200.00,
            "net_amount": 3000.00,
            "patient_age": 60,
            "gender": "FEMALE",
            "nationality": "INDIAN",
            "payer_id": "INS026",
            "insurance_plan_tier": "Basic",
            "profession": "Nephrology",
            "category": "Nephrology",
            "facility_type_id": "Hospital",
            "billing_lag_days": 20,
            "length_of_stay": 2,
            "encounter_type": "IP"
        }
    }
]

results = []
for s in scenarios:
    url = f"{API_URL}{s['endpoint']}"
    res = requests.post(url, json=s['payload'])
    
    if res.status_code == 200:
        data = res.json()
        results.append({
            "Scenario": s["scenario_name"],
            "Probability (%)": f"{data['denial_probability_pct']}%",
            "Risk Level": data["risk_level"],
            "Is High Risk": data["is_high_risk"],
            "Action Recommendation": data["action_recommendation"][:45] + "..."
        })
    else:
        results.append({
            "Scenario": s["scenario_name"],
            "Probability (%)": "ERR",
            "Risk Level": "ERROR",
            "Is High Risk": False,
            "Action Recommendation": f"HTTP {res.status_code}: {res.text}"
        })

df_results = pd.DataFrame(results)
print("\n" + "="*80)
print("                    UPDATED API EVALUATION SUMMARY")
print("="*80)
print(df_results.to_string(index=False))