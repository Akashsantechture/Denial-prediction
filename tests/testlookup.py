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
        "scenario_name": "5. Brand New Provider (Tests Memory Fallbacks)",
        "endpoint": "/predict_user_claim",
        "payload": {
            "primary_diagnosis_code": "Z99.8",
            "activity_code": "NEW_CODE_01",
            "activity_gross": 200.00,
            "activity_quantity": 1,
            "gross_amount": 200.00,
            "net_amount": 200.00,
            "patient_age": 50,
            "gender": "MALE",
            "nationality": "EMIRATI",
            "payer_id": "UNKNOWN_PAYER",
            "insurance_plan_tier": "Standard",
            "profession": "General Practice",
            "category": "General Practice",
            "facility_type_id": "Clinic",
            "billing_lag_days": 1,
            "length_of_stay": 0,
            "encounter_type": "OP"
        }
    },
    {
        "scenario_name": "6. Bad Data (Tests Schema Error Handler)",
        "endpoint": "/predict_user_claim",
        "payload": {
            # Intentionally missing primary_diagnosis_code
            "activity_code": "99213",
            "activity_gross": -50.00, # Intentionally negative
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
    }
]

print("Sending test payloads to the API...")
results = []

for s in scenarios:
    url = f"{API_URL}{s['endpoint']}"
    try:
        res = requests.post(url, json=s['payload'])
        
        if res.status_code == 200:
            data = res.json()
            results.append({
                "Scenario": s["scenario_name"],
                "Status": "✅ SUCCESS",
                "Probability": f"{data['denial_probability_pct']}%",
                "Risk Level": data["risk_level"],
                "Notes": data["action_recommendation"][:45] + "..."
            })
        elif res.status_code == 422: # Pydantic Validation Error
            data = res.json()
            error_msg = data.get("errors", [{"issue": "Unknown validation error"}])[0]["issue"]
            results.append({
                "Scenario": s["scenario_name"],
                "Status": "⚠️ REJECTED (422)",
                "Probability": "N/A",
                "Risk Level": "N/A",
                "Notes": error_msg[:45] + "..."
            })
        else:
            results.append({
                "Scenario": s["scenario_name"],
                "Status": f"❌ ERROR ({res.status_code})",
                "Probability": "ERR",
                "Risk Level": "ERR",
                "Notes": res.text[:45] + "..."
            })
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API. Is Uvicorn running?")
        exit()

df_results = pd.DataFrame(results)
print("\n" + "="*85)
print("                       API EVALUATION SUMMARY (LOCAL TEST)                       ")
print("="*85)
print(df_results.to_string(index=False))
print("="*85 + "\n")