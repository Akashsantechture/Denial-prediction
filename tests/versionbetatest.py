import requests
import pandas as pd
import io

API_URL = "http://127.0.0.1:8000/predict_user_claim"

# Your extracted validation data (10 Approved 'f', 10 Denied 't')
csv_data = """primary_diagnosis_code,activity_code,activity_gross,activity_quantity,gross_amount,net_amount,patient_age,gender,nationality,payer_id,insurance_plan_tier,profession,category,facility_type_id,billing_lag_days,length_of_stay,encounter_type,activity_denied
R14.0,96375,40.00,1,1047.82,57.05,25,Male,EMIRATI,E001,Insurance,General Practitioner,General Practitioner,Hospital,0,0,OP,f
E55.9,82306,106.00,1,707.70,69.60,37,Female,EMIRATI,E001,Insurance,Internal Medicine,Endocrinology,Hospital,0,0,OP,f
E11.49,82607,64.80,1,481.70,481.70,51,Male,EMIRATI,E001,Insurance,Internal Medicine,Internal Medicine,Hospital,38,0,OP,f
R79.82,82570,73.60,1,1405.80,1194.93,44,Female,FILIPINO,INS012,Insurance,INTERNAL MEDICINE,Rheumatology,Hospital,58,0,OP,f
R53.81,84550,19.20,1,1395.72,1395.72,23,Male,EMIRATI,E001,Insurance,General Practitioner,General Practitioner,Hospital,0,0,OP,f
L21.9,EX1-A460-11103-01,108.50,1,355.80,355.80,42,Female,EMIRATI,E001,Insurance,Dermatology,Dermatology,Pharmacy,35,0,OP,f
Z79.84,84450,22.80,1,872.52,148.00,65,Female,EMIRATI,E001,Insurance,Internal Medicine,Endocrinology,Hospital,0,0,OP,f
K30,99283,177.62,1,1317.22,1317.22,48,Male,EMIRATI,E001,Insurance,General Practitioner,General Practitioner,Hospital,43,0,OP,f
M25.861,97140,108.00,1,326.00,326.00,39,Female,EMIRATI,E001,Insurance,Physiotherapy,Physiotherapist 1,Hospital,32,0,OP,f
I10,82043,25.20,1,1167.08,368.18,57,Female,EMIRATI,E001,Insurance,Internal Medicine,Endocrinology,Hospital,0,0,OP,f
R07.0,U90-7991-05180-04,1.20,1,2428.67,2428.67,30,Female,EMIRATI,E001,Insurance,General Practitioner,General Practitioner,Hospital,33,0,OP,t
R05,86140,22.80,1,620.42,409.20,16,Female,EMIRATI,E001,Insurance,Emergency Medicine,Emergency Medicine.,Hospital,0,0,OP,t
I48.20,82607,64.80,1,1598.52,1598.52,65,Male,EMIRATI,E001,Insurance,Internal Medicine,Endocrinology,Hospital,0,0,OP,t
R42,82947,16.80,1,910.28,302.80,24,Male,EMIRATI,E001,Insurance,Family Medicine,Family Medicine,Hospital,0,0,OP,t
A09,80061,78.00,1,2887.88,2887.88,26,Male,EMIRATI,E001,Insurance,Internal Medicine,Internal Medicine,Hospital,0,0,OP,t
G44.59,84132,20.40,1,616.28,616.28,40,Female,EMIRATI,E001,Insurance,Internal Medicine,Neurology,Hospital,37,0,OP,t
M79.10,99203,303.88,1,303.88,303.88,41,Male,EMIRATI,E001,Insurance,General Practitioner,General Practitioner,Hospital,39,0,OP,t
K21.9,82570,22.80,1,796.00,796.00,37,Male,EMIRATI,E001,Insurance,Internal Medicine,Endocrinology,Hospital,0,0,OP,t
M43.02,84450,22.80,1,615.88,615.88,42,Female,EMIRATI,E001,Insurance,Internal Medicine,Neurology,Hospital,43,0,OP,t
K59.09,82043,25.20,1,1277.70,1277.70,76,Male,EMIRATI,E001,Insurance,Internal Medicine,Internal Medicine,Hospital,32,0,OP,t"""

# Load CSV into Pandas
df = pd.read_csv(io.StringIO(csv_data))

# --- NEW: Map old CSV headers to the exact Schema/Model expected names ---
df = df.rename(columns={
    'primary_diagnosis_code': 'diagnosis_code',
    'profession': 'clinician_profession',
    'category': 'clinician_category',
    'facility_type_id': 'facility_type',
    'gross_amount': 'claim_gross',  # ADDED: Maps to the new Pydantic schema
    'net_amount': 'claim_net'       # ADDED: Maps to the new Pydantic schema
})

# Add missing 'diagnosis_type' required by the model
df['diagnosis_type'] = 'Principal'

print("Validating model against ground truth data...\n")
results = []

# Convert to list of dictionaries for clean JSON parsing
records = df.drop(columns=['activity_denied']).to_dict(orient='records')
ground_truths = df['activity_denied'].tolist()

for i in range(len(records)):
    payload = records[i]
    # 't' means denied, 'f' means approved in the source data
    actual_status = "Denied" if ground_truths[i] == "t" else "Approved"
    
    try:
        res = requests.post(API_URL, json=payload)
        
        if res.status_code == 200:
            data = res.json()
            pred_prob = data['denial_probability_pct']
            pred_risk = data['risk_level']
            
            # Simple match logic: If Actual is Denied, we want High risk. If Actual is Approved, we want Low/Medium.
            if (actual_status == "Denied" and pred_risk == "HIGH") or (actual_status == "Approved" and pred_risk != "HIGH"):
                match_status = "✅ MATCH"
            else:
                match_status = "❌ MISMATCH"

            results.append({
                "Claim #": i + 1,
                "Actual Result": actual_status,
                "Predicted Risk": pred_risk,
                "Probability": f"{pred_prob}%",
                "Validation": match_status
            })
        else:
            # Added error detail printing to help debug 422 Pydantic Validation errors
            print(f"Error on Claim {i+1}: {res.text}")
            results.append({
                "Claim #": i + 1,
                "Actual Result": actual_status,
                "Predicted Risk": f"ERR: {res.status_code}",
                "Probability": "N/A",
                "Validation": "⚠️ FAILED"
            })
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API. Make sure Uvicorn is running on port 8000.")
        exit()

# Display Results
df_results = pd.DataFrame(results)
print("="*65)
print("                  MODEL VALIDATION SUMMARY                       ")
print("="*65)
print(df_results.to_string(index=False))
print("="*65)

# Calculate Accuracy
matches = len(df_results[df_results['Validation'] == '✅ MATCH'])
total = len(df_results)
accuracy = (matches / total) * 100
print(f"\nModel Validation Accuracy on Sample: {accuracy:.1f}%\n")