# import streamlit as st
# import requests
# import json
# import plotly.graph_objects as go

# # ----------------------------------------------------
# # Page Configuration
# # ----------------------------------------------------
# st.set_page_config(
#     page_title="Santechture Claim Denial Predictor",
#     page_icon="🏥",
#     layout="wide"
# )

# # ----------------------------------------------------
# # Custom CSS
# # ----------------------------------------------------
# st.markdown("""
# <style>

# .main-title{
#     font-size:42px;
#     font-weight:700;
#     color:#FACC15;
#     margin-bottom:0px;
# }

# .sub-title{
#     color:#CBD5E1;
#     font-size:17px;
#     margin-bottom:25px;
# }

# .result-card{
#     padding:20px;
#     border-radius:10px;
#     background:#F8FAFC;
#     border:1px solid #E2E8F0;
#     margin-bottom:15px;
# }

# .small-title{
#     font-size:18px;
#     font-weight:600;
#     margin-bottom:10px;
# }

# </style>
# """, unsafe_allow_html=True)

# # ----------------------------------------------------
# # Header
# # ----------------------------------------------------
# st.markdown('<div class="main-title">Claim Denial Predictor</div>', unsafe_allow_html=True)

# st.markdown(
#     '<div class="sub-title">ML-powered pre-submission risk evaluation for Revenue Cycle Management- powered by  ROBIN</div>',
#     unsafe_allow_html=True
# )

# # ----------------------------------------------------
# # Default JSON Payload
# # ----------------------------------------------------

# default_payload = {
#     "primary_diagnosis_code": "I10",
#     "activity_code": "99213",
#     "activity_gross": 250.0,
#     "activity_quantity": 1,
#     "gross_amount": 250.0,
#     "net_amount": 250.0,
#     "patient_age": 52,
#     "gender": "MALE",
#     "nationality": "EMIRATI",
#     "payer_id": "E001",
#     "insurance_plan_tier": "Standard",
#     "profession": "Internal Medicine",
#     "category": "Internal Medicine",
#     "facility_type_id": "Clinic",
#     "billing_lag_days": 1,
#     "length_of_stay": 0,
#     "encounter_type": "OP"
# }

# # ----------------------------------------------------
# # Layout
# # ----------------------------------------------------

# left, right = st.columns([1.1, 0.9])

# # ====================================================
# # LEFT SIDE
# # ====================================================

# with left:

#     st.subheader("📄 JSON Payload")

#     json_text = st.text_area(
#         "JSON Payload",
#         value=json.dumps(default_payload, indent=4),
#         height=650,
#         label_visibility="collapsed"
#     )

#     submit = st.button(
#         "🚀 Predict Claim",
#         use_container_width=True,
#         type="primary"
#     )

# # ====================================================
# # RIGHT SIDE
# # ====================================================

# with right:

#     st.subheader("📊 Prediction Result")

#     if submit:

#         # -----------------------------
#         # Validate JSON
#         # -----------------------------

#         try:
#             payload = json.loads(json_text)

#         except json.JSONDecodeError as e:
#             st.error("Invalid JSON Payload")
#             st.code(str(e))
#             st.stop()

#         # -----------------------------
#         # API Call
#         # -----------------------------

#         try:

#             with st.spinner("Running prediction..."):

#                 response = requests.post(
#                     "http://127.0.0.1:8000/predict_user_claim",
#                     json=payload,
#                     timeout=10
#                 )

#             # =====================================
#             # SUCCESS
#             # =====================================

#             if response.status_code == 200:

#                 result = response.json()

#                 probability = result["denial_probability_pct"]
#                 risk = result["risk_level"]
#                 recommendation = result["action_recommendation"]

#                 # -------------------------
#                 # Gauge
#                 # -------------------------

#                 fig = go.Figure(go.Indicator(
#                     mode="gauge+number",
#                     value=probability,
#                     number={"suffix": "%"},
#                     title={"text": "Denial Probability"},
#                     gauge={
#                         "axis": {"range": [0, 100]},
#                         "bar": {"color": "#1E3A8A"},
#                         "steps": [
#                             {"range": [0, 35], "color": "#22C55E"},
#                             {"range": [35, 60], "color": "#F59E0B"},
#                             {"range": [60, 100], "color": "#EF4444"}
#                         ],
#                         "threshold": {
#                             "line": {"color": "black", "width": 4},
#                             "value": probability
#                         }
#                     }
#                 ))

#                 fig.update_layout(
#                     height=320,
#                     margin=dict(l=20, r=20, t=40, b=20)
#                 )

#                 st.plotly_chart(fig, use_container_width=True)

#                 # -------------------------
#                 # Risk Level
#                 # -------------------------

#                 if risk.upper() == "HIGH":
#                     st.error(f"🔴 Risk Level : {risk}")

#                 elif risk.upper() == "MEDIUM":
#                     st.warning(f"🟡 Risk Level : {risk}")

#                 else:
#                     st.success(f"🟢 Risk Level : {risk}")

#                 # -------------------------
#                 # Recommendation
#                 # -------------------------

#                 st.markdown("### 💡 Recommended Action")

#                 st.info(recommendation)

#                 # -------------------------
#                 # Raw API Response
#                 # -------------------------

#                 with st.expander("📄 Raw API Response", expanded=False):
#                     st.json(result)

#             # =====================================
#             # Validation Error
#             # =====================================

#             elif response.status_code == 422:

#                 st.error("Schema Validation Failed")

#                 error = response.json()

#                 if "message" in error:
#                     st.write(error["message"])

#                 if "errors" in error:

#                     for err in error["errors"]:

#                         st.error(
#                             f"""
# Field : {err.get("field")}

# Issue : {err.get("issue")}

# Provided : {err.get("provided_value")}
# """
#                         )

#                 else:
#                     st.json(error)

#             # =====================================
#             # Other Error
#             # =====================================

#             else:

#                 st.error(f"HTTP {response.status_code}")

#                 try:
#                     st.json(response.json())
#                 except:
#                     st.write(response.text)

#         except requests.exceptions.ConnectionError:

#             st.error("❌ Cannot connect to FastAPI server.")

#             st.info(
#                 "Ensure your backend is running on\n\n"
#                 "http://127.0.0.1:8000"
#             )

#         except requests.exceptions.Timeout:

#             st.error("Request timed out.")

#         except Exception as ex:

#             st.exception(ex)

#     else:

#         st.info("👈 Edit the JSON payload on the left and click **Predict Claim**.")


import json
import requests
import streamlit as st
import plotly.graph_objects as go

# ======================================================
# CONFIG
# ======================================================

API_URL = "http://127.0.0.1:8000/predict_user_claim"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="Healthcare Claim Denial Predictor",
    page_icon="🏥",
    layout="wide"
)

# ======================================================
# CSS
# ======================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.main-title{
    font-size:42px;
    font-weight:700;
    color:#2563EB;
}

.subtitle{
    font-size:17px;
    color:#6B7280;
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    '<div class="main-title">🏥 Healthcare Claim Denial Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Behavioral XGBoost v1.5 • Real-time Claim Risk Prediction</div>',
    unsafe_allow_html=True
)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Backend Status")

try:

    health = requests.get(HEALTH_URL, timeout=2)

    if health.status_code == 200:
        st.sidebar.success("🟢 Backend Connected")
        st.sidebar.json(health.json())
    else:
        st.sidebar.error("Backend Error")

except Exception:
    st.sidebar.error("🔴 Backend Offline")

# ======================================================
# DEFAULT PAYLOAD
# ======================================================

default_payload = {
    "diagnosis_code": "I10",
    "diagnosis_type": "PRIMARY",

    "activity_code": "99213",
    "activity_quantity": 1,
    "activity_gross": 250.0,

    "gross_amount": 250.0,
    "net_amount": 250.0,

    "patient_age": 52,
    "gender": "MALE",
    "nationality": "EMIRATI",

    "encounter_type": "OP",

    "clinician_profession": "Internal Medicine",
    "clinician_category": "Internal Medicine",

    "facility_type": "Clinic",

    "insurance_plan_tier": "Standard",

    "billing_lag_days": 1,
    "length_of_stay": 0
}

# ======================================================
# LAYOUT
# ======================================================

left, right = st.columns([1.2, 1])

# ======================================================
# LEFT
# ======================================================

with left:

    st.subheader("📄 Claim Payload")

    json_text = st.text_area(
        "",
        value=json.dumps(default_payload, indent=4),
        height=620
    )

    predict = st.button(
        "🚀 Predict Claim Risk",
        use_container_width=True,
        type="primary"
    )

# ======================================================
# RIGHT
# ======================================================

with right:

    st.subheader("Prediction")

    if not predict:
        st.info("Edit the JSON payload and click **Predict Claim Risk**.")
        st.stop()

    # --------------------------------------------------

    try:
        payload = json.loads(json_text)

    except Exception as e:
        st.error("Invalid JSON")
        st.code(str(e))
        st.stop()

    # --------------------------------------------------

    try:

        with st.spinner("Running Behavioral Model..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=20
            )

    except requests.exceptions.ConnectionError:

        st.error("Cannot connect to FastAPI server.")
        st.stop()

    except requests.exceptions.Timeout:

        st.error("Request Timed Out")
        st.stop()

    # ==================================================
    # SUCCESS
    # ==================================================

    if response.status_code == 200:

        result = response.json()

        probability = result["denial_probability_pct"]
        risk = result["risk_level"]
        recommendation = result["action_recommendation"]

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=probability,

            number={"suffix":"%"},

            title={"text":"Denial Probability"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"#2563EB"},

                "steps":[

                    {"range":[0,22],"color":"#22C55E"},

                    {"range":[22,47],"color":"#FBBF24"},

                    {"range":[47,100],"color":"#EF4444"}

                ],

                "threshold":{

                    "line":{"color":"black","width":4},

                    "value":probability

                }

            }

        ))

        fig.update_layout(height=340)

        st.plotly_chart(fig, use_container_width=True)

        if risk.upper() == "HIGH":

            st.error(f"🔴 Risk Level : {risk}")

        elif risk.upper() == "MEDIUM":

            st.warning(f"🟡 Risk Level : {risk}")

        else:

            st.success(f"🟢 Risk Level : {risk}")

        st.markdown("### 💡 Recommendation")

        st.info(recommendation)

        st.markdown("### 📄 Response")

        st.json(result)

    # ==================================================
    # VALIDATION
    # ==================================================

    elif response.status_code == 422:

        error = response.json()

        st.error(error.get("message","Validation Failed"))

        if "errors" in error:

            st.markdown("### Validation Errors")

            for err in error["errors"]:

                st.error(
                    f"""
Field : **{err.get('field')}**

Issue : {err.get('issue')}

Provided Value :

`{err.get('provided_value')}`
"""
                )

        else:

            st.json(error)

    # ==================================================
    # OTHER
    # ==================================================

    else:

        st.error(f"HTTP {response.status_code}")

        try:
            st.json(response.json())
        except:
            st.code(response.text)