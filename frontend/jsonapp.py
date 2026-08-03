import streamlit as st
import requests
import json
import plotly.graph_objects as go

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Santechture Claim Denial Predictor",
    page_icon="🏥",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------
st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:700;
    color:#FACC15;
    margin-bottom:0px;
}

.sub-title{
    color:#CBD5E1;
    font-size:17px;
    margin-bottom:25px;
}

.result-card{
    padding:20px;
    border-radius:10px;
    background:#F8FAFC;
    border:1px solid #E2E8F0;
    margin-bottom:15px;
}

.small-title{
    font-size:18px;
    font-weight:600;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown('<div class="main-title">Claim Denial Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">ML-powered pre-submission risk evaluation for Revenue Cycle Management- powered by  ROBIN</div>',
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Default JSON Payload
# ----------------------------------------------------

default_payload = {
    "primary_diagnosis_code": "I10",
    "activity_code": "99213",
    "activity_gross": 250.0,
    "activity_quantity": 1,
    "gross_amount": 250.0,
    "net_amount": 250.0,
    "patient_age": 52,
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

# ----------------------------------------------------
# Layout
# ----------------------------------------------------

left, right = st.columns([1.1, 0.9])

# ====================================================
# LEFT SIDE
# ====================================================

with left:

    st.subheader("📄 JSON Payload")

    json_text = st.text_area(
        "JSON Payload",
        value=json.dumps(default_payload, indent=4),
        height=650,
        label_visibility="collapsed"
    )

    submit = st.button(
        "🚀 Predict Claim",
        use_container_width=True,
        type="primary"
    )

# ====================================================
# RIGHT SIDE
# ====================================================

with right:

    st.subheader("📊 Prediction Result")

    if submit:

        # -----------------------------
        # Validate JSON
        # -----------------------------

        try:
            payload = json.loads(json_text)

        except json.JSONDecodeError as e:
            st.error("Invalid JSON Payload")
            st.code(str(e))
            st.stop()

        # -----------------------------
        # API Call
        # -----------------------------

        try:

            with st.spinner("Running prediction..."):

                response = requests.post(
                    "http://127.0.0.1:8000/predict_user_claim",
                    json=payload,
                    timeout=10
                )

            # =====================================
            # SUCCESS
            # =====================================

            if response.status_code == 200:

                result = response.json()

                probability = result["denial_probability_pct"]
                risk = result["risk_level"]
                recommendation = result["action_recommendation"]

                # -------------------------
                # Gauge
                # -------------------------

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    number={"suffix": "%"},
                    title={"text": "Denial Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1E3A8A"},
                        "steps": [
                            {"range": [0, 35], "color": "#22C55E"},
                            {"range": [35, 60], "color": "#F59E0B"},
                            {"range": [60, 100], "color": "#EF4444"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "value": probability
                        }
                    }
                ))

                fig.update_layout(
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # -------------------------
                # Risk Level
                # -------------------------

                if risk.upper() == "HIGH":
                    st.error(f"🔴 Risk Level : {risk}")

                elif risk.upper() == "MEDIUM":
                    st.warning(f"🟡 Risk Level : {risk}")

                else:
                    st.success(f"🟢 Risk Level : {risk}")

                # -------------------------
                # Recommendation
                # -------------------------

                st.markdown("### 💡 Recommended Action")

                st.info(recommendation)

                # -------------------------
                # Raw API Response
                # -------------------------

                with st.expander("📄 Raw API Response", expanded=False):
                    st.json(result)

            # =====================================
            # Validation Error
            # =====================================

            elif response.status_code == 422:

                st.error("Schema Validation Failed")

                error = response.json()

                if "message" in error:
                    st.write(error["message"])

                if "errors" in error:

                    for err in error["errors"]:

                        st.error(
                            f"""
Field : {err.get("field")}

Issue : {err.get("issue")}

Provided : {err.get("provided_value")}
"""
                        )

                else:
                    st.json(error)

            # =====================================
            # Other Error
            # =====================================

            else:

                st.error(f"HTTP {response.status_code}")

                try:
                    st.json(response.json())
                except:
                    st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error("❌ Cannot connect to FastAPI server.")

            st.info(
                "Ensure your backend is running on\n\n"
                "http://127.0.0.1:8000"
            )

        except requests.exceptions.Timeout:

            st.error("Request timed out.")

        except Exception as ex:

            st.exception(ex)

    else:

        st.info("👈 Edit the JSON payload on the left and click **Predict Claim**.")