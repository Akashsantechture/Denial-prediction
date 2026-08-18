import streamlit as st
import requests
import plotly.graph_objects as go

# Configure Page
st.set_page_config(
    page_title="RCM Claim Denial Predictor",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FAfa33;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color:#FEF2F2;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .card-box {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.2rem;
    }
    .validation-error {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .error-field {
        font-weight: 600;
        color: #991B1B;
    }
    .error-message {
        color: #7F1D1D;
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">Santechture Denial Prediction </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-powered pre-submission risk evaluation for revenue cycle management compliance</div>', unsafe_allow_html=True)

# Define Form Inputs using Streamlit Layout Columns
with st.form("claim_form"):
    st.markdown("### Claim Information Entry")
    st.caption("Complete all required fields to evaluate denial risk probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        primary_diagnosis_code = st.text_input("Primary ICD Diagnosis", value="I10", help="e.g. I10, E11.22, M54.5")
        activity_code = st.text_input("CPT Activity Code", value="99213", help="e.g. 99213, 84432, 99284")
        encounter_type = st.selectbox("Encounter Type", options=["OP", "IP", "EM"], index=0)
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=52)
        gender = st.selectbox("Gender", options=["MALE", "FEMALE", "UNKNOWN"], index=0)
        nationality = st.selectbox("Nationality", options=["EMIRATI", "INDIAN", "OTHERS"], index=0)

    with col2:
        activity_gross = st.number_input("Activity Gross ($)", min_value=0.0, value=250.00, step=10.0)
        gross_amount = st.number_input("Gross Amount ($)", min_value=0.0, value=250.00, step=10.0)
        net_amount = st.number_input("Net Amount ($)", min_value=0.0, value=250.00, step=10.0)
        activity_quantity = st.number_input("Activity Quantity", min_value=1, value=1)
        billing_lag_days = st.number_input("Billing Lag (Days)", min_value=0, value=1)
        length_of_stay = st.number_input("Length of Stay (Days)", min_value=0, value=0)

    with col3:
        payer_id = st.text_input("Payer ID", value="E001")
        insurance_plan_tier = st.selectbox("Insurance Plan Tier", options=["Standard", "Basic", "VIP"], index=0)
        profession = st.text_input("Provider Profession", value="Internal Medicine")
        category = st.text_input("Specialty Category", value="Internal Medicine")
        facility_type_id = st.selectbox("Facility Type", options=["Clinic", "Hospital"], index=0)

    submit_button = st.form_submit_button("Evaluate Claim risk", use_container_width=True, type="primary")

# Trigger Evaluation on Form Submit
if submit_button:
    payload = {
        "primary_diagnosis_code": primary_diagnosis_code,
        "activity_code": activity_code,
        "activity_gross": float(activity_gross),
        "activity_quantity": int(activity_quantity),
        "gross_amount": float(gross_amount),
        "net_amount": float(net_amount),
        "patient_age": int(patient_age),
        "gender": gender,
        "nationality": nationality,
        "payer_id": payer_id,
        "insurance_plan_tier": insurance_plan_tier,
        "profession": profession,
        "category": category,
        "facility_type_id": facility_type_id,
        "billing_lag_days": int(billing_lag_days),
        "length_of_stay": int(length_of_stay),
        "encounter_type": encounter_type
    }

    with st.spinner("Evaluating claim parameters through the risk model..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict_user_claim", json=payload, timeout=5)

            # --- Successful Prediction ---
            if response.status_code == 200:
                result = response.json()
                prob = result["denial_probability_pct"]
                risk_level = result["risk_level"]
                action = result["action_recommendation"]

                st.divider()
                st.markdown("### risk Assessment Results")

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob,
                        number={'suffix': "%"},
                        title={'text': "Denial Probability Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1E293B"},
                            'steps': [
                                {'range': [0, 35], 'color': "#10B981"},
                                {'range': [35, 60], 'color': "#F59E0B"},
                                {'range': [60, 100], 'color': "#EF4444"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    st.markdown("#### Assessment Summary")
                    if risk_level == "HIGH":
                        st.error(f"**risk Level: HIGH** — Denial Probability: {prob}%")
                    elif risk_level == "MEDIUM":
                        st.warning(f"**risk Level: MEDIUM** — Denial Probability: {prob}%")
                    else:
                        st.success(f"**risk Level: LOW** — Denial Probability: {prob}%")

                    st.markdown(f"**Recommended Action**\n\n{action}")

                    st.info(
                        f"**CPT / ICD Pair Evaluated:** `{activity_code}` / `{primary_diagnosis_code}`  \n"
                        f"**Payer:** `{payer_id}` — Plan Tier: `{insurance_plan_tier}`"
                    )

            # --- Schema Validation Errors (422) ---
            elif response.status_code == 422:
                error_body = response.json()
                st.divider()
                st.markdown("### Submission Could Not Be Processed")
                st.markdown(
                    f"<p style='color:#7F1D1D; margin-bottom:1rem;'>"
                    f"{error_body.get('message', 'One or more fields failed validation. Please review and correct the entries below.')}"
                    f"</p>",
                    unsafe_allow_html=True
                )

                for err in error_body.get("errors", []):
                    field = err.get("field", "Unknown field")
                    issue = err.get("issue", "Invalid value")
                    provided = err.get("provided_value")

                    provided_text = (
                        f"<div style='margin-top:0.3rem; font-size:0.85rem; color:#6B7280;'>"
                        f"Provided value: <code>{provided}</code></div>"
                        if provided is not None else ""
                    )

                    st.markdown(
                        f"""
                        <div class="validation-error">
                            <div class="error-field">Field: {field}</div>
                            <div class="error-message">{issue}</div>
                            {provided_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # --- Other API Errors ---
            else:
                st.error(f"API returned an unexpected error (HTTP {response.status_code}). Please try again or contact support.")

        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to the backend service. Ensure the API server is running on `http://127.0.0.1:8000`.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The server may be under load. Please retry in a moment.")