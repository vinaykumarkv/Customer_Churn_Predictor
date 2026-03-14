import streamlit as st
import requests
import json

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/predict"   # change to your deployed URL later

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Real-Time Customer Churn Prediction")
st.markdown("Enter customer details below to get an instant churn risk assessment using a tuned LightGBM model.")

# ────────────────────────────────────────────────
# Form – all fields matching the Pydantic model
# ────────────────────────────────────────────────
with st.form("customer_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=70.0, step=0.1)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)

    submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# Handle submission
# ────────────────────────────────────────────────
if submitted:
    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(API_URL, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()

            # Display result
            st.success("Prediction complete!")

            col_left, col_right = st.columns([2, 3])

            with col_left:
                prob = result["churn_probability"]
                st.metric("Churn Probability", f"{prob:.1%}", delta=None)

                # Simple gauge-like bar
                st.progress(prob)
                st.caption(f"{prob:.1%} chance of churn")

            with col_right:
                risk = result["risk_level"]
                if risk == "High":
                    st.error(f"**{risk} Risk** – {result['message']}")
                elif risk == "Medium":
                    st.warning(f"**{risk} Risk** – {result['message']}")
                else:
                    st.success(f"**{risk} Risk** – {result['message']}")

                st.info(f"Model: {result['model_version']}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
        except json.JSONDecodeError:
            st.error("Invalid response from API.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by LightGBM | Trained on Telco Customer Churn dataset")