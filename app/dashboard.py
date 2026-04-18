import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import os

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Check API health
try:
    health = requests.get(f"{API_URL}/health").json()
    model_info = requests.get(f"{API_URL}/model-info").json()
    st.sidebar.success(f"API Status: Healthy")
    st.sidebar.info(f"Model: {model_info['model_name']}")
    st.sidebar.metric("ROC-AUC", model_info['metrics']['roc_auc'])
    st.sidebar.metric("Accuracy", model_info['metrics']['accuracy'])
    st.sidebar.metric("F1 Score", model_info['metrics']['f1_score'])
except:
    st.error("API is not running. Start it with: uvicorn api.main:app --reload")
    st.stop()

st.header("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 24)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 65.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, monthly * tenure)

# Auto-calculate engineered features
services = ["Yes", "Yes", internet, security, backup, protection, support, tv, movies]
total_services = sum(1 for s in services if s == "Yes")

if tenure <= 12:
    tenure_group = "0-1yr"
elif tenure <= 24:
    tenure_group = "1-2yr"
elif tenure <= 48:
    tenure_group = "2-4yr"
elif tenure <= 60:
    tenure_group = "4-5yr"
else:
    tenure_group = "5-6yr"

charges_per_tenure = total / (tenure + 1)

if st.button("Predict Churn", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "tenure_group": tenure_group,
        "charges_per_tenure": charges_per_tenure,
        "total_services": total_services
    }

    response = requests.post(f"{API_URL}/predict", json=payload).json()

    st.divider()
    st.header("Prediction Result")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if response['churn_prediction'] == 1:
            st.error("⚠️ Customer will CHURN")
        else:
            st.success("✅ Customer will STAY")

    with col_b:
        st.metric("Churn Probability", f"{response['churn_probability']:.1%}")

    with col_c:
        risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
        st.metric("Risk Level", response['risk_level'])

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=response['churn_probability'] * 100,
        title={'text': "Churn Probability %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    img_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'evaluation.png')
    st.image(img_path, caption='Model Evaluation Charts')