import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# App title
st.set_page_config(page_title="Loan Acceptance Predictor", layout="centered")
st.title("🏦 Loan Acceptance Predictor")
st.markdown("Predict whether a customer will accept personal loan.")

# Input form
with st.form("loan_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
    income = st.number_input("Annual Income (in $1000)", min_value=0, value=50)
    family = st.selectbox("Family Size", [1, 2, 3, 4])
    education = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Advanced/Professional"])
    ccavg = st.number_input("Average Credit Card Spend (in $1000)", min_value=0.0, value=1.0)
    mortgage = st.number_input("Mortgage Amount (in $1000)", min_value=0, value=0)
    securities_account = st.selectbox("Has Securities Account?", ["No", "Yes"])
    cd_account = st.selectbox("Has CD Account?", ["No", "Yes"])
    online = st.selectbox("Uses Online Banking?", ["No", "Yes"])
    creditcard = st.selectbox("Has Credit Card with Bank?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Map categorical values
    education_map = {"Undergraduate": 1, "Graduate": 2, "Advanced/Professional": 3}
    binary_map = {"No": 0, "Yes": 1}

    input_data = pd.DataFrame([[
        age,
        experience,
        income,
        family,
        ccavg,
        education_map[education],
        mortgage,
        binary_map[securities_account],
        binary_map[cd_account],
        binary_map[online],
        binary_map[creditcard]
    ]], columns=[
        "Age", "Experience", "Income", "Family", "CCAvg", "Education",
        "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"❌ The customer is likely to **accept** on the loan.")
    else:
        st.success(f"✅ The customer is **not likely to aceept** on the loan.")

    st.markdown(f"**Probability of Default:** `{probability:.2%}`")
