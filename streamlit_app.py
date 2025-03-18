import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction System")
st.write("Enter the applicant's details below to predict the loan default risk.")

age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Monthly Income ($)", min_value=500, max_value=100000, step=500)
loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
months_employed = st.number_input("Months Employed", min_value=0, max_value=600, step=1)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, step=1)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, step=0.1)
loan_term = st.number_input("Loan Term (Months)", min_value=6, max_value=120, step=6)
dti_ratio = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=1.0, step=0.01)

input_data = np.array([
    age, income, loan_amount, credit_score, months_employed, 
    num_credit_lines, interest_rate, loan_term, dti_ratio
]).reshape(1, -1)

input_data = scaler.transform(input_data)

if st.button("Predict Loan Default"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of default

    if prediction[0] == 1:
        st.error(f"🚨 High Risk: This applicant is **likely to default** (Risk: {probability:.2%}).")
    else:
        st.success(f"✅ Low Risk: This applicant is **not likely to default** (Risk: {probability:.2%}).")