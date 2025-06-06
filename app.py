
import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('/content/scaler.save')
model= joblib.load('/content/model.save')
st.title("Credit Card Default Prediction")
st.markdown("""
This app predicts the likelihood of a credit card customer defaulting on their payments based on financial history and profile information.
""")
sex_options = {
    "Male (1)": 1,
    "Female (2)": 2
}

education_options = {
    "Graduate School (1)": 1,
    "University (2)": 2,
    "High School (3)": 3,
    "Others (4)": 4
}

marriage_options = {
    "Married (1)": 1,
    "Single (2)": 2,
    "Others (3)": 3
}

pay_status_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # meaning from dataset
LIMIT_BAL = st.number_input("Credit Limit", min_value=0, value=20000)
SEX = sex_options[st.selectbox("SEX", options=list(sex_options.keys()))]
EDUCATION = education_options[st.selectbox("Education Level", options=list(education_options.keys()))]
MARRIAGE = marriage_options[st.selectbox("Marital Status", options=list(marriage_options.keys()))]
AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

# Repayment status inputs
PAY_0 = st.selectbox("Repayment Status in September ", options=pay_status_options)
PAY_2 = st.selectbox("Repayment Status in August ", options=pay_status_options)
PAY_3 = st.selectbox("Repayment Status in July ", options=pay_status_options)
PAY_4 = st.selectbox("Repayment Status in June", options=pay_status_options)
PAY_5 = st.selectbox("Repayment Status in May", options=pay_status_options)
PAY_6 = st.selectbox("Repayment Status in April", options=pay_status_options)

# Bill Amounts
BILL_AMT1 = st.number_input("Bill Amount in September ", value=0)
BILL_AMT2 = st.number_input("Bill Amount in August", value=0)
BILL_AMT3 = st.number_input("Bill Amount in July ", value=0)
BILL_AMT4 = st.number_input("Bill Amount in June", value=0)
BILL_AMT5 = st.number_input("Bill Amount in May ", value=0)
BILL_AMT6 = st.number_input("Bill Amount in April ", value=0)

# Payment Amounts
PAY_AMT1 = st.number_input("Payment in September ", value=0)
PAY_AMT2 = st.number_input("Payment in August ", value=0)
PAY_AMT3 = st.number_input("Payment in July ", value=0)
PAY_AMT4 = st.number_input("Payment in June ", value=0)
PAY_AMT5 = st.number_input("Payment in May ", value=0)
PAY_AMT6 = st.number_input("Payment in April ", value=0)

if st.button(" Predict Default Risk"):
    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                            BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
                            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]],dtype=np.float64)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    

    if prediction == 1:
        st.error("Prediction: High risk of default.")
    else:
        st.success("Prediction: Low risk of default.")
