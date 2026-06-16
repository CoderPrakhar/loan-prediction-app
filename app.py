import streamlit as st
import pandas as pd
import joblib

# Load Trained Model
model = joblib.load("loan_model.pkl")

st.title("Loan Approval Prediction System")
st.write("Predict whether the applicant is likely to repay the loan.")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Income", min_value=0, value=50000)
emp_exp = st.number_input("Employment Experience (Years)", min_value=0, value=5)

loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
cred_hist = st.number_input("Credit History Length", min_value=0, value=5)

previous_default = st.selectbox(
    "Previous Loan Default?",
    ["No", "Yes"]
)

# Predict Button
if st.button("Predict"):

    loan_percent_income = loan_amnt / income if income > 0 else 0

    input_data = pd.DataFrame({
        "person_age":[age],
        "person_income":[income],
        "person_emp_exp":[emp_exp],
        "loan_amnt":[loan_amnt],
        "loan_int_rate":[loan_int_rate],
        "loan_percent_income":[loan_percent_income],
        "cb_person_cred_hist_length":[cred_hist],
        "credit_score":[credit_score],

        "person_gender_male":[1],

        "person_education_Bachelor":[1],
        "person_education_Doctorate":[0],
        "person_education_High School":[0],
        "person_education_Master":[0],

        "person_home_ownership_OTHER":[0],
        "person_home_ownership_OWN":[0],
        "person_home_ownership_RENT":[1],

        "loan_intent_EDUCATION":[0],
        "loan_intent_HOMEIMPROVEMENT":[0],
        "loan_intent_MEDICAL":[0],
        "loan_intent_PERSONAL":[1],
        "loan_intent_VENTURE":[0],

        "previous_loan_defaults_on_file_Yes":[
            1 if previous_default == "Yes" else 0
        ]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Likely To Be Repaid")
    else:
        st.error("❌ High Risk Of Default")