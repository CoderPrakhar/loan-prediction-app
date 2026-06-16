import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Prediction System")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

income = st.number_input(
    "Income",
    min_value=0,
    value=100000
)

emp_exp = st.number_input(
    "Employment Experience",
    min_value=0,
    value=10
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0,
    value=5000
)

loan_int_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=5.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=850
)

cred_hist = st.number_input(
    "Credit History Length",
    min_value=0,
    value=10
)

previous_default = st.selectbox(
    "Previous Loan Default?",
    ["No", "Yes"]
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_emp_exp": [emp_exp],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_amnt / income if income > 0 else 0],
        "cb_person_cred_hist_length": [cred_hist],
        "credit_score": [credit_score],
        "previous_loan_defaults_on_file_Yes": [
            1 if previous_default == "Yes" else 0
        ]
    })

    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model.feature_names_in_]

    st.subheader("Debug Information")
    st.write(input_data)

    prediction = model.predict(input_data)

    st.write("Raw Prediction =", prediction[0])

    try:
        probability = model.predict_proba(input_data)
        st.write("Prediction Probability =", probability)
    except:
        pass

    if prediction[0] == 1:
        st.error("❌ High Risk Of Default")
    else:
        st.success("✅ Loan Likely To Be Repaid")