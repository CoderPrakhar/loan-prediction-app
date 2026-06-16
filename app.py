if st.button("Predict"):

    input_data = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_emp_exp": [emp_exp],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_amnt / income],
        "cb_person_cred_hist_length": [cred_hist],
        "credit_score": [credit_score],
        "previous_loan_defaults_on_file_Yes": [1 if previous_default == "Yes" else 0]
    })

    # Missing columns add karo
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # Same order as training
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