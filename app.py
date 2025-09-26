import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Loan Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.dataframe(data)

    # Keep original Loan_Status if exists
    if 'Loan_Status' in data.columns:
        original_status = data['Loan_Status']

    # Encode categorical columns
    for col, le in encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])

    # Make predictions
    predictions = model.predict(data)

    # Add predictions as a new column
    data['Approval'] = ["Yes" if p == 1 else "No" for p in predictions]

    # Restore original Loan_Status if it exists
    if 'Loan_Status' in data.columns:
        data['Loan_Status'] = original_status

    st.write("Predictions:")
    st.dataframe(data)

    # Download the output CSV
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="loan_predictions.csv",
        mime="text/csv"
    )
