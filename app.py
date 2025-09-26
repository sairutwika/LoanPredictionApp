import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")  # dict of LabelEncoders for categorical features

st.title("Loan Prediction App")
st.write("Upload your CSV file to predict loan approval.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Input Data")
    st.dataframe(data)

    # Keep original Loan_Status if exists
    if 'Loan_Status' in data.columns:
        original_status = data['Loan_Status']

    # Features used during training
    feature_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area']

    # Select only the training features
    X_input = data[feature_cols].copy()

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in X_input.columns:
            X_input[col] = le.transform(X_input[col])

    # Make predictions
    predictions = model.predict(X_input)
    data['Approval'] = ["Yes" if p == 1 else "No" for p in predictions]

    # Restore original Loan_Status if it exists
    if 'Loan_Status' in data.columns:
        data['Loan_Status'] = original_status

    st.subheader("Predictions")
    st.dataframe(data)

    # Download output CSV
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="loan_predictions.csv",
        mime="text/csv"
    )
