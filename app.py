import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Loan Approval Prediction App 💰")
st.write("Upload a CSV file with the same features as the training dataset (without target column)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop target column if present
    if 'Loan_Status' in data.columns:
        data = data.drop('Loan_Status', axis=1)

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in data.columns:
            try:
                data[col] = le.transform(data[col].astype(str))
            except ValueError:
                st.warning(f"Column '{col}' has unseen categories in uploaded data. They will be encoded as -1.")
                data[col] = data[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    # Fill missing values
    data.fillna(data.median(), inplace=True)

    # Predict approval
    predictions = model.predict(data)
    data['Approval'] = ['Yes' if i==1 else 'No' for i in predictions]

    # Display predictions
    st.write("Predictions:")
    st.dataframe(data)

    # Download CSV with predictions
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download CSV with Approval",
        data=csv,
        file_name='loan_predictions.csv',
        mime='text/csv'
    )
