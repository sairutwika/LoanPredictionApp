import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("loan_approval.csv")

# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # save encoder for this column

# Fill missing values
df.fillna(df.median(), inplace=True)

# Features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "loan_model.pkl")

# Save encoders for future use
joblib.dump(encoders, "encoders.pkl")

print("Model and encoders saved successfully!")
