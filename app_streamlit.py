import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
subscription_length = st.number_input("Subscription Length (Months)", min_value=1, value=12)
monthly_bill = st.number_input("Monthly Bill ($)", min_value=10, value=50)

if st.button("Predict"):
    # Convert input to DataFrame
    data = pd.DataFrame([[gender, age, subscription_length, monthly_bill]], columns=["Gender", "Age", "Subscription_Length", "Monthly_Bill"])
    
    # One-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Ensure correct columns
    expected_cols = ['Age', 'Subscription_Length', 'Monthly_Bill', 'Gender_Male']
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    prediction = model.predict(data)
    st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
