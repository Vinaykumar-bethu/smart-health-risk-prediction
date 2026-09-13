import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/health_risk_model.pkl")

# Page title
st.title("🩺 Smart Health Risk Prediction")

st.write("Enter the health details below to predict the risk level.")

# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=20
)

heart_rate = st.number_input(
    "Heart Rate (BPM)",
    min_value=40,
    max_value=200,
    value=75
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=30.0,
    max_value=45.0,
    value=36.5
)

spo2 = st.number_input(
    "SpO₂ (%)",
    min_value=50,
    max_value=100,
    value=98
)

# Predict button
if st.button("Predict Risk"):

    # Create input with correct feature names
    input_data = pd.DataFrame({
        "age": [age],
        "heart_rate": [heart_rate],
        "temperature": [temperature],
        "spo2": [spo2]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 0:
        st.success("🟢 Risk Level: LOW")
    else:
        st.error("🔴 Risk Level: HIGH")

# Disclaimer
st.info(
    "⚠️ This project is for educational purposes only "
    "and is not a medical diagnosis system."
)