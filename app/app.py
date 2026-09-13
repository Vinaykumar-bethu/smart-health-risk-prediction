import os
import pandas as pd
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Health Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Find project folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model path
model_path = os.path.join(
    project_folder,
    "model",
    "health_risk_model.pkl"
)

# Load model
model = joblib.load(model_path)

# Title
st.title("❤️ Smart Health Risk Prediction")

st.write(
    "Enter the health parameters below to get a risk prediction "
    "from the trained machine-learning model."
)

st.divider()

# Input section
st.subheader("📋 Health Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=21
    )

    heart_rate = st.number_input(
        "Heart Rate (BPM)",
        min_value=30,
        max_value=200,
        value=80
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=36.5,
        step=0.1
    )

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50,
        max_value=100,
        value=98
    )

st.divider()

# Prediction
if st.button("🔍 Predict Health Risk", use_container_width=True):

    new_data = pd.DataFrame(
        [[age, heart_rate, temperature, spo2]],
        columns=["age", "heart_rate", "temperature", "spo2"]
    )

    prediction = model.predict(new_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Higher Risk")
        st.write(
            "The trained model predicts a higher risk based on "
            "the entered values."
        )
    else:
        st.success("✅ Lower Risk")
        st.write(
            "The trained model predicts a lower risk based on "
            "the entered values."
        )

st.divider()

# Model information
st.subheader("🤖 About the Model")

st.write(
    "This project uses Logistic Regression with four input "
    "features: age, heart rate, temperature, and SpO₂."
)

# Disclaimer
st.warning(
    "⚠️ Educational Project: This application is a machine-learning "
    "prototype and is NOT a medical diagnostic tool. Do not use "
    "its predictions for medical decisions."
)

st.divider()

st.subheader("📊 Health Metrics")

chart_data = pd.DataFrame({
    "Metric": ["Heart Rate", "Temperature", "SpO₂"],
    "Value": [heart_rate, temperature, spo2]
})

st.bar_chart(
    chart_data.set_index("Metric")
)
