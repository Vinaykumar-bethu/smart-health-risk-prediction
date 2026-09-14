import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Smart Health Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Project folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Load trained model
model_path = os.path.join(
    project_folder,
    "model",
    "health_risk_model.pkl"
)

model = joblib.load(model_path)

# Title
st.title("❤️ Smart Health Risk Prediction")

st.write(
    "Enter the health sensor values below to predict "
    "health risk using a machine-learning model."
)

st.divider()

st.subheader("📋 Health Sensor Information")

col1, col2 = st.columns(2)

with col1:

    heart_rate = st.number_input(
        "Heart Rate (BPM)",
        min_value=30.0,
        max_value=200.0,
        value=80.0
    )

    oxygen_level = st.number_input(
        "Oxygen Level (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=36.5
    )

    acc_x = st.number_input(
        "Acceleration X",
        value=0.1
    )

    acc_y = st.number_input(
        "Acceleration Y",
        value=0.2
    )

    acc_z = st.number_input(
        "Acceleration Z",
        value=9.8
    )

    fall_detected = st.selectbox(
        "Fall Detected?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col2:

    gyro_x = st.number_input(
        "Gyroscope X",
        value=0.1
    )

    gyro_y = st.number_input(
        "Gyroscope Y",
        value=0.1
    )

    gyro_z = st.number_input(
        "Gyroscope Z",
        value=0.1
    )

    acceleration_magnitude = st.number_input(
        "Acceleration Magnitude",
        value=9.8
    )

    gyro_magnitude = st.number_input(
        "Gyroscope Magnitude",
        value=0.17
    )

    heart_rate_variability = st.number_input(
        "Heart Rate Variability",
        value=40.0
    )


st.divider()

# Prediction button
if st.button(
    "🔍 Predict Health Risk",
    use_container_width=True
):

    new_data = pd.DataFrame(
        [[
            heart_rate,
            oxygen_level,
            temperature,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            acceleration_magnitude,
            gyro_magnitude,
            heart_rate_variability,
            fall_detected
        ]],
        columns=[
            "heart_rate",
            "oxygen_level",
            "temperature",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "acceleration_magnitude",
            "gyro_magnitude",
            "heart_rate_variability",
            "fall_detected"
        ]
    )

    prediction = model.predict(new_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Higher Health Risk")

        st.write(
            "The trained model predicts a higher health risk "
            "based on the entered sensor values."
        )

    else:

        st.success("✅ Lower Health Risk")

        st.write(
            "The trained model predicts a lower health risk "
            "based on the entered sensor values."
        )


st.divider()

st.subheader("🤖 About the Model")

st.write(
    "This project uses Logistic Regression with 13 health "
    "and sensor features including heart rate, oxygen level, "
    "temperature, motion data, gyroscope data, heart rate "
    "variability, and fall detection."
)

st.warning(
    "⚠️ Educational Project: This application is a machine-learning "
    "prototype and is NOT a medical diagnostic tool. Do not use "
    "its predictions for medical decisions."
)
