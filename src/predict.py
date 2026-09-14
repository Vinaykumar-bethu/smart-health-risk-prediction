import os
import pandas as pd
import joblib

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

print("❤️ Smart Health Risk Prediction")
print("--------------------------------")

# Get health data
heart_rate = float(input("Enter heart rate: "))
oxygen_level = float(input("Enter oxygen level: "))
temperature = float(input("Enter temperature: "))

acc_x = float(input("Enter Acc X: "))
acc_y = float(input("Enter Acc Y: "))
acc_z = float(input("Enter Acc Z: "))

gyro_x = float(input("Enter Gyro X: "))
gyro_y = float(input("Enter Gyro Y: "))
gyro_z = float(input("Enter Gyro Z: "))

acceleration_magnitude = float(
    input("Enter acceleration magnitude: ")
)

gyro_magnitude = float(
    input("Enter gyro magnitude: ")
)

heart_rate_variability = float(
    input("Enter heart rate variability: ")
)

fall_detected = int(
    input("Fall detected? (0 = No, 1 = Yes): ")
)

# Create input data
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

# Make prediction
prediction = model.predict(new_data)

print("\nPrediction Result")
print("-----------------")

if prediction[0] == 1:
    print("⚠️ Higher Health Risk")
else:
    print("✅ Lower Health Risk")

print("\nPrediction completed successfully!")