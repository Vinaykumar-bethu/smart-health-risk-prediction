import os
import pandas as pd
import joblib

# Find the project folder
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
model_path = os.path.join(
    project_folder,
    "model",
    "health_risk_model.pkl"
)

model = joblib.load(model_path)

# Get user input
age = float(input("Enter age: "))
heart_rate = float(input("Enter heart rate: "))
temperature = float(input("Enter temperature: "))
spo2 = float(input("Enter SpO2: "))

# Create input data
new_data = pd.DataFrame(
    [[age, heart_rate, temperature, spo2]],
    columns=["age", "heart_rate", "temperature", "spo2"]
)

# Make prediction
prediction = model.predict(new_data)

print("\nPredicted Risk:", prediction[0])

if prediction[0] == 1:
    print("Higher risk according to the trained model.")
else:
    print("Lower risk according to the trained model.")