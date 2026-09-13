import joblib

# Load trained model
model = joblib.load("model/health_risk_model.pkl")

# Get user input
age = float(input("Enter age: "))
heart_rate = float(input("Enter heart rate: "))
temperature = float(input("Enter temperature: "))
spo2 = float(input("Enter SpO2: "))

# Make prediction
prediction = model.predict([
    [age, heart_rate, temperature, spo2]
])

# Display result
if prediction[0] == 0:
    print("Risk Level: LOW")
else:
    print("Risk Level: HIGH")