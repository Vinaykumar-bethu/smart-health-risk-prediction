import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Project folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset path
data_path = os.path.join(
    project_folder,
    "data",
    "Health Monitoring and Fall detection dataset.csv"
)

# Load dataset
data = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Rows:", len(data))
print("Columns:", len(data.columns))


# Features used by the model
features = [
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

X = data[features]
y = data["health_risk"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create and train model
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

print("Model trained successfully!")


# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Test Accuracy:", accuracy)
print("Test Accuracy Percentage:", accuracy * 100, "%")


# Save model
model_path = os.path.join(
    project_folder,
    "model",
    "health_risk_model.pkl"
)

joblib.dump(model, model_path)

print("Model saved successfully!")
print("Model path:", model_path)