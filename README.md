# ❤️ Smart Health Risk Prediction

A machine learning project that predicts health risk using health monitoring and motion sensor data.

## 📌 Project Overview

The Smart Health Risk Prediction system uses machine learning to analyze health and sensor parameters such as heart rate, oxygen level, temperature, motion, gyroscope data, heart rate variability, and fall detection.

A Logistic Regression model is trained to classify the input as:

- ✅ Lower Health Risk
- ⚠️ Higher Health Risk

The project also includes a Streamlit web application for interactive predictions.

## 🚀 Features

- ❤️ Heart rate monitoring
- 🫁 Oxygen level monitoring
- 🌡️ Temperature monitoring
- 📈 Heart rate variability analysis
- 📱 Accelerometer data
- 🔄 Gyroscope data
- 🧍 Fall detection
- 🤖 Machine learning risk prediction
- 🌐 Streamlit web interface

## 🧠 Machine Learning

### Algorithm

**Logistic Regression**

### Input Features

The model uses 13 features:

1. Heart rate
2. Oxygen level
3. Temperature
4. Acceleration X
5. Acceleration Y
6. Acceleration Z
7. Gyroscope X
8. Gyroscope Y
9. Gyroscope Z
10. Acceleration magnitude
11. Gyroscope magnitude
12. Heart rate variability
13. Fall detected

### Target

`health_risk`

- `0` → Lower Health Risk
- `1` → Higher Health Risk

## 📊 Model Performance

The dataset contains **612 records**.

The data was split into:

- Training: 489 samples
- Testing: 123 samples

### Test Accuracy

**100%**

The test set contained:

- 38 lower-risk samples
- 85 higher-risk samples

All 123 test samples were correctly classified in this evaluation.

> ⚠️ Note: The dataset is simulated, so the 100% accuracy should not be interpreted as real-world medical performance.

## 📂 Project Structure

```text
smart-health-risk-prediction
│
├── app
│   └── app.py
│
├── data
│   ├── health_data.csv
│   └── Health Monitoring and Fall detection dataset.csv
│
├── esp32
│
├── model
│   └── health_risk_model.pkl
│
├── notebooks
│   ├── 01_health_risk_model.ipynb
│   ├── 02_new_dataset_exploration.ipynb
│   └── model_evaluation.ipynb
│
├── src
│   ├── predict.py
│   └── train_model.py
│
└── requirements.txt
