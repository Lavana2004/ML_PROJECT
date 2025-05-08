import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("💖 Heart Disease Prediction App")

# User Input
st.subheader("Enter the Patient Details:")

age = st.number_input("Age", 29, 77, 50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure (90-200)", 90, 200, 120)
chol = st.number_input("Cholesterol (120-564)", 120, 564, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.number_input("Resting ECG Results (0-2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate (70-200)", 70, 200, 150)
exang = st.radio("Exercise-induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("Oldpeak (0.0-6.2)", 0.0, 6.2, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.number_input("Thal (0-2)", 0, 2, 1)

# Predict Button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ You are healthy!")
    else:
        st.error("❗ You may have a serious heart condition. Please consult a doctor.")
