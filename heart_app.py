import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
data = pd.read_csv('heart_disease_prediction.csv')

# Prepare features and target
features = data.drop(columns='target')
target = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=23
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Streamlit App
st.set_page_config(page_title="❤️ Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown(f"### Accuracy: {accuracy * 100:.2f}%")
st.write("Fill out the details below to check your heart health status.")

# User Inputs
age = st.number_input("Age (29-77)", min_value=29, max_value=77, value=50)
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (90-200)", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol (120-564)", min_value=120, max_value=564, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate (70-200)", min_value=70, max_value=200, value=150)
exang = st.radio("Exercise-induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("Oldpeak (0.0-6.2)", min_value=0.0, max_value=6.2, value=1.0)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thal (0-2)", min_value=0, max_value=2, value=1)

# Prediction button
if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_input)

    if prediction[0] == 1:
        st.success("✅ You are healthy!")
    else:
        st.error("❗ You may have a serious heart condition. Please consult a doctor.")

# Hide footer and style page
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
