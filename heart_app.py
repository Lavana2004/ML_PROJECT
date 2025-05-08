import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="❤️ Heart Disease Prediction App")

# Function to load data and train model — cached to avoid retraining every time
@st.cache_resource
def load_model():
    data = pd.read_csv('heart_disease_prediction.csv')
    features = data.drop(columns='target')
    target = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=23
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Load model and accuracy
model, accuracy = load_model()

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.markdown(f"### Model Accuracy: {accuracy*100:.2f}%")

# Input fields
age = st.number_input("Age (29-77)", min_value=29, max_value=77, value=50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (90-200)", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol (120-564)", min_value=120, max_value=564, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate (70-200)", min_value=70, max_value=200, value=150)
exang = st.radio("Exercise-induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("Oldpeak (0.0-6.2)", min_value=0.0, max_value=6.2, value=1.0)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thal (0-2)", min_value=0, max_value=2, value=1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ You are healthy!")
    else:
        st.error("❗ You may have a serious heart condition. Please consult a doctor.")

# Hide default Streamlit footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
