
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_disease_decision_tree.pkl")

# Model Accuracy (you can update it if improved)
model_accuracy = 0.88  # 88%

# Streamlit Page Setup
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: crimson;'>Heart Disease Prediction App</h1>", unsafe_allow_html=True)

# Model Info Box
st.info(f"📊 **Model Accuracy:** {model_accuracy * 100:.2f}% (Decision Tree Classifier)")

# Sidebar
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This app uses a **Decision Tree Machine Learning model** to predict the likelihood of heart disease based on clinical parameters like:
- Age, Chest Pain, Blood Pressure, Cholesterol, etc.

💡 Built by Nikhil Donthusaram
""")

# Form Layout - 2 Columns
st.markdown("### 🧾 Enter Patient Details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200)
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prepare input for prediction
input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict Button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("💔 **Result: The patient is likely to have heart disease. Please consult a doctor.**")
    else:
        st.success("✅ **Result: The patient is NOT likely to have heart disease.**")



