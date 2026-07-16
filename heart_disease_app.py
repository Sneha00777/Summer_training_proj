import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("Heart Disease Risk Predictor")
st.header("Decision Tree Classification Model")
st.markdown("---")

# 1. Loading the trained Heart Disease model

with open('heart_disease.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("### Enter Patient Health Metrics:")

# Organizing inputs into columns to fit 15 features neatly
col1, col2, col3 = st.columns(3)

with col1:
    sex_input = st.selectbox("Sex", options=["Female", "Male"])
    # Map sex using LabelEncoder logic (female -> 0, male -> 1)
    sex = 1 if sex_input == "Male" else 0

    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200, step=1)

with col2:
    fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    fbs = 1 if fbs_input == "Yes" else 0

    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150, step=1)

    exang_input = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    exang = 1 if exang_input == "Yes" else 0

    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col3:
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4) Colored by Flourosopy", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversable Defect")

    max_hr_reserve = st.number_input("Max Heart Rate Reserve", min_value=-100, max_value=200, value=10, step=1)
    risk_score = st.number_input("Heart Disease Risk Score", min_value=0.0, max_value=50.0, value=12.0, step=0.01)

st.markdown("---")

if st.button("Analyze Health Data", use_container_width=True):
    # Constructing input array matching the order of df.columns (excluding target)
    input_data = np.array([[
        sex, age, cp, resting_BP, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal, 
        max_hr_reserve, risk_score
    ]])

    # Predict the Target (0 = No Disease, 1 = Disease)
    prediction = model.predict(input_data)
    result = prediction[0]

    st.write("### **Analysis Result:**")

    if result == 1:
        st.error("⚠️ **High Risk Detected:** The model predicts a high likelihood of heart disease.")
    else:
        st.success("**Low Risk:** The model predicts a low likelihood of heart disease.")
