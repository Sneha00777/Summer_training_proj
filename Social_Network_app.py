%%writefile Social_Network_app.py
import streamlit as st
import pickle
import numpy as np

# 1. Page Header
st.title("Social Network Ads - Purchase Predictor")
st.header("Predict User Purchasing Behavior")
st.markdown("---")

# 2. Function to load the model
@st.cache_resource
def load_model():
    with open('Social_Network_model.pkl', 'rb') as f:
        return pickle.load(f)

# 3. Call the function to load the model into 'model' variable
try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'Social_Network_model.pkl' not found.")
    st.stop()

# 4. User Inputs (Age & Salary)
st.subheader("Enter User Details:")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=200000, value=50000, step=1000)

st.markdown("---")

# 5. Prediction Logic
if st.button("Predict Purchase Intent", use_container_width=True):
    # Scale inputs manually using training set mean & std
    age_scaled = (age - 37.85) / 10.32
    salary_scaled = (salary - 69806.25) / 34605.02
    
    input_data = np.array([[age_scaled, salary_scaled]])
    
    prediction = model.predict(input_data)[0]
    
    st.write("**Analysis Result:**")
    if prediction == 1:
        st.success("🎯 **Likely to Purchase!** User is predicted to buy the product.")
    else:
        st.info("ℹ️ **Unlikely to Purchase.** User is not predicted to buy.")
