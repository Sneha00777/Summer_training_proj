import streamlit as st
import pickle
import numpy as np

# 1. Page Header
st.title("Social Network Ads - Purchase Predictor")
st.header("Predict User Purchasing Behavior")
st.markdown("---")

# 2. Load the Single Model Pickle File
@st.cache_resource
def load_model():
    with open('Social_Network_model.pkl', 'rb') as f:
        return pickle.load(f)

st.subheader("Enter User Details:")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=200000, value=50000, step=1000)

st.markdown("---")

# 4. Prediction Logic
if st.button("Predict Purchase Intent", use_container_width=True):
    age_scaled = (age - 37.85) / 10.32
    salary_scaled = (salary - 69806.25) / 34605.02

    input_data = np.array([[age_scaled, salary_scaled]])
    prediction = model.predict(input_data)[0]

    st.write("**Analysis Result:**")
    if prediction == 1:
        st.success("**Likely to Purchase!** User is predicted to buy the product.")
    else:
        st.info("**Unlikely to Purchase.** User is not predicted to buy.")
