import streamlit as st
import pickle
import numpy as np
st.title("Medical Health Assistant")
st.header("Diabetes Risk Prediction Model")
st.markdown("---")

# 2. Loading the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("Enter Patient Health Metric:")
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=500, value=100)
st.markdown("---")

if st.button("Analyze Health Data", use_container_width=True):
    input_data = np.array([[glucose]])
    prediction = model.predict(input_data)
    st.write("**Analysis Result:**")
    if prediction[0] >= 0.5:
        st.error(f"High Risk Detected (Score: {prediction[0]:.2f}): The model predicts a high probability of diabetes.")
    else:
        st.success(f"Low Risk Detected (Score: {prediction[0]:.2f}): The model predicts a low probability of diabetes.")
