import streamlit as st
import pickle
import numpy as np

st.title("Health Metrics App")
st.header("BMI Prediction Model")
st.markdown("---")

# 1. Loading the trained BMI model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("### Enter Patient Health Metrics:")

# Creating inputs for the features the model was trained on (Weight and Height)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)

st.markdown("---")

if st.button("Analyze Health Data", use_container_width=True):
    # The model expects an array matching the training columns: [['Weight', 'Height']]
    input_data = np.array([[weight, height]])

    # Predict the BMI
    prediction = model.predict(input_data)
    bmi_result = prediction[0]

    st.write("### **Analysis Result:**")
    st.info(f"**Predicted BMI:** {bmi_result:.1f}")

    # Providing a quick interpretation based on standard BMI classifications
    if bmi_result < 18.5:
        st.warning("Category: Underweight")
    elif 18.5 <= bmi_result < 25.0:
        st.success("Category: Normal weight")
    elif 25.0 <= bmi_result < 30.0:
        st.warning("Category: Overweight")
    else:
        st.error("Category: Obese")
