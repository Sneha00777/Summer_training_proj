import streamlit as st
import pickle
import numpy as np

# 1. App Header Layout
st.title("Student Success Assistant")
st.header("Academic Performance Prediction Model")
st.markdown("---")

# 2. Loading your student performance model
with open('Student_Performance_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. Getting User Input
st.write("### Enter Student Metrics:")
overall_score = st.number_input("Overall Score (0 - 100)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
st.markdown("---")

# 4. Prediction Logic
if st.button("Result", use_container_width=True):
    # Form formatting for prediction (1 feature)
    input_data = np.array([[overall_score]])
    prediction = model.predict(input_data)

    st.write("**Analysis Result:**")
    # Logistic Regression returns 0 or 1 
    if prediction[0] == 1:
        st.balloons()
        st.success("🎉 **Predicted Status: Pass / Good Performance** (Grade D or higher)")
    else:
        st.error("**Predicted Status: failed / Poor Performance** (Grade E or F)")
