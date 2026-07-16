import streamlit as st
import pickle
import numpy as np

# 1. App Titles and Headers
st.title("Online Sales Revenue Predictor")
st.header("Predict Transaction Value Using Sales Specs")
st.markdown("---")

# 2. Loading your trained Random Forest model
with open('Online_Sales_Data_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("### Enter Sales Metrics:")

# 3. Inputs for Units Sold and Unit Price
units_sold = st.number_input("Units Sold", min_value=1, max_value=100, value=2, step=1)
unit_price = st.number_input("Unit Price ($)", min_value=0.1, max_value=10000.0, value=99.99, step=0.01)

st.markdown("---")

# 4. Prediction Logic
if st.button("Predict Total Revenue", use_container_width=True):
    # Format the input features for the model
    features = np.array([[units_sold, unit_price]])
    prediction = model.predict(features)

    # Display the final forecasted amount
    st.success(f"Estimated Total Revenue: ${prediction[0]:,.2f}")
