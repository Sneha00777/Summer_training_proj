import streamlit as st
import pickle
import numpy as np

st.title("💰 Loan Approval Prediction App")
st.header("Random Forest Classifier Model")
st.markdown("---")

with open('loan_approval_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("### Enter Applicant Details:")


col1, col2 = st.columns(2)

with col1:
    loan_id = st.number_input("Loan ID", min_value=1, value=1, step=1)

    # Categorical Input: Education (Graduate = 0, Not Graduate = 1)
    education_text = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    education = 0 if education_text == "Graduate" else 1

    income_annum = st.number_input("Annual Income ($)", min_value=0, value=5000000, step=50000)
    loan_term = st.number_input("Loan Term (Months)", min_value=2, max_value=360, value=12, step=2)

    residential_assets_value = st.number_input("Residential Assets Value ($)", min_value=0, value=2400000, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value ($)", min_value=0, value=22700000, step=10000)

with col2:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
    self_employed_text = st.selectbox("Self Employed?", ["No", "Yes"])
    self_employed = 1 if self_employed_text == "Yes" else 0

    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=29900000, step=50000)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=778, step=1)

    commercial_assets_value = st.number_input("Commercial Assets Value ($)", min_value=0, value=17600000, step=10000)
    bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, value=8000000, step=10000)

st.markdown("---")

if st.button("Predict Loan Approval Status", use_container_width=True):
    input_data = np.array([[
        loan_id, 
        no_of_dependents, 
        education, 
        self_employed, 
        income_annum, 
        loan_amount, 
        loan_term, 
        cibil_score, 
        residential_assets_value, 
        commercial_assets_value, 
        luxury_assets_value, 
        bank_asset_value
    ]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write("### **Prediction Result:**")
    if prediction[0] == 0:
        st.success(f"**Loan Approved!** (Confidence: {prediction_proba[0][0]*100:.1f}%)")
    else:
        st.error(f"**Loan Rejected** (Confidence: {prediction_proba[0][1]*100:.1f}%)")
