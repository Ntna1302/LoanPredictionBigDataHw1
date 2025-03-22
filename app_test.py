import streamlit as st
import numpy as np
import pickle
import time

# Load models and transformations
with open('ann_model.pkl', 'rb') as ann_file:
    ann_model = pickle.load(ann_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('pca.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

with open("model_accuracies.pkl", "rb") as acc_file:
    model_accuracies = pickle.load(acc_file)

# Get ANN accuracy
ann_accuracy = model_accuracies.get("ANN", "N/A")

# Streamlit UI
st.set_page_config(page_title="Loan Prediction", page_icon="💰", layout="wide")
st.title("🏦 Loan Prediction App")
st.write("### Get an instant loan approval prediction using Machine Learning model!")

# Define columns for layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Enter Your Details")
    no_of_dependents = st.slider(
        'Number of Dependents', 
        0, 10, 0, 
        help="Select the number of dependents (children, spouse, or other family members) financially dependent on you."
    )

    education = st.radio(
        "Education Level", 
        ["Graduate", "Not Graduate"], 
        help="Select 'Graduate' if you have completed a degree, otherwise select 'Not Graduate'."
    )

    self_employed = st.radio(
        "Self Employed", 
        ["Yes", "No"], 
        help="Select 'Yes' if you own a business or work independently; otherwise, select 'No'."
    )

    income_annum = st.number_input(
        "Annual Income ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter your total yearly income before taxes in US dollars."
    )

    loan_amount = st.number_input(
        "Loan Amount ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter the amount of loan you are applying for in US dollars."
    )

    loan_term = st.slider(
        "Loan Term (months)", 
        1, 360, 60, 
        help="Select the duration of the loan repayment in months (e.g., 60 months = 5 years)."
    )

    cibil_score = st.slider(
        'CIBIL Score (300 - 900)', 
        300, 900, 700, 
        help="Enter your CIBIL score, which ranges from 300 to 900 and indicates your creditworthiness (higher is better)."
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter the total value of your residential properties in US dollars."
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter the total value of your commercial properties (shops, offices, etc.) in US dollars."
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter the total value of your luxury assets (cars, jewelry, art, etc.) in US dollars."
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value ($)", 
        min_value=0.0, step=1000.0, format="%.2f", 
        help="Enter the total value of your bank balance and fixed deposits in US dollars."
    )

    # Convert categorical values using One-Hot Encoding (match training)
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0

# Function to transform input features using PCA
def transform_input():
    # Ensure features match training set
    input_features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, 
                                 cibil_score, residential_assets_value, commercial_assets_value, 
                                 luxury_assets_value, bank_asset_value, education_encoded, self_employed_encoded]])

    # Apply scaling
    scaled_features = scaler.transform(input_features)

    # Apply PCA transformation
    pca_features = pca.transform(scaled_features)

    return pca_features

# Function to predict using ANN model
def predict_ann():
    pca_features = transform_input()
    prediction = ann_model.predict(pca_features)
    return "✅ Approved" if prediction[0] >= 0.5 else "❌ Rejected"

# Check if all inputs have default values
if (no_of_dependents == 0 and income_annum == 0.0 and loan_amount == 0.0 and loan_term == 60 and 
    cibil_score == 700 and residential_assets_value == 0.0 and commercial_assets_value == 0.0 and 
    luxury_assets_value == 0.0 and bank_asset_value == 0.0):
    default_inputs = True
else:
    default_inputs = False

# Prediction section
with col2:
    st.subheader("📈 Prediction Results")

    if default_inputs:
        st.warning("⚠ You need to input values to start the prediction.")
    else:
        result_text = predict_ann()
        st.subheader(f"**Loan Status: {result_text}**")
        st.write(f"🔍 **Model Accuracy (ANN): {ann_accuracy}**")
        if "✅" in result_text:
            st.success("🎉 Your loan is likely to be approved using **ANN**.")
        else:
            st.error("⚠ Your loan may be rejected using **ANN** .")

# Footer
st.markdown("---")
st.markdown("💡 *This app uses Machine Learning model to provide loan approval predictions. Final decisions depend on financial institutions.*")
