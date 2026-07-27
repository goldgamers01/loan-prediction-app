import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Status Prediction")
st.write("Enter the applicant details below.")

# ----------------------------
# Load Files
# ----------------------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "loan.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

@st.cache_resource
def load_files():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

try:
    model, label_encoders = load_files()
except Exception as e:
    st.error(f"Unable to load model files.\n\n{e}")
    st.stop()

# ----------------------------
# Inputs
# ----------------------------
age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    label_encoders["Gender"].classes_
)

qualification = st.selectbox(
    "Qualification",
    label_encoders["Qualification"].classes_
)

annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

credit_score = st.slider(
    "Credit Score",
    300,
    900,
    650
)

dependents = st.number_input(
    "Dependents",
    0,
    10,
    0
)

property_value = st.number_input(
    "Property Value",
    min_value=0.0,
    value=100000.0
)

gold_grams = st.number_input(
    "Gold Available (grams)",
    min_value=0.0,
    value=50.0
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Loan Status"):

    gender_encoded = label_encoders["Gender"].transform([gender])[0]
    qualification_encoded = label_encoders["Qualification"].transform([qualification])[0]

    features = np.array([[
        age,
        gender_encoded,
        qualification_encoded,
        annual_income,
        loan_amount,
        credit_score,
        dependents,
        property_value,
        gold_grams
    ]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("🎉 Loan Approved")
    else:
        st.error("❌ Loan Rejected")