import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction App", page_icon="💉", layout="centered")

st.title("💉 Diabetes Prediction App")
st.markdown("Enter the patient details below to predict whether they have diabetes or not.")

# -----------------------------
# Input fields
# -----------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=100, value=33)

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("Predict Diabetes"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"🔴 The model predicts **Diabetes** (Probability: {prob:.2f})" if prob else "🔴 The model predicts **Diabetes**")
    else:
        st.success(f"🟢 The model predicts **No Diabetes** (Probability: {prob:.2f})" if prob else "🟢 The model predicts **No Diabetes**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Durkadevi S | Diabetes Prediction using XGBoost (Tuned Model)")
