import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("kNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()

# ---------- Header ----------
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's clinical details below. The model (K-Nearest Neighbors) "
    "will estimate the likelihood of heart disease based on the patterns it learned "
    "from the training data."
)

st.divider()

# ---------- Input form ----------
with st.form("patient_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=40, max_value=250, value=150)
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "TA (Typical Angina)", "ASY (Asymptomatic)"],
        )
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    # Build a raw input dict matching the one-hot encoding scheme used in training
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain.startswith("ATA") else 0,
        "ChestPainType_NAP": 1 if chest_pain.startswith("NAP") else 0,
        "ChestPainType_TA": 1 if chest_pain.startswith("TA") else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }

    # Ensure column order matches training exactly
    input_df = pd.DataFrame([raw_input])[expected_columns]

    # Scale (the saved scaler was fit on the full one-hot encoded feature set)
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ The model predicts a **high likelihood of heart disease** "
                 f"(confidence: {proba[1] * 100:.1f}%).")
    else:
        st.success(f"✅ The model predicts **no heart disease** "
                   f"(confidence: {proba[0] * 100:.1f}%).")

    with st.expander("See input data used for prediction"):
        st.dataframe(input_df)

    st.caption(
        "⚠️ This tool is for educational/demo purposes only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment."
    )

st.divider()
st.caption("Model: K-Nearest Neighbors classifier trained on the Heart Failure Prediction dataset.")
