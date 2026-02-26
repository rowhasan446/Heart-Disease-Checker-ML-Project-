# heart_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
import joblib
from sklearn.ensemble import RandomForestClassifier

# ───────────────────────────────────────────────
#  1. Page config & look
# ───────────────────────────────────────────────

st.set_page_config(
    page_title="Heart Risk Checker",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.4);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 2rem;
    }
    
    .result-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    h1 {
        color: #1e293b;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #ff416c;
    }
</style>
""", unsafe_allow_html=True)

# Lottie Animation Helper
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_heart = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_96mscz64.json")

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title("❤️ Heart Risk Checker")
    st.markdown("""
    <div style="background: rgba(255, 75, 43, 0.1); border-left: 5px solid #ff4b2b; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <strong>Precision Diagnostic Tool</strong><br>
        Analyze your cardiovascular health factors using advanced machine learning. 
    </div>
    """, unsafe_allow_html=True)
with col2:
    if lottie_heart:
        st_lottie(lottie_heart, height=120, key="heart_anim")

# ───────────────────────────────────────────────
#  2. Load model & data once (caching)
# ───────────────────────────────────────────────

@st.cache_resource
def load_model_and_data():
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

model, feature_names = load_model_and_data()

# ───────────────────────────────────────────────
#  3. Elegant Input Form
# ───────────────────────────────────────────────

st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form("patient_form"):
    st.markdown("### 📋 Patient Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
    with col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    with col3:
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])

    st.markdown("---")
    st.markdown("### 🩺 Vital Signs")
    
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.slider("Rest Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Serum Cholestrol (mg/dL)", 100, 600, 240)
    with col2:
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        oldpeak = st.slider("ST Depression (Exercise)", 0.0, 6.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("### 📊 Clinical Details")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        fbs = st.radio("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c2:
        exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c3:
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

    c4, c5 = st.columns(2)
    with c4:
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with c5:
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Fixed Defect", "Normal", "Reversible Defect"][x-1])

    submitted = st.form_submit_button("GENERATE RISK ANALYSIS")

st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
#  4. Results & Analysis (Modal)
# ───────────────────────────────────────────────

@st.dialog("Risk Analysis Results")
def show_results(prob, risk_pct, feature_names, importances):
    if prob > 0.60:
        color = "#ff4b2b"
        status = "CRITICAL RISK"
        msg = "Immediate consultation with a specialist is strongly advised."
        icon = "🚨"
    elif prob > 0.35:
        color = "#ffa502"
        status = "MODERATE RISK"
        msg = "It's recommended to schedule a check-up and monitor your vitals."
        icon = "⚠️"
    else:
        color = "#2ed573"
        status = "LOW RISK"
        msg = "Healthy profile detected. Keep maintaining a balanced lifestyle!"
        icon = "✅"

    st.markdown(f"""
    <div class="result-card" style="background-color: {color}22; border: 2px solid {color}; color: white; padding: 20px; border-radius: 15px;">
        <div style="font-size: 3rem;">{icon}</div>
        <h2 style="color: {color}; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{status}</h2>
        <div style="font-size: 3.5rem; font-weight: 800; margin: 10px 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{risk_pct:.1f}%</div>
        <p style="font-size: 1.2rem; line-height: 1.5; color: #f8fafc; font-weight: 500;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("🔍 Why this score? (Feature Importance)"):
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).head(8)
        st.bar_chart(df_imp.set_index("feature"))
        st.caption("This chart shows which factors contributed most to your specific risk assessment.")

if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prob = model.predict_proba(input_data)[0][1]
    risk_pct = prob * 100
    importances = model.feature_importances_
    
    show_results(prob, risk_pct, feature_names, importances)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    Built with ❤️ for Health Awareness • Data: UCI Machine Learning Repository<br>
    <strong>Disclaimer:</strong> This tool is for informational purposes only. Consult a doctor for medical advice.
</div>
""", unsafe_allow_html=True)
