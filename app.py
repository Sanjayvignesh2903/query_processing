import streamlit as st
import pandas as pd
import joblib
import os
import base64
import altair as alt

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="RiskVision AI | Traffic Safety",
    page_icon="🚦",
    layout="wide"
)

# ---------------- 2. PATH HANDLING (DEPLOYMENT READY) ----------------
# This logic finds the folder on the Streamlit server automatically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Update filenames to match your exact files on GitHub
BG_PATH = os.path.join(BASE_DIR, "background.png")
MODEL_PATH = os.path.join(BASE_DIR, "rf_pipeline_best_balanced.pkl")
DATA_PATH = os.path.join(BASE_DIR, "dataset_traffic_accident_prediction.csv")

# Vibrant Neon Palette
THEME_COLOR = "#d199ff" 
NEON_GREEN = "#39FF14"
NEON_YELLOW = "#FFFB00"
NEON_RED = "#FF073A"

# ---------------- 3. STYLING & BACKGROUND ----------------
def get_img_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

img_b64 = get_img_base64(BG_PATH)

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_b64 if img_b64 else ''}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #0f001e;
    }}
    .main-block {{
        background: rgba(15, 0, 30, 0.85);
        padding: 40px;
        border-radius: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(209, 153, 255, 0.3);
    }}
    label, .stMarkdown p, .stSelectbox label, .stSlider label {{
        color: {THEME_COLOR} !important;
        font-weight: 700 !important;
    }}
    h1, h2, h3 {{
        color: #ffffff !important;
        text-shadow: 0 0 10px {THEME_COLOR};
        border-left: 6px solid {THEME_COLOR};
        padding-left: 15px;
    }}
    .stButton>button {{
        background: linear-gradient(45deg, #6a0dad, {THEME_COLOR}) !important;
        color: white !important;
        font-weight: bold !important;
        height: 3.5em !important;
        border-radius: 12px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- 4. ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    # Loading model (This is where imblearn is required)
    m = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    # Loading default data
    d = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    return m, d

model, df_default = load_assets()

# Sidebar Setup
st.sidebar.title("📁 Data Control")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type="csv")
df = pd.read_csv(uploaded_file) if uploaded_file else df_default

# ---------------- 5. MAIN UI ----------------
st.markdown('<div class="main-block">', unsafe_allow_html=True)
st.title("⚡ RiskVision AI Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Individual Predictor", "📊 Data Insights", "💾 Batch Export"])

with tab1:
    st.subheader("Scenario Configuration")
    
    preset_cat = st.radio("Severity Focus:", ["None", "High", "Moderate", "Low"], horizontal=True)
    presets = {
        "High": ["DUI on Highway", "Midnight Storm", "High Speed Icy Road", "Blind Mountain Curve"],
        "Moderate": ["Urban Peak Hour", "Construction Hazard", "Evening Wet Surface"],
        "Low": ["Low Speed Maneuver", "Expert Driver Day", "Clear Rural Cruise"]
    }
    preset_name = st.selectbox("Quick Presets:", ["None"] + presets.get(preset_cat, []))

    # Values mapping
    v = dict(road="City Road", weather="Clear", time="Afternoon", cond="Dry", light="Daylight", traffic=1.0, speed=60, alc=0, age=40, exp=15)
    if "DUI" in preset_name: v.update(dict(road="Highway", speed=110, alc=1, light="Artificial Light"))
    elif "Storm" in preset_name: v.update(dict(road="Rural Road", weather="Stormy", cond="Wet", time="Night"))
    elif "Icy" in preset_name: v.update(dict(road="Highway", speed=125, cond="Icy"))

    c1, c2 = st.columns(2)
    with c1:
        weather = st.selectbox("🌦️ Weather", sorted(df["Weather"].unique()) if not df.empty else ["Clear"], index=0)
        rtype = st.selectbox("🛣️ Road Type", sorted(df["Road_Type"].unique()) if not df.empty else ["Highway"], index=0)
        tod = st.selectbox("🕒 Time", sorted(df["Time_of_Day"].unique()) if not df.empty else ["Afternoon"], index=0)
        rcond = st.selectbox("🚧 Surface", sorted(df["Road_Condition"].unique()) if not df.empty else ["Dry"], index=0)
        rlight = st.selectbox("💡 Lighting", sorted(df["Road_Light_Condition"].unique()) if not df.empty else ["Daylight"], index=0)
    with c2:
        traffic = st.slider("🚦 Traffic Density", 0.0, 2.0, float(v["traffic"]), 1.0)
        speed = st.slider("📈 Speed Limit", 20, 200, int(v["speed"]), 5)
        alc = st.selectbox("🍷 Alcohol Influence", [0, 1], index=int(v["alc"]), format_func=lambda x: "Detected" if x==1 else "None")
        age = st.slider("👤 Driver Age", 18, 90, int(v["age"]))
        exp = st.slider("🏅 Driver Experience", 0, 65, int(v["exp"]))

    if st.button("🔥 EVALUATE ACCIDENT RISK"):
        if model:
            input_row = pd.DataFrame([{
                "Weather": weather, "Road_Type": rtype, "Time_of_Day": tod,
                "Traffic_Density": traffic, "Speed_Limit": float(speed), "Number_of_Vehicles": 2.0,
                "Driver_Alcohol": float(alc), "Road_Condition": rcond, "Vehicle_Type": "Car",
                "Driver_Age": float(age), "Driver_Experience": float(exp), "Road_Light_Condition": rlight,
                "Accident": 1.0,
                "High_Speed": 1 if speed > 80 else 0,
                "Night_Time": 1 if tod in ["Night", "Evening"] else 0,
                "Wet_Icy": 1 if rcond in ["Wet", "Icy"] else 0,
                "Young_Inexperienced": 1 if (age < 25 or exp < 2) else 0
            }])
            
            pred = model.predict(input_row)[0]
            probs = model.predict_proba(input_row)[0]
            
            res_color = NEON_RED if pred == "High" else (NEON_YELLOW if pred == "Moderate" else NEON_GREEN)
            st.markdown(f"""
                <div style="background: rgba(0,0,0,0.5); padding:30px; border-radius:20px; text-align:center; border: 3px solid {res_color}; box-shadow: 0 0 30px {res_color};">
                    <h1 style="margin:0; color:{res_color} !important; font-size: 50px; border:none;">{pred} RISK</h1>
                </div>
            """, unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({"Severity": model.classes_, "Probability": probs})
            chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=15, cornerRadiusTopRight=15, size=100).encode(
                x=alt.X("Severity", sort=["Low", "Moderate", "High"]),
                y=alt.Y("Probability", axis=alt.Axis(format='%')),
                color=alt.Color("Severity", scale=alt.Scale(domain=["Low", "Moderate", "High"], range=[NEON_GREEN, NEON_YELLOW, NEON_RED]), legend=None)
            ).properties(height=500)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.error("⚠️ Model file not found. Check GitHub repo.")

with tab2:
    if not df.empty:
        st.subheader("Data Forensics")
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='Weather', y='count()', color='Accident_Severity').properties(height=400), use_container_width=True)
        st.dataframe(df.head(20), use_container_width=True)

with tab3:
    st.subheader("Batch Prediction")
    if uploaded_file and model:
        if st.button("▶️ RUN FULL ANALYSIS"):
            batch = df.copy()
            batch["High_Speed"] = (batch["Speed_Limit"] > 80).astype(int)
            batch["Night_Time"] = batch["Time_of_Day"].isin(["Night", "Evening"]).astype(int)
            batch["Wet_Icy"] = batch["Road_Condition"].isin(["Wet", "Icy"]).astype(int)
            batch["Young_Inexperienced"] = ((batch["Driver_Age"] < 25) | (batch["Driver_Experience"] < 2)).astype(int)
            batch["Predicted_Severity"] = model.predict(batch)
            st.dataframe(batch.head(50), use_container_width=True)
            csv = batch.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results", csv, "predictions.csv", "text/csv")

st.markdown('</div>', unsafe_allow_html=True)
