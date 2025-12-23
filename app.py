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
# Use relative paths so it works on any server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_PATH = os.path.join(BASE_DIR, "background.png")
MODEL_PATH = os.path.join(BASE_DIR, "rf_pipeline_best_balanced.pkl")
DEFAULT_DATA = os.path.join(BASE_DIR, "dataset_traffic_accident_prediction_clean_final.csv")

THEME_COLOR = "#d199ff" 
NEON_GREEN = "#39FF14"
NEON_YELLOW = "#FFFB00"
NEON_RED = "#FF073A"

def get_img_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

img_b64 = get_img_base64(BG_PATH)

# CSS Injection
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_b64 if img_b64 else ''}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #0f001e; /* Fallback color */
    }}
    .main-block {{
        background: rgba(20, 0, 40, 0.85);
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

# ---------------- 3. CORE ASSETS ----------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# Sidebar Setup
st.sidebar.title("📁 Data Control")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(DEFAULT_DATA):
    df = pd.read_csv(DEFAULT_DATA)
else:
    df = pd.DataFrame()

# ---------------- 4. MAIN LAYOUT ----------------
st.markdown('<div class="main-block">', unsafe_allow_html=True)
st.title("⚡ RiskVision AI Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction Engine", "📊 Deep Insights", "💾 Batch Export"])

# --- TAB 1: INDIVIDUAL PREDICTOR ---
with tab1:
    st.subheader("Scenario Configuration")
    
    preset_cat = st.radio("Severity Focus:", ["None", "High", "Moderate", "Low"], horizontal=True)
    presets = {
        "High": ["DUI on Highway", "Midnight Storm", "High Speed Icy Road"],
        "Moderate": ["Urban Peak Hour", "Construction Hazard"],
        "Low": ["Low Speed Maneuver", "Expert Driver Day"]
    }
    preset_name = st.selectbox("Quick Presets:", ["None"] + presets.get(preset_cat, []))

    # Logic remains same as previous version...
    vals = dict(road="City Road", weather="Clear", time="Afternoon", road_cond="Dry", light="Daylight", traffic=1.0, speed=60, alc=0, age=40, exp=15)

    if preset_name != "None":
        if "DUI" in preset_name: vals.update(dict(road="Highway", speed=110, alc=1, light="Artificial Light"))
        elif "Storm" in preset_name: vals.update(dict(road="Rural Road", weather="Stormy", road_cond="Wet", time="Night"))
        elif "Icy" in preset_name: vals.update(dict(road="Highway", speed=120, road_cond="Icy"))

    c1, c2 = st.columns(2)
    with c1:
        weather = st.selectbox("🌦️ Weather", sorted(df["Weather"].unique()) if not df.empty else ["Clear"], index=0)
        road_type = st.selectbox("🛣️ Road Type", sorted(df["Road_Type"].unique()) if not df.empty else ["Highway"], index=0)
        time_of_day = st.selectbox("🕒 Time", sorted(df["Time_of_Day"].unique()) if not df.empty else ["Afternoon"], index=0)
        road_cond = st.selectbox("🚧 Surface", sorted(df["Road_Condition"].unique()) if not df.empty else ["Dry"], index=0)
        light = st.selectbox("💡 Lighting", sorted(df["Road_Light_Condition"].unique()) if not df.empty else ["Daylight"], index=0)
    with c2:
        traffic = st.slider("🚦 Traffic Density", 0.0, 2.0, float(vals["traffic"]), 1.0)
        speed = st.slider("📈 Speed Limit", 20, 200, int(vals["speed"]), 5)
        alcohol = st.selectbox("🍷 Alcohol Influence", [0, 1], index=int(vals["alc"]), format_func=lambda x: "Detected" if x==1 else "None")
        age = st.slider("👤 Driver Age", 18, 90, int(vals["age"]))
        exp = st.slider("🏅 Driver Experience", 0, 65, int(vals["exp"]))

    if st.button("🔥 EVALUATE ACCIDENT RISK"):
        if model:
            input_row = pd.DataFrame([{
                "Weather": weather, "Road_Type": road_type, "Time_of_Day": time_of_day,
                "Traffic_Density": traffic, "Speed_Limit": float(speed), "Number_of_Vehicles": 2.0,
                "Driver_Alcohol": float(alcohol), "Road_Condition": road_cond, "Vehicle_Type": "Car",
                "Driver_Age": float(age), "Driver_Experience": float(exp), "Road_Light_Condition": light,
                "Accident": 1.0, "High_Speed": 1 if speed > 80 else 0, "Night_Time": 1 if time_of_day in ["Night", "Evening"] else 0,
                "Wet_Icy": 1 if road_cond in ["Wet", "Icy"] else 0, "Young_Inexperienced": 1 if (age < 25 or exp < 2) else 0
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
            chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=15, cornerRadiusTopRight=15, size=120).encode(
                x=alt.X("Severity", sort=["Low", "Moderate", "High"]),
                y=alt.Y("Probability", axis=alt.Axis(format='%')),
                color=alt.Color("Severity", scale=alt.Scale(domain=["Low", "Moderate", "High"], range=[NEON_GREEN, NEON_YELLOW, NEON_RED]), legend=None)
            ).properties(height=500)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.error("Model file not found in the repository folder.")

# Tabs 2 and 3 logic stays the same...
# (Batch processing and insights should use 'df' which is now handled by relative paths)

st.markdown('</div>', unsafe_allow_html=True)
