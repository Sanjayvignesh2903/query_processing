import streamlit as st
import pandas as pd
import joblib
import os
import base64
import altair as alt
from io import BytesIO

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="RiskVision AI | Traffic Safety",
    page_icon="🚦",
    layout="wide"
)

# ---------------- 2. STYLING & BACKGROUND ----------------
BG_PATH = r"C:\Users\sanja\OneDrive\Documents\query processing\background.png"
THEME_COLOR = "#d199ff"  # Neon Violet/Purple
NEON_GREEN = "#39FF14"
NEON_YELLOW = "#FFFB00"
NEON_RED = "#FF073A"

def get_img_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

img_b64 = get_img_base64(BG_PATH)

st.markdown(f"""
    <style>
    /* Full Page Background */
    .stApp {{
        background-image: url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Transparent Glass Containers */
    .main-block {{
        background: rgba(20, 0, 40, 0.85);
        padding: 40px;
        border-radius: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(209, 153, 255, 0.3);
        box-shadow: 0 0 20px rgba(209, 153, 255, 0.2);
    }}

    /* Neon Text Styling */
    label, .stMarkdown p, .stSelectbox label, .stSlider label {{
        color: {THEME_COLOR} !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
    }}
    
    h1, h2, h3 {{
        color: #ffffff !important;
        text-shadow: 0 0 10px {THEME_COLOR};
        border-left: 6px solid {THEME_COLOR};
        padding-left: 15px;
        margin-bottom: 25px;
    }}

    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(45deg, #6a0dad, {THEME_COLOR}) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        height: 3.5em !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 20px {THEME_COLOR};
    }}

    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        color: white !important;
        font-family: 'Courier New', monospace;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- 3. CORE ASSETS ----------------
MODEL_PATH = r"C:\Users\sanja\OneDrive\Documents\query processing\rf_pipeline_best_balanced.pkl"
DEFAULT_DATA = r"C:\Users\sanja\OneDrive\Documents\query processing\dataset_traffic_accident_prediction_clean_final.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

# Sidebar Setup
st.sidebar.title("📁 Data Control")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Dataset Loaded")
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
    
    # Preset Logic
    preset_cat = st.radio("Severity Focus:", ["None", "High", "Moderate", "Low"], horizontal=True)
    presets = {
        "High": ["DUI on Highway", "Midnight Storm", "High Speed Icy Road", "Blind Mountain Curve", "Multi-Vehicle Pileup"],
        "Moderate": ["Urban Peak Hour", "Construction Hazard", "Evening Wet Surface", "Slippery City Road"],
        "Low": ["Low Speed Maneuver", "Expert Driver Day", "Clear Rural Cruise"]
    }
    preset_name = st.selectbox("Quick Presets:", ["None"] + presets.get(preset_cat, []))

    vals = dict(road="City Road", weather="Clear", time="Afternoon", road_cond="Dry", light="Daylight", traffic=1.0, speed=60, alc=0, age=40, exp=15)

    if preset_name != "None":
        if "DUI" in preset_name: vals.update(dict(road="Highway", speed=110, alc=1, light="Artificial Light"))
        elif "Storm" in preset_name: vals.update(dict(road="Rural Road", weather="Stormy", road_cond="Wet", time="Night"))
        elif "Icy" in preset_name: vals.update(dict(road="Highway", speed=120, road_cond="Icy"))
        elif "Urban" in preset_name: vals.update(dict(traffic=2.0, time="Morning"))
        elif "Low Speed" in preset_name: vals.update(dict(speed=25, exp=20))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        weather = st.selectbox("🌦️ Weather", sorted(df["Weather"].unique()) if not df.empty else ["Clear"], index=list(sorted(df["Weather"].unique())).index(vals["weather"]) if not df.empty else 0)
        road_type = st.selectbox("🛣️ Road Type", sorted(df["Road_Type"].unique()) if not df.empty else ["Highway"], index=list(sorted(df["Road_Type"].unique())).index(vals["road"]) if not df.empty else 0)
        time_of_day = st.selectbox("🕒 Time", sorted(df["Time_of_Day"].unique()) if not df.empty else ["Afternoon"], index=list(sorted(df["Time_of_Day"].unique())).index(vals["time"]) if not df.empty else 0)
        road_cond = st.selectbox("🚧 Surface", sorted(df["Road_Condition"].unique()) if not df.empty else ["Dry"], index=list(sorted(df["Road_Condition"].unique())).index(vals["road_cond"]) if not df.empty else 0)
        light = st.selectbox("💡 Lighting", sorted(df["Road_Light_Condition"].unique()) if not df.empty else ["Daylight"], index=list(sorted(df["Road_Light_Condition"].unique())).index(vals["light"]) if not df.empty else 0)
    with c2:
        traffic = st.slider("🚦 Traffic Density (0=Low, 2=High)", 0.0, 2.0, float(vals["traffic"]), 1.0)
        speed = st.slider("📈 Speed Limit (km/h)", 20, 200, int(vals["speed"]), 5)
        alcohol = st.selectbox("🍷 Alcohol Influence", [0, 1], index=int(vals["alc"]), format_func=lambda x: "Detected" if x==1 else "None")
        age = st.slider("👤 Driver Age", 18, 90, int(vals["age"]))
        exp = st.slider("🏅 Driver Experience", 0, 65, int(vals["exp"]))

    if st.button("🔥 EVALUATE ACCIDENT RISK"):
        input_row = pd.DataFrame([{
            "Weather": weather, "Road_Type": road_type, "Time_of_Day": time_of_day,
            "Traffic_Density": traffic, "Speed_Limit": float(speed), "Number_of_Vehicles": 2.0,
            "Driver_Alcohol": float(alcohol), "Road_Condition": road_cond, "Vehicle_Type": "Car",
            "Driver_Age": float(age), "Driver_Experience": float(exp), "Road_Light_Condition": light,
            "Accident": 1.0, "High_Speed": 1 if speed > 80 else 0, "Night_Time": 1 if time_of_day in ["Night", "Evening"] else 0,
            "Wet_Icy": 1 if road_cond in ["Wet", "Icy"] else 0, "Young_Inexperienced": 1 if (age < 25 or exp < 2) else 0
        }])

        if model:
            pred = model.predict(input_row)[0]
            probs = model.predict_proba(input_row)[0]
            
            # Glowing Prediction Result
            res_color = NEON_RED if pred == "High" else (NEON_YELLOW if pred == "Moderate" else NEON_GREEN)
            st.markdown(f"""
                <div style="background: rgba(0,0,0,0.5); padding:30px; border-radius:20px; text-align:center; border: 3px solid {res_color}; box-shadow: 0 0 30px {res_color};">
                    <h1 style="margin:0; color:{res_color} !important; font-size: 50px; border:none;">{pred} RISK</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # ENLARGED VIBRANT CHART
            prob_df = pd.DataFrame({"Severity": model.classes_, "Probability": probs})
            chart = alt.Chart(prob_df).mark_bar(
                cornerRadiusTopLeft=15, 
                cornerRadiusTopRight=15,
                size=120  # Bigger Bars
            ).encode(
                x=alt.X("Severity", sort=["Low", "Moderate", "High"], title="Predicted Severity Level"),
                y=alt.Y("Probability", title="Confidence Level", axis=alt.Axis(format='%')),
                color=alt.Color("Severity", 
                               scale=alt.Scale(domain=["Low", "Moderate", "High"], 
                                               range=[NEON_GREEN, NEON_YELLOW, NEON_RED]), 
                               legend=None),
                tooltip=["Severity", alt.Tooltip("Probability", format=".2%")]
            ).properties(
                height=500,  # MUCH BIGGER CHART
            ).configure_axis(
                labelColor='white',
                titleColor=THEME_COLOR,
                gridColor='rgba(255,255,255,0.1)'
            ).configure_view(
                strokeOpacity=0
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.error("Model Error: Ensure pkl file is at " + MODEL_PATH)

# --- TAB 2: DATA INSIGHTS ---
with tab2:
    if not df.empty:
        st.subheader("Data Forensics")
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown("### 🗺️ Road Type vs Severity")
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                x='Road_Type', y='count()', color='Accident_Severity'
            ).properties(height=350), use_container_width=True)
        with ci2:
            st.markdown("### 🌡️ Risk Heatmap (Speed/Age)")
            st.altair_chart(alt.Chart(df).mark_rect().encode(
                x=alt.X('Driver_Age:Q', bin=True),
                y=alt.Y('Speed_Limit:Q', bin=True),
                color=alt.Color('count()', scale=alt.Scale(scheme='purples'))
            ).properties(height=350), use_container_width=True)
    else:
        st.info("Upload data to activate analysis.")

# --- TAB 3: BATCH EXPORT ---
with tab3:
    st.subheader("Mass Prediction Engine")
    if uploaded_file and model:
        if st.button("▶️ RUN FULL FILE ANALYSIS"):
            batch = df.copy()
            # Feature engineering
            batch["High_Speed"] = (batch["Speed_Limit"] > 80).astype(int)
            batch["Night_Time"] = batch["Time_of_Day"].isin(["Night", "Evening"]).astype(int)
            batch["Wet_Icy"] = batch["Road_Condition"].isin(["Wet", "Icy"]).astype(int)
            batch["Young_Inexperienced"] = ((batch["Driver_Age"] < 25) | (batch["Driver_Experience"] < 2)).astype(int)
            
            batch["Predicted_Severity"] = model.predict(batch)
            st.success(f"Analysis Complete: {len(batch)} rows processed.")
            st.dataframe(batch.head(50), use_container_width=True)
            
            csv = batch.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Predicted CSV", csv, "results.csv", "text/csv")
    else:
        st.warning("Upload a CSV in the sidebar to use Batch mode.")

st.markdown('</div>', unsafe_allow_html=True)
