import streamlit as st
import pandas as pd
import joblib
import os
import base64
import altair as alt
import sklearn

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="RiskVision AI | Traffic Safety",
    page_icon="🚦",
    layout="wide"
)

# ---------------- 2. PATH HANDLING (RELATIVE) ----------------
# This logic ensures it works on the Streamlit server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "rf_pipeline_best_balanced.pkl"
DATA_FILE = "dataset_traffic_accident_prediction.csv"
BG_FILE = "background.png"

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)
BG_PATH = os.path.join(BASE_DIR, BG_FILE)

# Neon Palette
THEME_COLOR = "#d199ff" 
NEON_GREEN = "#39FF14"
NEON_YELLOW = "#FFFB00"
NEON_RED = "#FF073A"

# ---------------- 3. STYLING ----------------
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
        background: rgba(15, 0, 30, 0.88);
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
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- 4. ROBUST DATA & MODEL LOADING ----------------
@st.cache_resource
def load_assets():
    model, df, err = None, pd.DataFrame(), None
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            err = f"Model Load Error: {e}"
    else:
        err = f"File '{MODEL_FILE}' missing from GitHub repo."

    # Load Data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    
    return model, df, err

model, df_default, load_error = load_assets()

# ---------------- 5. DASHBOARD UI ----------------
st.markdown('<div class="main-block">', unsafe_allow_html=True)
st.title("🛡️ RiskVision AI Predictor")

if load_error:
    st.error(f"📡 System Alert: {load_error}")
    st.stop()

st.sidebar.info(f"System: Sklearn {sklearn.__version__} | Match: 1.5.1")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type="csv")
active_df = pd.read_csv(uploaded_file) if uploaded_file else df_default

tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "📊 Data Insights", "📂 Batch Process"])

with tab1:
    st.subheader("Configure Situation Parameters")
    c1, c2 = st.columns(2)
    
    with c1:
        weather = st.selectbox("🌦️ Weather", sorted(active_df["Weather"].unique()) if not active_df.empty else ["Clear"])
        road_type = st.selectbox("🛣️ Road Type", sorted(active_df["Road_Type"].unique()) if not active_df.empty else ["Highway"])
        time_of_day = st.selectbox("🕒 Time", ["Morning", "Afternoon", "Evening", "Night"])
        speed = st.slider("📈 Speed Limit (km/h)", 20, 200, 60)
    
    with c2:
        traffic = st.slider("🚦 Traffic Density (0-2)", 0.0, 2.0, 1.0)
        age = st.slider("👤 Driver Age", 18, 90, 35)
        exp = st.slider("🏅 Driver Experience", 0, 65, 10)
        road_cond = st.selectbox("🚧 Surface", ["Dry", "Wet", "Icy", "Under Construction"])

    if st.button("🔥 EVALUATE ACCIDENT RISK"):
        # MATCH FEATURE LOGIC
        input_row = pd.DataFrame([{
            "Weather": weather, "Road_Type": road_type, "Time_of_Day": time_of_day,
            "Traffic_Density": traffic, "Speed_Limit": float(speed), "Number_of_Vehicles": 2.0,
            "Driver_Alcohol": 0.0, "Road_Condition": road_cond, "Vehicle_Type": "Car",
            "Driver_Age": float(age), "Driver_Experience": float(exp), "Road_Light_Condition": "Daylight",
            "Accident": 1.0,
            "High_Speed": 1 if speed > 80 else 0,
            "Night_Time": 1 if time_of_day in ["Night", "Evening"] else 0,
            "Wet_Icy": 1 if road_cond in ["Wet", "Icy"] else 0,
            "Young_Inexperienced": 1 if (age < 25 or exp < 2) else 0
        }])
        
        try:
            pred = model.predict(input_row)[0]
            probs = model.predict_proba(input_row)[0]
            res_color = NEON_RED if pred == "High" else (NEON_YELLOW if pred == "Moderate" else NEON_GREEN)
            
            st.markdown(f'<div style="background:rgba(0,0,0,0.6); padding:30px; border-radius:20px; text-align:center; border:3px solid {res_color}; box-shadow:0 0 30px {res_color};"><h1 style="margin:0; color:{res_color} !important; font-size:55px; border:none;">{pred} RISK</h1></div>', unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({"Severity": model.classes_, "Probability": probs})
            chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=15, cornerRadiusTopRight=15, size=90).encode(
                x=alt.X("Severity", sort=["Low", "Moderate", "High"]),
                y=alt.Y("Probability", axis=alt.Axis(format='%')),
                color=alt.Color("Severity", scale=alt.Scale(domain=["Low", "Moderate", "High"], range=[NEON_GREEN, NEON_YELLOW, NEON_RED]), legend=None)
            ).properties(height=500)
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction logic error: {e}")

with tab2:
    if not active_df.empty:
        st.subheader("Current Dataset View")
        st.dataframe(active_df.head(50), use_container_width=True)
    else:
        st.warning("No data found.")

with tab3:
    st.subheader("Mass Batch Export")
    if st.button("▶️ RUN BATCH PREDICTION"):
        batch = active_df.copy()
        # Add required dummy columns if missing
        batch["High_Speed"] = (batch["Speed_Limit"] > 80).astype(int)
        batch["Night_Time"] = batch["Time_of_Day"].isin(["Night", "Evening"]).astype(int)
        batch["Wet_Icy"] = batch["Road_Condition"].isin(["Wet", "Icy"]).astype(int)
        batch["Young_Inexperienced"] = ((batch["Driver_Age"] < 25) | (batch["Driver_Experience"] < 2)).astype(int)
        
        batch["Predicted_Severity"] = model.predict(batch)
        st.dataframe(batch.head(50), use_container_width=True)
        csv = batch.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results", csv, "batch_results.csv", "text/csv")

st.markdown('</div>', unsafe_allow_html=True)
