import streamlit as st
import pandas as pd
import joblib
import os
import base64
import altair as alt

# --- IMPORTANT: If you used any custom functions or classes during training, 
# --- you MUST paste their code here (above the load_model function).

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="RiskVision AI | Traffic Safety",
    page_icon="🚦",
    layout="wide"
)

# ---------------- 2. PATH HANDLING (RELATIVE) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# EXACT filenames as they appear in your GitHub repo
BG_FILE = "background.png"
MODEL_FILE = "rf_pipeline_best_balanced.pkl"
DATA_FILE = "dataset_traffic_accident_prediction.csv"

BG_PATH = os.path.join(BASE_DIR, BG_FILE)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

# Color Palette
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

# ---------------- 4. ROBUST ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    model = None
    df = pd.DataFrame()
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except AttributeError as e:
            st.error(f"⚠️ **AttributeError:** Your model is looking for a library or class that isn't loaded. {e}")
        except Exception as e:
            st.error(f"⚠️ **Loading Error:** {e}")
    else:
        st.error(f"❌ Model file NOT found at {MODEL_PATH}. Check GitHub filenames.")

    # Load Data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        st.warning(f"⚠️ Default dataset {DATA_FILE} not found.")
        
    return model, df

model, df_default = load_assets()

# Sidebar Setup
st.sidebar.title("📁 Data Control")
uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type="csv")
active_df = pd.read_csv(uploaded_file) if uploaded_file else df_default

# ---------------- 5. MAIN UI ----------------
st.markdown('<div class="main-block">', unsafe_allow_html=True)
st.title("⚡ RiskVision AI Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "📊 Insights", "💾 Batch Export"])

with tab1:
    st.subheader("Scenario Inputs")
    c1, c2 = st.columns(2)
    
    # Logic to handle empty data safely
    weather_list = sorted(active_df["Weather"].unique()) if "Weather" in active_df.columns else ["Clear"]
    road_list = sorted(active_df["Road_Type"].unique()) if "Road_Type" in active_df.columns else ["City Road"]

    with c1:
        weather = st.selectbox("🌦️ Weather", weather_list)
        rtype = st.selectbox("🛣️ Road Type", road_list)
        speed = st.slider("📈 Speed Limit", 20, 200, 60)
        alc = st.selectbox("🍷 Alcohol Influence", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c2:
        traffic = st.slider("🚦 Traffic Density", 0.0, 2.0, 1.0)
        age = st.slider("👤 Driver Age", 18, 90, 35)
        exp = st.slider("🏅 Experience", 0, 60, 10)
        rcond = st.selectbox("🚧 Surface Condition", ["Dry", "Wet", "Icy"])

    if st.button("🔥 EVALUATE ACCIDENT RISK"):
        if model:
            # Prepare data to match EXACT columns your model expects
            input_row = pd.DataFrame([{
                "Weather": weather, "Road_Type": rtype, "Time_of_Day": "Afternoon",
                "Traffic_Density": traffic, "Speed_Limit": float(speed), "Number_of_Vehicles": 2.0,
                "Driver_Alcohol": float(alc), "Road_Condition": rcond, "Vehicle_Type": "Car",
                "Driver_Age": float(age), "Driver_Experience": float(exp), "Road_Light_Condition": "Daylight",
                "Accident": 1.0,
                "High_Speed": 1 if speed > 80 else 0,
                "Night_Time": 0, "Wet_Icy": 1 if rcond in ["Wet", "Icy"] else 0,
                "Young_Inexperienced": 1 if (age < 25 or exp < 2) else 0
            }])
            
            pred = model.predict(input_row)[0]
            probs = model.predict_proba(input_row)[0]
            
            res_color = NEON_RED if pred == "High" else (NEON_YELLOW if pred == "Moderate" else NEON_GREEN)
            st.markdown(f'<div style="background: rgba(0,0,0,0.5); padding:25px; border-radius:20px; text-align:center; border: 3px solid {res_color}; box-shadow: 0 0 30px {res_color};"><h1 style="margin:0; color:{res_color} !important; border:none; font-size:45px;">{pred} RISK</h1></div>', unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({"Severity": model.classes_, "Probability": probs})
            chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=15, cornerRadiusTopRight=15, size=80).encode(
                x=alt.X("Severity", sort=["Low", "Moderate", "High"]),
                y=alt.Y("Probability", axis=alt.Axis(format='%')),
                color=alt.Color("Severity", scale=alt.Scale(domain=["Low", "Moderate", "High"], range=[NEON_GREEN, NEON_YELLOW, NEON_RED]), legend=None)
            ).properties(height=450)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.error("Model is not loaded. Fix the AttributeError shown above.")

with tab2:
    if not active_df.empty:
        st.subheader("Dataset Summary")
        st.dataframe(active_df.head(10), use_container_width=True)
    else:
        st.info("No data found to display.")

with tab3:
    st.subheader("Mass Batch Processing")
    if model and not active_df.empty:
        if st.button("▶️ RUN FULL FILE PREDICTION"):
            # Minimal feature engineering for the batch
            batch = active_df.copy()
            batch["High_Speed"] = (batch["Speed_Limit"] > 80).astype(int)
            batch["Night_Time"] = batch["Time_of_Day"].isin(["Night", "Evening"]).astype(int)
            batch["Wet_Icy"] = batch["Road_Condition"].isin(["Wet", "Icy"]).astype(int)
            batch["Young_Inexperienced"] = ((batch["Driver_Age"] < 25) | (batch["Driver_Experience"] < 2)).astype(int)
            
            batch["Predicted_Severity"] = model.predict(batch)
            st.dataframe(batch.head(50), use_container_width=True)
            csv = batch.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Predicted CSV", csv, "results.csv", "text/csv")

st.markdown('</div>', unsafe_allow_html=True)
