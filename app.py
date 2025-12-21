import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# --------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Intelligent Traffic Accident Severity Predictor",
    layout="wide"
)

# --------- FILE PATHS (RELATIVE) ----------
DATA_PATH = "dataset_traffic_accident_prediction1_clean.csv"
MODEL_PATH = "rf_model.pkl"
BACKGROUND_IMAGE = "background.png"   # file in same repo

# --------- CUSTOM CSS: BACKGROUND + COLORS ----------
bg_path = Path(BACKGROUND_IMAGE).as_posix()

st.markdown(
    f"""
    <style>
    /* Main app background image */
    [data-testid="stAppViewContainer"] {{
        background: url("{bg_path}") no-repeat center center fixed;
        background-size: cover;
    }}

    /* Dark overlay for content */
    .main-overlay {{
        background: linear-gradient(
            to bottom right,
            rgba(0, 0, 0, 0.82),
            rgba(0, 0, 0, 0.90)
        );
        padding: 24px 32px;
        border-radius: 18px;
        border: 1px solid #00E5FF;   /* Primary accent cyan */
        box-shadow: 0 0 28px rgba(0, 0, 0, 0.9);
    }}

    html, body, [class*="css"] {{
        color: #FFFFFF;              /* Primary text white */
        font-family: "Segoe UI", sans-serif;
    }}

    h1, h2, h3, h4 {{
        color: #FFFFFF;
        font-weight: 800;
    }}

    .accent-safe {{
        color: #00E5FF;              /* Safe / primary accent */
    }}

    .secondary-text {{
        color: #B0BEC5;              /* Secondary text */
        font-size: 0.9rem;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {{
        background: rgba(0, 0, 0, 0.95);
        border-right: 1px solid #00E5FF;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: #FFFFFF !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #00E5FF;
        color: #000000;
        border-radius: 8px;
        border: 1px solid #00B8D4;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: #FF3D00;   /* Warning / risk */
        color: #FFFFFF;
        border-color: #FF6E40;
    }}

    /* Risk colors */
    .risk-high {{
        color: #FF3D00;              /* Neon red */
        font-weight: 700;
    }}
    .risk-medium {{
        color: #FFC107;              /* Amber caution */
        font-weight: 700;
    }}
    .risk-low {{
        color: #00E5FF;              /* Cyan safe */
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- LOAD DATA & MODEL ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# --------- MAIN UI ----------
st.markdown('<div class="main-overlay">', unsafe_allow_html=True)

st.markdown(
    '<h1>Intelligent Traffic Accident Severity Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="secondary-text">Leverage real-time conditions to estimate accident severity across complex traffic networks.</p>',
    unsafe_allow_html=True,
)

st.sidebar.title("Input Conditions")

# Sidebar inputs
weather = st.sidebar.selectbox("Weather", sorted(df["Weather"].unique()))
road_type = st.sidebar.selectbox("Road Type", sorted(df["Road_Type"].unique()))
time_of_day = st.sidebar.selectbox("Time of Day", sorted(df["Time_of_Day"].unique()))
road_condition = st.sidebar.selectbox("Road Condition", sorted(df["Road_Condition"].unique()))
vehicle_type = st.sidebar.selectbox("Vehicle Type", sorted(df["Vehicle_Type"].unique()))
road_light = st.sidebar.selectbox("Road Light Condition", sorted(df["Road_Light_Condition"].unique()))

traffic_density = st.sidebar.slider(
    "Traffic Density",
    float(df["Traffic_Density"].min()),
    float(df["Traffic_Density"].max()),
    float(df["Traffic_Density"].median()),
    step=1.0,
)
speed_limit = st.sidebar.slider(
    "Speed Limit",
    float(df["Speed_Limit"].min()),
    float(df["Speed_Limit"].max()),
    float(df["Speed_Limit"].median()),
    step=10.0,
)
num_vehicles = st.sidebar.slider(
    "Number of Vehicles",
    int(df["Number_of_Vehicles"].min()),
    int(df["Number_of_Vehicles"].max()),
    int(df["Number_of_Vehicles"].median()),
    step=1,
)
driver_alcohol = st.sidebar.selectbox("Driver Alcohol (0 = No, 1 = Yes)", [0, 1])
driver_age = st.sidebar.slider(
    "Driver Age",
    int(df["Driver_Age"].min()),
    int(df["Driver_Age"].max()),
    int(df["Driver_Age"].median()),
    step=1,
)
driver_exp = st.sidebar.slider(
    "Driver Experience (years)",
    int(df["Driver_Experience"].min()),
    int(df["Driver_Experience"].max()),
    int(df["Driver_Experience"].median()),
    step=1,
)
accident_flag = st.sidebar.selectbox("Accident Flag (0 = No, 1 = Yes)", [0, 1])

# Input row
input_dict = {
    "Weather": weather,
    "Road_Type": road_type,
    "Time_of_Day": time_of_day,
    "Road_Condition": road_condition,
    "Vehicle_Type": vehicle_type,
    "Road_Light_Condition": road_light,
    "Traffic_Density": traffic_density,
    "Speed_Limit": speed_limit,
    "Number_of_Vehicles": num_vehicles,
    "Driver_Alcohol": int(driver_alcohol),
    "Driver_Age": driver_age,
    "Driver_Experience": driver_exp,
    "Accident": int(accident_flag),
}
input_df = pd.DataFrame([input_dict])

st.subheader("Current Input Snapshot")
st.dataframe(input_df, use_container_width=True)

# Prediction
if st.button("Run Severity Prediction"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    if pred == "High":
        risk_class = "risk-high"
    elif pred == "Moderate":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    st.markdown(
        f'<h2 class="{risk_class}">Predicted Severity: {pred}</h2>',
        unsafe_allow_html=True,
    )

    st.markdown('<h3 class="accent-safe">Class Probabilities</h3>', unsafe_allow_html=True)
    for cls, p in zip(classes, proba):
        color_class = (
            "risk-high" if cls == "High"
            else "risk-medium" if cls == "Moderate"
            else "risk-low"
        )
        st.markdown(
            f'<p class="{color_class}">{cls}: {p:.2%}</p>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)
