import pandas as pd
import joblib
import streamlit as st

# 1. Load cleaned dataset just to get categories / ranges (optional)
DATA_PATH = "dataset_traffic_accident_prediction1_clean.csv"
df = pd.read_csv(DATA_PATH)

# 2. Load trained model (Pipeline with preprocessor + RandomForest)
#    Make sure you save this pipeline from your training script as 'rf_model.pkl'
MODEL_PATH = "rf_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Intelligent Traffic Accident Severity Predictor")

st.write("Fill in the conditions to predict **Accident_Severity** (Low / Moderate / High).")

# 3. Build input widgets based on your columns
weather = st.selectbox("Weather", sorted(df["Weather"].unique()))
road_type = st.selectbox("Road Type", sorted(df["Road_Type"].unique()))
time_of_day = st.selectbox("Time of Day", sorted(df["Time_of_Day"].unique()))
road_condition = st.selectbox("Road Condition", sorted(df["Road_Condition"].unique()))
vehicle_type = st.selectbox("Vehicle Type", sorted(df["Vehicle_Type"].unique()))
road_light = st.selectbox("Road Light Condition", sorted(df["Road_Light_Condition"].unique()))

traffic_density = st.slider("Traffic Density", float(df["Traffic_Density"].min()),
                            float(df["Traffic_Density"].max()), 1.0, step=1.0)
speed_limit = st.slider("Speed Limit", float(df["Speed_Limit"].min()),
                        float(df["Speed_Limit"].max()), 60.0, step=10.0)
num_vehicles = st.slider("Number of Vehicles", int(df["Number_of_Vehicles"].min()),
                         int(df["Number_of_Vehicles"].max()), 2, step=1)
driver_alcohol = st.selectbox("Driver Alcohol (0 = No, 1 = Yes)", [0, 1])
driver_age = st.slider("Driver Age", int(df["Driver_Age"].min()),
                       int(df["Driver_Age"].max()), 30, step=1)
driver_exp = st.slider("Driver Experience", int(df["Driver_Experience"].min()),
                       int(df["Driver_Experience"].max()), 5, step=1)
accident_flag = st.selectbox("Accident (0 = No, 1 = Yes)", [0, 1])

# 4. Create input dataframe
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

# 5. Predict
if st.button("Predict Severity"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader(f"Predicted Accident Severity: **{pred}**")
    st.write("Class probabilities:")
    for cls, p in zip(model.classes_, proba):
        st.write(f"- {cls}: {p:.2f}")
