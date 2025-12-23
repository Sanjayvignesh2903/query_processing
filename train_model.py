

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load dataset
path = r"C:\Users\sanja\OneDrive\Documents\query processing\dataset_traffic_accident_prediction.csv"
df = pd.read_csv(path)

df.columns = df.columns.str.strip()

# ----------------- CLEANING -----------------

# Drop rows with missing target
df = df.dropna(subset=["Accident_Severity"])

# Remove Unknown class
df = df[df["Accident_Severity"] != "Unknown"]

# Drop rows with too many missing values
max_missing_per_row = 3
df = df[df.isna().sum(axis=1) <= max_missing_per_row]

# Detect numeric / categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# Impute numeric with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Impute categorical with mode / "Unknown"
for col in cat_cols:
    if df[col].isna().all():
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype(str).str.strip()

# ----------------- FEATURE ENGINEERING (small improvement) -----------------

# 1) High speed flag
if "Speed_Limit" in df.columns:
    df["High_Speed"] = (df["Speed_Limit"] > 80).astype(int)

# 2) Night‑time flag
if "Time_of_Day" in df.columns:
    df["Night_Time"] = df["Time_of_Day"].isin(["Evening", "Night"]).astype(int)

# 3) Wet or icy road flag
if "Road_Condition" in df.columns:
    df["Wet_Icy"] = df["Road_Condition"].isin(["Wet", "Icy"]).astype(int)

# 4) Young & inexperienced driver flag
if "Driver_Age" in df.columns and "Driver_Experience" in df.columns:
    df["Young_Inexperienced"] = (
        (df["Driver_Age"] < 25) | (df["Driver_Experience"] < 2)
    ).astype(int)

# Save cleaned + engineered version (optional)
clean_path = r"C:\Users\sanja\OneDrive\Documents\query processing\dataset_traffic_accident_prediction_clean_final.csv"
df.to_csv(clean_path, index=False)
print("Cleaned & engineered data saved to:", clean_path)

# ----------------- FEATURES / SPLIT -----------------

X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# ----------------- SMOTE + RANDOM FOREST -----------------

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

pipe = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("model", rf),
])

param_distributions = {
    "model__n_estimators": randint(200, 600),
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 5),
    "model__max_features": ["sqrt", "log2", 0.5],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=40,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest params:", search.best_params_)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ----------------- SAVE MODEL -----------------

import joblib
model_path = r"C:\Users\sanja\OneDrive\Documents\query processing\rf_pipeline_best_balanced.pkl"
joblib.dump(best_model, model_path)
print("\nBest model with feature engineering saved to:", model_path)
