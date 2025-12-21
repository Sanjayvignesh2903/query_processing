import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("Training Traffic Accident Severity Model...")

df = pd.read_csv(r"C:\Users\sanja\OneDrive\Documents\query processing\dataset_traffic_accident_prediction1_clean.csv")
df = df[df["Accident_Severity"] != "Unknown"]

X = df.drop(columns=["Accident_Severity"])
y = df["Accident_Severity"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

classes = np.unique(y)
class_weights = compute_class_weight("balanced", classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

rf_clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weight_dict))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_clf.fit(X_train, y_train)

joblib.dump(rf_clf, "rf_model.pkl")
print("✅ SUCCESS: rf_model.pkl created!")
print("Now you can run: streamlit run app.py")
