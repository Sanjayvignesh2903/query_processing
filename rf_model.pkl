import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

# 1. Load cleaned data
clean_path = r"C:\Users\sanja\OneDrive\Documents\query processing\dataset_traffic_accident_prediction1_clean.csv"
df = pd.read_csv(clean_path)

# 2. Drop rows where Accident_Severity is 'Unknown' (optional but recommended)
df = df[df["Accident_Severity"] != "Unknown"]

# 3. Features and target
target_col = "Accident_Severity"
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# 5. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 6. Class weights for imbalance
classes = np.unique(y)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# 7. Models

log_reg_clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            class_weight=class_weight_dict
        ))
    ]
)

rf_clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=class_weight_dict
        ))
    ]
)

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train Logistic Regression (baseline)
print("\nTraining Logistic Regression...")
log_reg_clf.fit(X_train, y_train)
y_pred_lr = log_reg_clf.predict(X_test)
print("\nLogistic Regression results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# 10. Train Random Forest (main model)
print("\nTraining Random Forest...")
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 11. SAVE the Random Forest pipeline for Streamlit
joblib.dump(rf_clf, "rf_model.pkl")
print("\nSaved Random Forest model to rf_model.pkl")
