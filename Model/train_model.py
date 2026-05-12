import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../Dataset/Combined_Dataset/CICIDS2017_GROUPED.csv")

print("Dataset Shape:", df.shape)
print("\nLabel Distribution:")
print(df["Label"].value_counts())


# =========================
# CLEAN (extra safety)
# =========================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


# =========================
# SPLIT FEATURES / LABEL
# =========================
X = df.drop("Label", axis=1)
y_raw = df["Label"]


# =========================
# LABEL ENCODING (FIX FOR XGBOOST)
# =========================
le = LabelEncoder()
y = le.fit_transform(y_raw)

print("\nLabel Mapping:")
for i, cls in enumerate(le.classes_):
    print(i, "=", cls)


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =========================
# MODEL 1 - RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

evaluate(rf, X_test, y_test, "Random Forest")


# =========================
# MODEL 2 - XGBOOST
# =========================
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)

print("\nTraining XGBoost...")
xgb.fit(X_train, y_train)

evaluate(xgb, X_test, y_test, "XGBoost")


# =========================
# SAVE MODELS (IMPORTANT FOR FASTAPI)
# =========================
joblib.dump(rf, "../Model/rf_model.pkl")
joblib.dump(xgb, "../Model/xgb_model.pkl")
joblib.dump(le, "../Model/label_encoder.pkl")

print("\nModels saved successfully!")