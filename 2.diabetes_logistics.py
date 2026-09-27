import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Step 1 & 2: Load the Dataset and Assign Column Names ---

df = pd.read_csv("1.diabetes.csv")
column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
df.columns = column_names

# --- Step 3: Check for Missing or Zero Values ---
print("--- Zero Values Check ---")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print((df[zero_cols] == 0).sum())
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# --- Step 4: Split Dataset into Train (80%) and Test (20%) ---
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: Apply Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 6: Train Logistic Regression Model ---
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# --- Step 7: Evaluate the Model ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Precision, Recall, F1-score):")
print(classification_report(y_test, y_pred))

# --- Step 8: Interpret Model Coefficients ---
print("\n--- Coefficient Interpretation ---")
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
coefficients["Odds_Ratio"] = np.exp(coefficients["Coefficient"])
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)
print(coefficients.to_string(index=False))
