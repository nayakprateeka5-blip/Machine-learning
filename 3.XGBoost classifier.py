import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset into a pandas DataFrame
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("--- 1. Dataset Loaded Successfully ---")
print(df.head())
# 2. Assign / Verify appropriate column names
features_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
df = df.drop(columns=features_to_drop)
print("\n--- 2. Columns Cleaned ---")
print(df.columns.tolist())
# 3. Check for missing or zero values
print("\n--- 3. Missing Values Check ---")
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
X = df.drop(columns=["Survived"])
y = df["Survived"]
# 4. Split the dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(
    f"\n--- 4. Data Split --- \nTrain shape: {X_train.shape}, Test shape: {X_test.shape}"
)
# 5. Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
# 6. Train an XGBoost model with specific parameters
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train_scaled, y_train)
print("\n--- 6. XGBoost Model Trained Successfully ---")
# 7. Evaluate the model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- 7. Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# Note: Gradient boosting trees use feature importance (Gain/Weight) rather than linear coefficients.
feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance (Gain)": model.feature_importances_,
    }
).sort_values(by="Importance (Gain)", ascending=False)

print("\n--- 8. Feature Importance Interpretation ---")
print(feature_importance.to_string(index=False))
