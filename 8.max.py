import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
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
df = pd.read_csv(url, names=column_names)

# 3. Handle unrealistic zero values
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 4 & 5. Define features and target, then split dataset
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6 & 7. Train and evaluate the full Decision Tree (for comparison)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
y_pred_full = dt_full.predict(X_test)

# 8. Train another Decision Tree with restricted depth (max_depth=3)
dt_restricted = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_restricted.fit(X_train, y_train)

# Predictions for restricted model
y_pred_restricted = dt_restricted.predict(X_test)

print("=== Restricted Depth (max_depth=3) Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred_restricted))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_restricted))
print("Classification Report:\n", classification_report(y_test, y_pred_restricted))

# Compare Performance Insights
print("\n=== Performance Comparison ===")
print(f"Full Tree Accuracy: {accuracy_score(y_test, y_pred_full):.4f}")
print(f"Restricted Tree Accuracy: {accuracy_score(y_test, y_pred_restricted):.4f}")

# Extract and display feature importance
print("\n=== Feature Importances (Full Tree) ===")
feature_importances = pd.DataFrame(
    {"Feature": X.columns, "Importance": dt_full.feature_importances_}
)
print(
    feature_importances.sort_values(by="Importance", ascending=False).to_string(
        index=False
    )
)

print("\n=== Feature Importances (Restricted Tree, max_depth=3) ===")
feature_importances_restricted = pd.DataFrame(
    {"Feature": X.columns, "Importance": dt_restricted.feature_importances_}
)
print(
    feature_importances_restricted.sort_values(
        by="Importance", ascending=False
    ).to_string(index=False)
)
