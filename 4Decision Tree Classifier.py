import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset into a pandas DataFrame
# Using a standard public repository URL for the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# 2. Assign appropriate column names
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

print("--- 1 & 2. Dataset Loaded & Columns Assigned ---")
print(df.head())

# 3. Check for missing or unrealistic zero values and handle them properly
# In the Pima dataset, features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI
# cannot biologically have a value of 0. These represent hidden missing values.
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\n--- 3. Unrealistic Zero Values Check ---")
print((df[cols_with_zeros] == 0).sum())

# Replace 0s with NaN so they can be imputed properly using the median
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Fill NaN values with the median of each respective column
for col in cols_with_zeros:
    df[col] = df[col].fillna(df[col].median())

print("\nZero values handled successfully (imputed with median).")

# Separate features (X) and target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 4. Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n--- 4. Data Split ---")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Optional: Train and evaluate the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
