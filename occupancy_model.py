# === Smart Campus - Occupancy Prediction ===
# Goal: Predict whether a room is occupied (1) or empty (0) based on sensor data (Temperature, Humidity, CO2, Light, etc.)
# Why? This demonstrates how AI can help optimize energy usage (e.g., lights/AC only when the room is occupied).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------
# 1) Load & Prepare Data
# -------------------
# Load the dataset from CSV
df = pd.read_csv("data.csv")

# Remove missing values to avoid errors during training
df = df.dropna()

# Separate features (X) and target (y)
# - X: all numeric sensor values except "date" and "Occupancy"
# - y: Occupancy (0 = empty, 1 = occupied)
X = df.drop(columns=["date", "Occupancy"], errors="ignore")
y = df["Occupancy"]

# -------------------
# 2) Train/Test Split
# -------------------
# Split data into training (80%) and testing (20%)
# Stratify=y ensures that the ratio of occupied vs empty is similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------
# 3) Feature Scaling
# -------------------
# Logistic Regression is sensitive to feature scale, so we standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 4) Train Models
# -------------------
# Logistic Regression (simple, interpretable, good baseline)
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_scaled, y_train)

# Random Forest (powerful, non-linear model, can handle feature importance)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # no scaling needed for trees

# -------------------
# 5) Predictions
# -------------------
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

# -------------------
# 6) Evaluation
# -------------------
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -------------------
# 7) Visualizations for Presentation
# -------------------

# Confusion Matrix - shows where the model is correct/incorrect
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Empty", "Occupied"],
            yticklabels=["Empty", "Occupied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Feature Importance - shows which sensors matter most for predicting Occupancy
importances = rf.feature_importances_
feat_names = X.columns
feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance - Random Forest")
plt.show()

# Prediction vs Actual - shows model performance over a time slice
plt.figure(figsize=(12,5))
plt.plot(y_test.values[:100], label="Actual", drawstyle="steps-post")
plt.plot(y_pred_rf[:100], label="Predicted", linestyle="--", drawstyle="steps-post")
plt.legend()
plt.title("Occupancy Prediction vs Actual (First 100 samples)")
plt.xlabel("Sample")
plt.ylabel("Occupancy")
plt.show()
