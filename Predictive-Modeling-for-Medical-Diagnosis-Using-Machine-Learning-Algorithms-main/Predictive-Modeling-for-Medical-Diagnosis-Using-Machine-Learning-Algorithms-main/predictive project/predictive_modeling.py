"""
==========================================================
PREDICTIVE MODELING USING MACHINE LEARNING
==========================================================
Goal   : Build a model to predict outcomes based on given data.
Dataset: Breast Cancer Wisconsin dataset (built into scikit-learn)
         -> Predict whether a tumor is Malignant (0) or Benign (1)
Models : Logistic Regression (linear model), Decision Tree, Random Forest
Steps  : Train/Test split -> Train -> Evaluate -> Confusion Matrix -> ROC Curve

Note: The task lists "Linear Regression" as an algorithm. Linear Regression
predicts continuous numbers, so it can't directly produce a confusion
matrix / ROC curve (those need a classifier). Since the task also asks for
confusion matrices and ROC curves, this project uses Logistic Regression
as the "linear model" for classification, and includes a small separate
Linear Regression demo at the end for a continuous target, so both bases
are covered.
==========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, mean_squared_error, r2_score
)

RANDOM_STATE = 42

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Class distribution:\n", pd.Series(y).value_counts())

# ----------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ----------------------------------------------------------
# 3. TRAIN MODELS
# ----------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
roc_data = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("/home/claude/model_comparison.csv", index=False)

# ----------------------------------------------------------
# 4. CONFUSION MATRICES (one figure, 3 subplots)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, model) in zip(axes, fitted_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrices.png", dpi=150)
plt.close()

# ----------------------------------------------------------
# 5. ROC CURVES (all models on one plot)
# ----------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/roc_curves.png", dpi=150)
plt.close()

# ----------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest) - bonus insight
# ----------------------------------------------------------
rf = fitted_models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh", color="teal")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/feature_importance.png", dpi=150)
plt.close()

# ----------------------------------------------------------
# 7. BONUS: LINEAR REGRESSION ON A CONTINUOUS TARGET
#    (Diabetes dataset - predicting disease progression score,
#     a true regression example)
# ----------------------------------------------------------
diabetes = load_diabetes()
Xh = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
yh = diabetes.target  # disease progression score (continuous)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    Xh, yh, test_size=0.2, random_state=RANDOM_STATE
)
lin_reg = LinearRegression()
lin_reg.fit(Xh_train, yh_train)
yh_pred = lin_reg.predict(Xh_test)

rmse = np.sqrt(mean_squared_error(yh_test, yh_pred))
r2 = r2_score(yh_test, yh_pred)

plt.figure(figsize=(6, 6))
plt.scatter(yh_test, yh_pred, alpha=0.5, s=20, color="darkorange")
plt.plot([yh_test.min(), yh_test.max()], [yh_test.min(), yh_test.max()], "r--")
plt.xlabel("Actual Disease Progression Score")
plt.ylabel("Predicted Disease Progression Score")
plt.title(f"Linear Regression: Actual vs Predicted\nRMSE={rmse:.3f}, R2={r2:.3f}")
plt.tight_layout()
plt.savefig("/home/claude/linear_regression_actual_vs_pred.png", dpi=150)
plt.close()

with open("/home/claude/summary_report.txt", "w") as f:
    f.write("PREDICTIVE MODELING PROJECT - SUMMARY REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write("PART 1: CLASSIFICATION (Breast Cancer Dataset)\n")
    f.write("-" * 50 + "\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    best = results_df.iloc[0]
    f.write(f"Best performing model: {best['Model']} (Accuracy={best['Accuracy']:.3f}, AUC={best['ROC-AUC']:.3f})\n\n")
    f.write("PART 2: REGRESSION (Diabetes Dataset)\n")
    f.write("-" * 50 + "\n")
    f.write(f"Linear Regression -> RMSE: {rmse:.3f}, R2 Score: {r2:.3f}\n")

print("\nAll files generated successfully.")
