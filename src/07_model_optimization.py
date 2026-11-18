
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Load encoded data
data = joblib.load(BASE_DIR / "models" / "encoded_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Base Random Forest
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Hyperparameter search space
param_grid = {
    "n_estimators": [200, 300, 500, 700],
    "max_depth": [5, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"]
}

# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Randomized Search
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Starting Hyperparameter Tuning...")

search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross Validation F1:")
print(round(search.best_score_, 4))

# Final evaluation on test data
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n========== OPTIMIZED MODEL ==========")

print("Accuracy:", round(
    accuracy_score(y_test, y_pred), 4
))

print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))

print("F1 Score:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, y_prob), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Save optimized model
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(
    best_model,
    models_dir / "optimized_model.pkl"
)

print("\nOptimized Model Saved Successfully!")
