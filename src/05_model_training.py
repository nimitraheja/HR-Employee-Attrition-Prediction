
import joblib
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Load encoded data
data = joblib.load(BASE_DIR / "models" / "encoded_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Convert sparse matrices to dense for Gradient Boosting
if hasattr(X_train, "toarray"):
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
else:
    X_train_dense = X_train
    X_test_dense = X_test

# Define Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
best_model = None
best_f1 = -1
best_name = ""

# Train and Evaluate
for name, model in models.items():

    print("\nTraining:", name)

    if name == "Gradient Boosting":
        model.fit(X_train_dense, y_train)
        predictions = model.predict(X_test_dense)
        probabilities = model.predict_proba(X_test_dense)[:, 1]
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    # Select best model based on F1 Score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Compare Results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(results_df.sort_values(
    by="F1 Score", ascending=False
).to_string(index=False))

# Save Best Model
models_dir = BASE_DIR / "models"

joblib.dump(
    best_model,
    models_dir / "best_model.pkl"
)

# Save results
results_df.to_csv(
    models_dir / "model_comparison.csv",
    index=False
)

print("\nBest Model:", best_name)
print("Best F1 Score:", round(best_f1, 4))
print("\nModel Training Completed Successfully!")
