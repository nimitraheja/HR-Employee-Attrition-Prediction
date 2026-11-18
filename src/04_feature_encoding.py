
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Load cleaned dataset
file_path = BASE_DIR / "dataset" / "cleaned_hr_attrition.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# Separate Features and Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Fit only on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_encoded = preprocessor.transform(X_test)

# Create models directory
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

# Save preprocessor
joblib.dump(
    preprocessor,
    models_dir / "preprocessor.pkl"
)

# Save transformed data
joblib.dump(
    {
        "X_train": X_train_encoded,
        "X_test": X_test_encoded,
        "y_train": y_train,
        "y_test": y_test
    },
    models_dir / "encoded_data.pkl"
)

print("\nEncoded Training Shape:", X_train_encoded.shape)
print("Encoded Testing Shape:", X_test_encoded.shape)

print("\nFeature Encoding Completed Successfully!")
