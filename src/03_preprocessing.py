
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Load dataset
file_path = BASE_DIR / "dataset" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

print("Original Dataset Shape:", df.shape)

# 1. Check Duplicate Records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# 2. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 3. Check Constant Columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique() == 1
]

print("\nConstant Columns:", constant_columns)

# Remove constant columns
df.drop(columns=constant_columns, inplace=True)

# 4. Remove EmployeeNumber
if "EmployeeNumber" in df.columns:
    df.drop(columns=["EmployeeNumber"], inplace=True)

# 5. Convert Target Variable
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# 6. Check Target Values
print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

# 7. Final Dataset Information
print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Dataset Info:")
df.info()

# 8. Save Cleaned Dataset
output_path = BASE_DIR / "dataset" / "cleaned_hr_attrition.csv"

df.to_csv(output_path, index=False)

print("\nPreprocessing Completed Successfully!")
print("Cleaned dataset saved at:", output_path)
