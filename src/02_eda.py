
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load Dataset
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "dataset" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

# Basic Information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

# Attrition Distribution
print("\nAttrition Count:")
print(df["Attrition"].value_counts())

# Set plot style
sns.set_style("whitegrid")

# 1. Attrition Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.show()

# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Employee Age Distribution")
plt.show()

# 3. Attrition vs Overtime
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Attrition vs Overtime")
plt.show()

# 4. Monthly Income vs Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income vs Attrition")
plt.show()

# 5. Attrition vs Job Satisfaction
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="JobSatisfaction", hue="Attrition")
plt.title("Attrition vs Job Satisfaction")
plt.show()

# 6. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(14, 10))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")
