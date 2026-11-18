
# Import pandas library
import pandas as pd

# Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Check rows and columns
print("\nDataset Shape:")
print(df.shape)

# Check column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
