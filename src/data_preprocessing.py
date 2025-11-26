import pandas as pd
import numpy as np
import re
import os


# 1. Load Dataset

def load_dataset(file_path):
    print("📌 Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"✔ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df



# 2. Clean Numeric Columns (Fix values like '58.0.0', '40..', '5a3')

def clean_numeric_columns(df):
    print("📌 Cleaning numeric columns...")

    for col in df.columns:
        # Skip non-object columns
        if df[col].dtype != 'object':
            continue

        # Detect if column SHOULD be numeric (contains digits)
        sample_values = df[col].astype(str).head(50)
        if sample_values.str.contains(r'\d').sum() > 25:
            print(f"➡ Cleaning numeric-like column: {col}")

            # Remove invalid characters
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9.-]', '', regex=True)   # keep only numbers, dot & minus
                .str.replace(r'\.{2,}', '.', regex=True)    # replace multiple dots
                .str.replace(r'-{2,}', '-', regex=True)     # replace multiple minus
                .str.replace(r'^\.','', regex=True)         # remove starting dot
            )

            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# 3. Clean bank_balance (remove commas, ₹ symbols, spaces)

def clean_bank_balance(df):
    if "bank_balance" in df.columns:
        print("📌 Cleaning 'bank_balance' column...")

        df["bank_balance"] = (
            df["bank_balance"]
            .astype(str)
            .str.replace(r'[₹, ]', '', regex=True)
        )
        df["bank_balance"] = pd.to_numeric(df["bank_balance"], errors='coerce')

    return df



# 4. Clean existing_loans (convert to count of loans)

def clean_existing_loans(df):
    if "existing_loans" in df.columns:
        print("📌 Cleaning 'existing_loans' column...")

        df["existing_loans"] = df["existing_loans"].replace("None", "[]")

        # Count number of loans from string like "[12000, 5000]"
        df["existing_loans_count"] = df["existing_loans"].apply(
            lambda x: len(re.findall(r'\d+', str(x)))
        )

    return df



# 5. Handle Missing Values

def handle_missing_values(df):
    print("📌 Handling missing values...")

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df



# 6. Remove Duplicates

def remove_duplicates(df):
    print("📌 Removing duplicate rows...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"✔ Removed {before - after} duplicates")
    return df


# 7. Outlier Handling using IQR
def handle_outliers(df):
    print("📌 Handling outliers (IQR method)...")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    return df



# 8. Save Cleaned Dataset
def save_cleaned_data(df, output_path="cleaned_data.csv"):
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved as: {output_path}")


# MAIN FUNCTION
def main():
    input_file = "D:\EMI_Financial_Risk_Assessment\data\emi_prediction_dataset.csv"          # Change if needed
    output_file = "cleaned_data.csv"

    df = load_dataset(input_file)
    df = clean_numeric_columns(df)
    df = clean_bank_balance(df)
    df = clean_existing_loans(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)

    save_cleaned_data(df, output_file)
    print("🎉 Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
