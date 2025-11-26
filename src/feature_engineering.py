import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_clean_data(input_path="D:\EMI_Financial_Risk_Assessment\data\cleaned_data.csv"):
    """Load cleaned dataset"""
    df = pd.read_csv(input_path)
    return df

def create_financial_features(df):
    """
    Create new engineered ratios:
    - Debt-to-Income (DTI)
    - Expense-to-Income Ratio
    - EMI Affordability Score
    """

    # Map your dataset columns
    df["monthly_income"] = df["monthly_salary"].replace(0, np.nan)

    # Sum all expense-related columns
    expense_cols = [
        "monthly_rent",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses"
    ]
    df["monthly_expenses"] = df[expense_cols].sum(axis=1).replace(0, np.nan)

    # Total EMI paid
    df["total_emi_paid"] = df["current_emi_amount"].replace(0, np.nan)

    # Debt-to-Income Ratio
    df["dti_ratio"] = df["total_emi_paid"] / df["monthly_income"]

    # Expense-to-Income Ratio
    df["expense_income_ratio"] = df["monthly_expenses"] / df["monthly_income"]

    # EMI Affordability Score
    df["emi_affordability"] = df["monthly_income"] - df["monthly_expenses"] - df["total_emi_paid"]

    # Replace infinities or invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df



from sklearn.impute import SimpleImputer

def build_feature_pipeline(df):
    """Create encoding + scaling pipeline with imputation"""

    # Identify columns
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if present
    if "emi_status" in numeric_features:
        numeric_features.remove("emi_status")
    if "emi_status" in categorical_features:
        categorical_features.remove("emi_status")

    # Preprocess numeric columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # fill NaN with median
        ("scaler", StandardScaler())
    ])

    # Preprocess categorical columns
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaN with mode
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return full_pipeline, numeric_features, categorical_features



def process_features():
    """Main execution function"""

    print("📌 Loading cleaned data…")
    df = load_clean_data()

    print("📌 Creating financial features…")
    df = create_financial_features(df)

    print("📌 Building feature pipeline…")
    pipeline, num_cols, cat_cols = build_feature_pipeline(df)

    print("📌 Fitting transformation pipeline…")
    X = df.drop(columns=["emi_status"], errors="ignore")
    pipeline.fit(X)

    print("📌 Transforming dataset…")
    X_processed = pipeline.transform(X)

    # Save transformed data
    output_csv = "data/X_processed.csv"
    # Save transformed data with column names
    if hasattr(X_processed, "toarray"):
     X_processed_df = pd.DataFrame(
        X_processed.toarray(), 
        columns=pipeline.get_feature_names_out()
    )
    else:
     X_processed_df = pd.DataFrame(
        X_processed, 
        columns=pipeline.get_feature_names_out()
    )

    X_processed_df.to_csv(output_csv, index=False)


    # Save pipeline
    joblib.dump(pipeline, "models/feature_pipeline.pkl")

    print("\n✅ Feature engineering complete!")
    print(f"Processed dataset saved to: {output_csv}")
    print(f"Pipeline saved to: models/feature_pipeline.pkl\n")


if __name__ == "__main__":
    process_features()
