
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Load Processed Data
def load_data():
    """Load transformed features and target variable"""

    # Load transformed features (with proper column names)
    X = pd.read_csv("data/X_processed.csv")


    # Classification Target

    # Select one-hot encoded emi_eligibility columns
    y_class_cols = [col for col in X.columns if "emi_eligibility" in col]
    y_class = X[y_class_cols]
    

    # Regression Target

    y_reg = X["num__total_emi_paid"]

    # Drop target columns from features
    X = X.drop(columns=y_class_cols + ["num__total_emi_paid"])

    return X, y_class, y_reg



# Train + evaluate classification models

def train_classification_models(X_train, X_val, y_train, y_val):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=300),
        "XGBoost Classifier": XGBClassifier(
            eval_metric='logloss',
            learning_rate=0.05,
            n_estimators=300,
            max_depth=5
        )
    }

    results = []
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\n🔵 Training: {name}")
        model.fit(X_train, y_train.values.argmax(axis=1))  # convert one-hot to label index

        preds = model.predict(X_val)
        # Map predictions back to one-hot labels
        f1 = f1_score(y_val.values.argmax(axis=1), preds, average="weighted")

        results.append([name, f1])
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_classification_model.pkl")
    print("\n✅ Best Classification Model Saved!")

    return pd.DataFrame(results, columns=["Model", "Weighted F1-Score"])



#  Train + evaluate regression models

def train_regression_models(X_train, X_val, y_train, y_val):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300),
        "XGBoost Regressor": XGBRegressor(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=6
        )
    }

    results = []
    best_model = None
    best_rmse = 10**9  # very high number

    for name, model in models.items():
        print(f"\n🟣 Training: {name}")
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)

        results.append([name, rmse, r2])
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_regression_model.pkl")
    print("\n✅ Best Regression Model Saved!")

    return pd.DataFrame(results, columns=["Model", "RMSE", "R2 Score"])



#  Main Execution Function

def train_all_models():

    print("📌 Loading processed feature files…")
    X, y_class, y_reg = load_data()

    print("📌 Performing train-validation-test split…")
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=42
    )

    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.50, random_state=42
    )


    # Classification Training

    print("\n🔷 Training Classification Models…")
    class_results = train_classification_models(
        X_train, X_val, y_class_train, y_class_val
    )


    # Regression Training

    print("\n🟪 Training Regression Models…")
    reg_results = train_regression_models(
        X_train, X_val, y_reg_train, y_reg_val
    )

    # Save metrics to CSV
    results_file = "models/model_performance_metrics.csv"
    final_results = pd.concat([class_results, reg_results], axis=0)
    final_results.to_csv(results_file, index=False)

    print("\n📁 Metrics saved to:", results_file)
    print("🎉 Training complete! All models processed.")


if __name__ == "__main__":
    train_all_models()
