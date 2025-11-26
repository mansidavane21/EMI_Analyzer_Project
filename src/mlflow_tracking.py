import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

sns.set()


# Utility: robust file loader

def find_cleaned_dataset():
    candidates = [
        "data/cleaned_data.csv",
        "data/cleaned_dataset.csv",
        "cleaned_data.csv",
        "data/cleaned_emi_dataset.csv",
        "data/cleaned_emi_dataset.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cleaned dataset not found. Expected one of: " + ", ".join(candidates))

def find_processed_features():
    candidates = [
        "data/X_processed.csv",
        "data/X_processed.csv",
        "X_processed.csv",
        "data/processed_features.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Processed features file not found. Expected X_processed.csv in data/")

def find_models(pattern="models/*.pkl"):
    return glob.glob(pattern)


# Plot helpers

def plot_confusion_matrix(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_regression_scatter(y_true, y_pred, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_feature_importance(model, feature_names, outpath):
    # Many models expose feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[-30:]  # top 30
        names = np.array(feature_names)[idx]
        values = imp[idx]

        plt.figure(figsize=(8,6))
        plt.barh(names, values)
        plt.title("Feature Importance (top features)")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return True
    return False


# Main MLflow logging function

def main():
    os.makedirs("mlflow_artifacts", exist_ok=True)
    cleaned_path = find_cleaned_dataset()
    features_path = find_processed_features()

    print("Loading cleaned data:", cleaned_path)
    df = pd.read_csv(cleaned_path)

    print("Loading processed features:", features_path)
    X = pd.read_csv(features_path)

    # Attempt to detect target names (robust)
    # Common names seen in conversation: 'emi_eligibility' (classification), 'max_monthly_emi' (regression)
    if "emi_eligibility" in df.columns:
        y_class = df["emi_eligibility"]
    elif "emi_status" in df.columns:
        y_class = df["emi_status"]
    elif "emi" in df.columns:
        y_class = df["emi"]
    else:
        # fallback: try to find an object/categorical column with small unique count
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        y_class = None
        for c in cat_cols:
            if df[c].nunique() <= 10:
                y_class = df[c]
                print(f"Using {c} as classification target.")
                break

    # Regression target candidates
    if "max_monthly_emi" in df.columns:
        y_reg = df["max_monthly_emi"]
    elif "total_emi_paid" in df.columns:
        y_reg = df["total_emi_paid"]
    elif "requested_amount" in df.columns:
        y_reg = df["requested_amount"]
    else:
        y_reg = None

    # If classification target is missing, we can't evaluate classifiers
    if y_class is None and y_reg is None:
        raise ValueError("No recognizable target columns for classification or regression found in cleaned dataset.")

    # Align shapes: make sure X rows == df rows
    if len(X) != len(df):
        print("Warning: feature rows and cleaned dataset rows mismatch. Aligning by min length.")
        min_len = min(len(X), len(df))
        X = X.iloc[:min_len].reset_index(drop=True)
        df = df.iloc[:min_len].reset_index(drop=True)
        if y_class is not None:
            y_class = y_class.iloc[:min_len]
        if y_reg is not None:
            y_reg = y_reg.iloc[:min_len]

    # Create a test split for evaluation (20% test)
    test_size = 0.2
    random_state = 42
    if y_class is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=test_size, random_state=random_state, stratify=y_class)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=test_size, random_state=random_state)

    # find models
    model_files = find_models("models/*.pkl")
    if not model_files:
        raise FileNotFoundError("No model .pkl files found under models/. Please run training to produce models.")

    print("Found model files:", model_files)

    # Start MLflow experiment
    mlflow.set_experiment("EMI_Prediction_Experiment")

    # We'll evaluate each model file and log a separate run
    eval_rows = []
    for model_fp in model_files:
        model_name = os.path.splitext(os.path.basename(model_fp))[0]
        print(f"\nEvaluating model: {model_name}")

        # load model
        try:
            model = joblib.load(model_fp)
        except Exception as e:
            print(f"Could not load {model_fp}: {e}")
            continue

        with mlflow.start_run(run_name=model_name):
            # Log some basic params (if sklearn estimator has get_params)
            try:
                params = model.get_params()
                # log some top params
                for k, v in list(params.items())[:30]:
                    mlflow.log_param(str(k), str(v))
            except Exception:
                pass

            # Determine if classifier or regressor by predict_proba existence or type
            is_classifier = hasattr(model, "predict_proba") or "Classifier" in type(model).__name__

            # Make predictions on test set
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                print("Model predict failed (maybe pipeline expects different input). Error:", e)
                continue

            # Classification evaluation
            if is_classifier and (y_class is not None):
                # Ensure y_test is categorical strings or numbers consistent with predictions
                # compute metrics
                try:
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                except Exception as e:
                    print("Error computing classification metrics:", e)
                    acc = prec = rec = f1 = None

                # try ROC-AUC if we have binary labels and predict_proba
                roc_auc = None
                if hasattr(model, "predict_proba"):
                    try:
                        # if labels are strings, convert to codes for AUC when binary
                        if len(np.unique(y_test)) == 2:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            # need numeric binary labels
                            if y_test.dtype == "object":
                                # map labels to 0/1 in the same order as unique()
                                labels = list(np.unique(y_test))
                                y_test_bin = y_test.map({labels[0]:0, labels[1]:1})
                            else:
                                y_test_bin = y_test
                            roc_auc = roc_auc_score(y_test_bin, y_proba)
                    except Exception:
                        roc_auc = None

                # log metrics
                if acc is not None:
                    mlflow.log_metric("accuracy", float(acc))
                if prec is not None:
                    mlflow.log_metric("precision_macro", float(prec))
                if rec is not None:
                    mlflow.log_metric("recall_macro", float(rec))
                if f1 is not None:
                    mlflow.log_metric("f1_macro", float(f1))
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", float(roc_auc))

                # save confusion matrix artifact
                cm_path = f"mlflow_artifacts/{model_name}_confusion_matrix.png"
                plot_confusion_matrix(y_test, y_pred, cm_path)
                mlflow.log_artifact(cm_path)

                # try feature importance
                fi_path = f"mlflow_artifacts/{model_name}_feature_importance.png"
                feature_names = X.columns.tolist()
                if plot_feature_importance(model, feature_names, fi_path):
                    mlflow.log_artifact(fi_path)

                eval_rows.append({
                    "model_file": model_fp,
                    "model_name": model_name,
                    "type": "classification",
                    "accuracy": acc,
                    "precision_macro": prec,
                    "recall_macro": rec,
                    "f1_macro": f1,
                    "roc_auc": roc_auc
                })

                # log the model artifact to mlflow
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception as e:
                    print("Could not log sklearn model to mlflow:", e)

            # Regression evaluation
            elif (not is_classifier) and (y_reg is not None):
                # If model predicts array-like of shape (n,) or (n,1)
                try:
                    y_pred_reg = np.array(y_pred).reshape(-1)
                    y_true_reg = np.array(y_test).reshape(-1)

                    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
                    mae = mean_absolute_error(y_true_reg, y_pred_reg)
                    r2 = r2_score(y_true_reg, y_pred_reg)
                except Exception as e:
                    print("Error computing regression metrics:", e)
                    rmse = mae = r2 = None

                # log metrics
                if rmse is not None:
                    mlflow.log_metric("rmse", float(rmse))
                if mae is not None:
                    mlflow.log_metric("mae", float(mae))
                if r2 is not None:
                    mlflow.log_metric("r2", float(r2))

                # save scatter plot
                scatter_path = f"mlflow_artifacts/{model_name}_reg_scatter.png"
                plot_regression_scatter(y_true_reg, y_pred_reg, scatter_path)
                mlflow.log_artifact(scatter_path)

                # try feature importance
                fi_path = f"mlflow_artifacts/{model_name}_feature_importance.png"
                feature_names = X.columns.tolist()
                if plot_feature_importance(model, feature_names, fi_path):
                    mlflow.log_artifact(fi_path)

                eval_rows.append({
                    "model_file": model_fp,
                    "model_name": model_name,
                    "type": "regression",
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                })

                # log the model artifact to mlflow
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception as e:
                    print("Could not log sklearn model to mlflow:", e)

            else:
                print("Could not determine model type or no matching target available; skipping metric computation for", model_name)
                continue

            # End run automatically on exiting context

    # Save evaluation summary CSV
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv("mlflow_artifacts/evaluation_summary.csv", index=False)
    print("\nSaved evaluation summary to: mlflow_artifacts/evaluation_summary.csv")
    print("Done. Open MLflow UI with: mlflow ui (then visit http://127.0.0.1:5000 )")


if __name__ == "__main__":
    main()
