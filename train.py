"""Train the churn model and save ONE file (churn_model.pkl) that the app loads.

Run:  python train.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = "churn_model.pkl"


def build_and_save(verbose=True):
    df = pd.read_csv("customer_churn_dataset.csv")

    # TotalCharges has 11 blank strings -> convert to a real number.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.drop(columns=["customerID", "gender"])  # same drops as the notebook

    X, y = df.drop(columns="Churn"), df["Churn"]
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [c for c in X.columns if c not in numeric]

    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    # Best hyper-parameters from the notebook's RandomizedSearchCV.
    # class_weight replaces SMOTE (handles the 73/27 imbalance inside one pipeline).
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
        max_features="log2", class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    model = Pipeline([("prep", pre), ("rf", rf)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)

    pred, prob = model.predict(X_te), model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1 score": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, prob),
    }
    if verbose:
        for k, v in metrics.items():
            print(f"{k:<10}: {v:.3f}")

    model.fit(X, y)  # final model trained on all data
    bundle = {"model": model, "columns": list(X.columns), "metrics": metrics}
    joblib.dump(bundle, MODEL_PATH)  # no compression = most robust file
    if verbose:
        print("Saved", MODEL_PATH)
    return bundle


def load_or_train():
    """Load the saved model; if the file is missing/corrupt/incompatible, retrain it."""
    try:
        bundle = joblib.load(MODEL_PATH)
        if "metrics" not in bundle:
            raise ValueError("old model file")
        return bundle
    except Exception:
        return build_and_save(verbose=False)


if __name__ == "__main__":
    build_and_save()
