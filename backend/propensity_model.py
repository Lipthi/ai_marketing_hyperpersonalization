# backend/propensity_model.py

from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
import os, sys

# --- Hybrid Import ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "propensity_model.pkl")


# -------------------------
# Feature engineering utils
# -------------------------
LOYALTY_TO_NUM = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
INCOME_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce numeric features suitable for a simple model.
    Uses columns your synthetic generator creates.
    """
    df2 = df.copy()

    # Defensive defaults
    for col in [
        "engagement_score",
        "churn",
        "total_spent",
        "avg_order_value",
        "last_purchase_days",
        "loyalty_status",
        "income",
        "recent_transactions",
    ]:
        if col not in df2.columns:
            df2[col] = 0 

    # numeric features
    df2["engagement_score"] = pd.to_numeric(df2["engagement_score"], errors="coerce").fillna(0.0)
    df2["churn"] = pd.to_numeric(df2["churn"], errors="coerce").fillna(0).astype(int)
    df2["total_spent"] = pd.to_numeric(df2["total_spent"], errors="coerce").fillna(0.0)
    df2["avg_order_value"] = pd.to_numeric(df2["avg_order_value"], errors="coerce").fillna(0.0)
    df2["last_purchase_days"] = pd.to_numeric(df2["last_purchase_days"], errors="coerce").fillna(999)

    # derived features
    # number of recent transactions
    def _num_tx(x):
        try:
            return len(x) if isinstance(x, list) else int(x)
        except Exception:
            return 0
    df2["num_transactions"] = df2["recent_transactions"].apply(_num_tx)

    # map categorical into numeric ordinals
    df2["loyalty_num"] = df2["loyalty_status"].map(LOYALTY_TO_NUM).fillna(0)
    df2["income_num"] = df2["income"].map(INCOME_TO_NUM).fillna(1)

    # scale/normalize total_spent
    df2["total_spent_k"] = df2["total_spent"] / 1000.0

    # target: if churn==1 treat lower propensity; if you have historical 'converted' flag use that instead.
    # Here we create a synthetic target: (engagement > 0.6 and churn==0) => likely convert (1)
    if "converted" not in df2.columns:
        df2["_target"] = ((df2["engagement_score"] >= 0.6) & (df2["churn"] == 0)).astype(int)
    else:
        df2["_target"] = pd.to_numeric(df2["converted"], errors="coerce").fillna(0).astype(int)

    features = [
        "engagement_score",
        "churn",
        "last_purchase_days",
        "num_transactions",
        "loyalty_num",
        "income_num",
        "avg_order_value",
        "total_spent_k",
    ]
    return df2[features + ["_target"]]


# -------------------------
# Model train / save / load
# -------------------------
def train_and_save_model(customers_csv: str = "../data/processed/customers.csv", model_path: str = MODEL_PATH, test_size: float = 0.2, random_state: int = 42):
    """
    Train a propensity model and save it. If training data contains only one class,
    fall back to a DummyClassifier to avoid training errors and to ensure predict_proba
    has a consistent API.
    Returns the trained estimator.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(customers_csv)
    Xy = _prepare_dataframe(df)
    X = Xy.drop(columns=["_target"])
    y = Xy["_target"]

    # If y has only one class, do not stratify split
    stratify_param = y if len(set(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None)

    # Standard pipeline (imputer -> scaler -> classifier)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
     
    if len(set(y_train)) < 2:
        print("⚠️ Only one class found in training data. Using DummyClassifier instead of LogisticRegression.")
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="uniform")
        dummy.fit(X_train, y_train)
        estimator = dummy
    else:
        pipeline.fit(X_train, y_train)
        estimator = pipeline

    # eval
    acc, auc = None, None
    try:
        if hasattr(pipeline, "predict"):
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        # Attempt AUC only when possible and when there are at least 2 classes in y_test
        if hasattr(estimator, "predict_proba") and len(set(y_test)) > 1:
            proba = estimator.predict_proba(X_test)
            # safe extraction of positive-class probability (see predict_batch for logic)
            if proba.shape[1] == 1:
                # only one class present in estimator; AUC not meaningful
                auc = None
            else:
                classes = getattr(estimator, "classes_", None)
                if classes is not None and 1 in classes:
                    idx = list(classes).index(1)
                else:
                    idx = -1
                y_proba = proba[:, idx]
                auc = roc_auc_score(y_test, y_proba)
    except Exception:
        pass


    # Persist the estimator
    try:
        joblib.dump(estimator, model_path)
        print(f"✅ Saved propensity model -> {model_path}")
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")

    if acc is not None:
        print(f"Accuracy: {acc:.3f}" + (f", AUC: {auc:.3f}" if auc is not None else ""))

    return estimator

def load_model(model_path: str = MODEL_PATH):
    """Load a saved model if present, else return None."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    return None  


# -------------------------
# Prediction helpers
# -------------------------

def _safe_extract_positive_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    """
    Return a (n_samples,) array of probability for the positive class (label 1)
    using a variety of fallbacks to avoid indexing errors.


    Rules used:
    - If estimator.predict_proba exists, use it.
    - If predict_proba returns 2+ columns, pick the column corresponding to class '1' if present,
    otherwise pick the last column.
    - If predict_proba returns a single column (single-class model), infer whether that column
    represents class 1 (then return it) or class 0 (then return zeros).
    - Else if estimator.decision_function exists, apply a logistic transform to produce pseudo-probabilities.
    - Else return a neutral 0.5 probability for all samples.
    """
    n = len(X)


    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
    # proba shape: (n_samples, n_classes)
        if proba.ndim != 2:
    # Unexpected shape; fallback to neutral
            return np.full(n, 0.5)

        if proba.shape[1] == 1:
            # Single-class model. Check which class it is.
            classes = getattr(estimator, "classes_", None)
            if classes is not None and len(classes) == 1 and classes[0] == 1:
                # Model only knows class 1 -> probability of positive is the single column
                return proba[:, 0]
            else:
                # Model only knows class 0 -> probability of positive is zero
                return np.zeros(n)
        else:
            classes = getattr(estimator, "classes_", None)
            if classes is not None and 1 in classes:
                idx = list(classes).index(1)
                return proba[:, idx]
            else:
                # no explicit class=1, use last column as heuristic
                return proba[:, -1]


    # Fallback: decision_function -> sigmoid
    if hasattr(estimator, "decision_function"):
        try:
            scores = estimator.decision_function(X)
            # If scores is 1D or 2D, normalize accordingly
            if scores.ndim == 1:
                return 1.0 / (1.0 + np.exp(-scores))
            else:
            # multi-class decision_function: reduce to max score heuristic
                scores = np.max(scores, axis=1)
                return 1.0 / (1.0 + np.exp(-scores))
        except Exception:
            pass


# Last resort: neutral probability
    return np.full(n, 0.5)

def predict_batch(customers: List[Dict[str, Any]], model=None) -> List[float]:
    """
    Given a list of customer dicts, returns a list of propensity probabilities (0..1).
    """
    if model is None:
        model = load_model()
        if model is None:
            # If no model on disk, train a quick one from CSV automatically (convenience for experiments)
            print("No saved model found — training a quick model on ../data/processed/customers.csv")
            model = train_and_save_model()

    df = pd.DataFrame(customers)
    prepared = _prepare_dataframe(df)
    X = prepared.drop(columns=["_target"])
    probs = _safe_extract_positive_proba(model, X)
    # Ensure numeric list output
    return [float(x) for x in np.clip(probs, 0.0, 1.0).tolist()]


def predict_one(customer: Dict[str, Any], model=None) -> float:
    return float(predict_batch([customer], model=model)[0])


# -------------------------
# Quick CLI for training
# -------------------------
if __name__ == "__main__":
    train_and_save_model()
