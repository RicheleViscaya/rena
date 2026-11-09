"""Rebuild the midterm Decision Tree as ONE saved sklearn pipeline (preprocessing + model).

Run once (not on every page refresh):
    python train_model.py                      # uses data/facility_reservations.csv if present,
                                               # otherwise data/sample_facility_reservations.csv
    python train_model.py path/to/your.csv     # or give your own midterm CSV

Settings copied from the midterm notebook (Task 7):
  predictors  : Lead_Time, Participants, Facility_Type, User_Type, Lead_Time_Category
  target      : Reservation_Outcome  (Completed / Not Completed)
  split       : 80/20, random_state=42, stratified
  model       : DecisionTreeClassifier(max_depth=4, random_state=42)
  encoding    : one-hot with first category dropped (same as pd.get_dummies(drop_first=True))
"""
import json, os, sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from modules.data_processing import clean_dataset

NUMERIC = ["Lead_Time", "Participants"]
CATEGORICAL = ["Facility_Type", "User_Type", "Lead_Time_Category"]
FEATURES = NUMERIC + CATEGORICAL          # same order as the midterm notebook
TARGET = "Reservation_Outcome"
POSITIVE = "Not Completed"
MODEL_VERSION = "1.0"


def build_pipeline():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])
    return Pipeline([("preprocess", pre),
                     ("model", DecisionTreeClassifier(max_depth=4, random_state=42))])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "data/facility_reservations.csv" if os.path.exists("data/facility_reservations.csv")
        else "data/sample_facility_reservations.csv")
    print("Training data:", src)
    raw = pd.read_csv(src)
    df, log = clean_dataset(raw)
    print("Cleaning:", log)

    X, y = df[FEATURES].copy(), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, pred, labels=labels)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)

    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "training_file": os.path.basename(src),
        "is_sample_data": os.path.basename(src).startswith("sample_"),
        "algorithm": "DecisionTreeClassifier(max_depth=4, random_state=42)",
        "target": TARGET, "positive_label": POSITIVE, "classes": list(pipe.classes_),
        "features": FEATURES, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in ["Facility_Type", "User_Type"]},
        "ranges": {"Lead_Time": [0, 365], "Participants": [1, 500]},
        "cleaning_log": log,
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "metrics": {
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, pos_label=POSITIVE, zero_division=0)),
            "recall": float(recall_score(y_te, pred, pos_label=POSITIVE, zero_division=0)),
            "f1": float(f1_score(y_te, pred, pos_label=POSITIVE, zero_division=0)),
        },
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "feature_importances": {k: float(v) for k, v in imp.items()},
        "majority_baseline": float(y_te.value_counts(normalize=True).max()),
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/trained_model.joblib")
    with open("models/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(16, 8))
    plot_tree(pipe.named_steps["model"], feature_names=names,
              class_names=[str(c) for c in pipe.classes_], filled=True, rounded=True, fontsize=9)
    plt.title("Decision Tree structure (max_depth=4)")
    plt.tight_layout(); plt.savefig("assets/decision_tree.png", dpi=110); plt.close()

    print("Metrics:", {k: round(v, 4) for k, v in meta["metrics"].items()})
    print("Confusion matrix", labels, cm.tolist())
    print("Top predictors:", imp.head(3).round(4).to_dict())
    print("Saved models/trained_model.joblib, models/model_metadata.json, assets/decision_tree.png")


if __name__ == "__main__":
    main()
