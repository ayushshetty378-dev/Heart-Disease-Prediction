import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
(BASE / "models").mkdir(exist_ok=True)

TARGET_CANDIDATES = ["target", "num", "HeartDisease", "heart_disease", "output", "condition"]
DROP_CANDIDATES = ["id", "ID", "Id", "dataset", "Unnamed: 0"]


def find_csv():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for folder in (BASE / "data", BASE):
        for f in sorted(folder.glob("*.csv")):
            return f
    sys.exit("No CSV found. Download the Kaggle 'Heart Disease UCI' dataset and put the "
             "CSV in the 'data' folder (or pass its path as an argument).")


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean(df):
    report = {"rows_raw": int(len(df))}
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        sys.exit(f"Could not find a target column. Columns are: {list(df.columns)}")
    df = df.drop(columns=[c for c in DROP_CANDIDATES if c in df.columns])

    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = df[target].astype(str).str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    df = df.dropna(subset=[target])
    df["target"] = (df[target].astype(float) > 0).astype(int)
    if target != "target":
        df = df.drop(columns=[target])

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(str)

    report["missing_values"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["rows_clean"] = int(len(df))
    report["class_counts"] = {str(k): int(v) for k, v in df["target"].value_counts().items()}
    print("Cleaning report:", report)
    return df, report


def split_columns(X):
    """Numeric = real numbers. Everything else (text, bool, True/False with gaps) = categorical."""
    cat = [c for c in X.columns
           if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c])]
    for c in X.columns:
        if c not in cat and X[c].dtype.kind in "iu" and X[c].nunique() <= 5:
            cat.append(c)
    num = [c for c in X.columns if c not in cat]
    return num, cat


def stringify_categoricals(X, cat):
    """Make every categorical value a plain string (keeping real gaps as NaN) so one-hot works."""
    X = X.copy()
    for c in cat:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X


def build_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

def eda(df, num):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.countplot(x="target", data=df, ax=ax, palette=["#4C9F70", "#D1495B"])
    ax.set_xticklabels(["No disease", "Disease"])
    ax.set_title("Class balance"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG / "class_balance.png", dpi=150); plt.close(fig)

    corr = df[num + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation (numeric features)")
    fig.tight_layout(); fig.savefig(FIG / "correlation.png", dpi=150); plt.close(fig)


def main():
    df = load_data(find_csv())
    df, report = clean(df)
    X, y = df.drop(columns="target"), df["target"]
    num, cat = split_columns(X)
    X = stringify_categoricals(X, cat)
    print(f"Numeric: {num}\nCategorical: {cat}")
    eda(df, num)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pre = build_preprocessor(num, cat)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": (
            Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"]}),
        "Decision Tree": (
            Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [2, 3, 4, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10],
             "clf__criterion": ["gini", "entropy"]}),
    }

    results, fitted = {}, {}
    for name, (pipe, grid) in candidates.items():
        base_cv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        pred = best.predict(X_te)
        proba = best.predict_proba(X_te)[:, 1]
        results[name] = {
            "cv_accuracy_mean": float(base_cv.mean()), "cv_accuracy_std": float(base_cv.std()),
            "best_params": {k: (None if v is None else v) for k, v in gs.best_params_.items()},
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        fitted[name] = (best, pred, proba)
        print(f"\n{name}: {json.dumps(results[name], indent=2)}")

    for name, (_, pred, _) in fitted.items():
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ConfusionMatrixDisplay.from_predictions(
            y_te, pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax)
        ax.set_title(f"Confusion matrix - {name}")
        fig.tight_layout()
        fig.savefig(FIG / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (_, _, proba) in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_curve.png", dpi=150); plt.close(fig)

    winner = max(results, key=lambda n: (round(results[n]["roc_auc"], 4), results[n]["f1"]))
    joblib.dump(fitted[winner][0], BASE / "models" / "heart_model.joblib")
    results_out = {"cleaning": report, "numeric_features": num, "categorical_features": cat,
                   "models": results, "selected_model": winner,
                   "test_size": int(len(y_te)), "train_size": int(len(y_tr))}
    (BASE / "results.json").write_text(json.dumps(results_out, indent=2))
    pd.DataFrame(results).T[["cv_accuracy_mean", "accuracy", "precision", "recall", "f1", "roc_auc"]] \
        .round(4).to_csv(BASE / "metrics.csv")
    print(f"\nSelected model: {winner} -> models/heart_model.joblib")

    model = joblib.load(BASE / "models" / "heart_model.joblib")
    sample = X_te.iloc[:3]
    print("\nPredictions on 3 unseen patients:", model.predict(sample).tolist(),
          "| actual:", y_te.iloc[:3].tolist())


if __name__ == "__main__":
    main()
