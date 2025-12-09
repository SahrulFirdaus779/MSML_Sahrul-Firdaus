import argparse
import os
from pathlib import Path

import joblib
import matplotlib

# Use non-interactive backend to avoid Tkinter thread issues on headless/CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    import dagshub

    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False

DEFAULT_TARGET = "Target"
DEFAULT_EXPERIMENT = "data_clustering_tuning"


def load_dataset(path: Path, target_col: str):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def plot_confusion(y_true, y_pred, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_feature_importance(model, feature_names, out_path: Path, top_k: int = 20):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    names = np.array(feature_names)[idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names[::-1], values[::-1], color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def train_with_tuning(
    X,
    y,
    experiment_name: str,
    random_state: int,
    test_size: float,
    out_dir: Path,
    tracking_uri: str | None,
    dagshub_owner: str | None,
    dagshub_repo: str | None,
    use_dagshub_init: bool,
):
    if use_dagshub_init and tracking_uri:
        if not DAGSHUB_AVAILABLE:
            raise ImportError("Paket dagshub belum terpasang. Jalankan pip install dagshub di environment ini.")
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="random_forest_grid_tuning"):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        preds_val = best_model.predict(X_val)
        preds_train = best_model.predict(X_train)

        val_metrics = {
            "val_accuracy": float(accuracy_score(y_val, preds_val)),
            "val_precision_macro": float(precision_score(y_val, preds_val, average="macro", zero_division=0)),
            "val_recall_macro": float(recall_score(y_val, preds_val, average="macro", zero_division=0)),
            "val_f1_macro": float(f1_score(y_val, preds_val, average="macro", zero_division=0)),
        }
        train_metrics = {
            "train_accuracy": float(accuracy_score(y_train, preds_train)),
            "train_precision_macro": float(precision_score(y_train, preds_train, average="macro", zero_division=0)),
            "train_recall_macro": float(recall_score(y_train, preds_train, average="macro", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, preds_train, average="macro", zero_division=0)),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_best_score_f1_macro", float(grid.best_score_))
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(train_metrics)

        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "rf_grid_model.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion(y_val, preds_val, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = plot_feature_importance(best_model, X.columns, out_dir / "feature_importance.png")
        if fi_path:
            mlflow.log_artifact(fi_path)

        metrics_df = pd.DataFrame([val_metrics | train_metrics])
        metrics_csv = out_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        mlflow.log_artifact(metrics_csv)

        print("Best params:")
        for k, v in grid.best_params_.items():
            print(f"  {k}: {v}")
        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with hyperparameter tuning and manual MLflow logging")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data_clustering_preprocessing/train_processed.csv"),
        help="Path to preprocessed train CSV",
    )
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("tuning_artifacts"),
        help="Directory to save model and plots",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (set to DagsHub for remote logging)",
    )
    parser.add_argument(
        "--dagshub-owner",
        type=str,
        default=os.getenv("DAGSHUB_OWNER", "SahrulFirdaus779"),
        help="Owner DagsHub repo (dipakai saat dagshub.init)",
    )
    parser.add_argument(
        "--dagshub-repo",
        type=str,
        default=os.getenv("DAGSHUB_REPO", "Eksperimen_SML_Sahrul-Firdaus"),
        help="Nama repo DagsHub (tanpa .mlflow)",
    )
    parser.add_argument(
        "--use-dagshub-init",
        action="store_true",
        help="Aktifkan dagshub.init agar kredensial dari env otomatis dipakai",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.train_path, args.target_col)
    print(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")

    auto_use_dagshub = bool(args.tracking_uri and "dagshub.com" in args.tracking_uri and DAGSHUB_AVAILABLE)
    use_dagshub = args.use_dagshub_init or auto_use_dagshub

    train_with_tuning(
        X,
        y,
        experiment_name=args.experiment_name,
        random_state=args.random_state,
        test_size=args.test_size,
        out_dir=args.artifacts_dir,
        tracking_uri=args.tracking_uri,
        dagshub_owner=args.dagshub_owner,
        dagshub_repo=args.dagshub_repo,
        use_dagshub_init=use_dagshub,
    )


if __name__ == "__main__":
    main()
