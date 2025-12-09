import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DEFAULT_TARGET = "Target"
DEFAULT_EXPERIMENT = "data_clustering_basic"


def load_split(train_path: Path, test_path: Path, target_col: str):
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing in train data")
    if target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' missing in test data")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def train_and_log(X_train, X_test, y_train, y_test, experiment_name: str, random_state: int):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=True)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="random_forest_autolog"):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline model with MLflow autolog")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data_clustering_preprocessing/train_processed.csv"),
        help="Path to preprocessed train CSV",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data_clustering_preprocessing/test_processed.csv"),
        help="Path to preprocessed test CSV",
    )
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_split(args.train_path, args.test_path, args.target_col)

    print(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    class_counts = np.bincount(y_train.to_numpy()) if y_train.dtype != object else y_train.value_counts().to_dict()
    print(f"Target distribution (train): {class_counts}")

    train_and_log(X_train, X_test, y_train, y_test, args.experiment_name, args.random_state)


if __name__ == "__main__":
    main()
