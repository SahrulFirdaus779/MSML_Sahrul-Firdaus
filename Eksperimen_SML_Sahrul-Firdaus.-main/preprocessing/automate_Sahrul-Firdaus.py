import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

DEFAULT_TARGET = "Target"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("data_clustering_preprocessing")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def build_preprocess(feature_names):
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([
        ("num", numeric_steps, feature_names),
    ])


def preprocess(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocess_pipe = build_preprocess(X.columns.tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocess_pipe.fit(X_train)
    X_train_processed = preprocess_pipe.transform(X_train)
    X_test_processed = preprocess_pipe.transform(X_test)
    feature_names = preprocess_pipe.get_feature_names_out()

    return preprocess_pipe, feature_names, X_train_processed, y_train, X_test_processed, y_test


def save_outputs(out_dir: Path, feature_names, preprocess_pipe, X_train_processed, y_train, X_test_processed, y_test, target_col: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocess_pipe, out_dir / "preprocess_pipeline.joblib")

    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df[target_col] = y_train.to_numpy()
    train_df.to_csv(out_dir / "train_processed.csv", index=False)

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df[target_col] = y_test.to_numpy()
    test_df.to_csv(out_dir / "test_processed.csv", index=False)


def summarize(df: pd.DataFrame, target_col: str):
    total_missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    target_counts = df[target_col].value_counts().to_dict() if target_col in df.columns else {}
    return total_missing, duplicates, target_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset using the MSML template steps")
    parser.add_argument("--data-path", type=Path, default=Path("../data_clustering_raw/data_clustering.csv"), help="Path to input CSV dataset")
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test size fraction for the split")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random state for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store processed outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_dataset(args.data_path)
    missing, duplicates, target_counts = summarize(df, args.target_col)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {args.data_path}")
    print(f"Missing values (total): {missing}")
    print(f"Duplicate rows: {duplicates}")
    if target_counts:
        print(f"Target distribution: {target_counts}")

    preprocess_pipe, feature_names, X_train_p, y_train, X_test_p, y_test = preprocess(
        df, args.target_col, args.test_size, args.random_state
    )

    save_outputs(
        args.output_dir,
        feature_names,
        preprocess_pipe,
        X_train_p,
        y_train,
        X_test_p,
        y_test,
        args.target_col,
    )

    print(f"Train processed shape: {X_train_p.shape}")
    print(f"Test processed shape:  {X_test_p.shape}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
