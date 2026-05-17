"""Feature-engineering utilities shared by notebooks, training, and inference.

The saved preprocessor is the contract between training and serving. Keeping
the same transformations here avoids one-hot encoding drift between the model
artifact and FastAPI requests.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bank-feature-engineering")

TARGET_COLUMN = "deposit"
BINARY_FEATURES: list[str] = ["default", "housing", "loan"]
CATEGORICAL_FEATURES: list[str] = ["job", "marital", "education", "contact", "month", "poutcome"]
NUMERICAL_FEATURES: list[str] = ["age", "balance", "day", "campaign", "previous"]
OPTIONAL_NUMERICAL_FEATURES: list[str] = ["duration"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature rules before one-hot encoding.

    Binary banking indicators become numeric ``0/1`` fields. The remaining
    categorical variables stay as strings because the persisted preprocessor is
    responsible for creating a stable dummy-variable schema.
    """
    logger.info("Creating bank marketing features")

    df_featured = df.copy()

    for column in df_featured.select_dtypes(include=["object", "string"]).columns:
        df_featured[column] = df_featured[column].astype(str).str.strip().str.lower()

    if TARGET_COLUMN in df_featured.columns and df_featured[TARGET_COLUMN].dtype == object:
        df_featured[TARGET_COLUMN] = df_featured[TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)

    for column in BINARY_FEATURES:
        if column in df_featured.columns:
            df_featured[column] = df_featured[column].map({"no": 0, "yes": 1, "unknown": 0}).astype(int)

    return df_featured


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across supported scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def get_feature_groups(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split available columns into numeric and categorical preprocessing groups."""
    available = set(columns)
    numerical = [column for column in NUMERICAL_FEATURES + OPTIONAL_NUMERICAL_FEATURES if column in available]
    numerical += [column for column in BINARY_FEATURES if column in available]
    categorical = [column for column in CATEGORICAL_FEATURES if column in available]
    return numerical, categorical


def create_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    """Create the preprocessing pipeline used by training and inference.

    The first level is dropped for each categorical variable to avoid duplicate
    dummy columns. Unknown categories are ignored at inference time so the API
    can still score valid requests if a rare category appears after training.
    """
    logger.info("Creating bank marketing preprocessor")

    numerical_features, categorical_features = get_feature_groups(columns)

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def to_dense_frame(transformed: Any, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Convert transformer output into a named DataFrame for model training."""
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    cleaned_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    return pd.DataFrame(transformed, columns=cleaned_names)


def run_feature_engineering(
    input_file: str | Path,
    output_file: str | Path,
    preprocessor_file: str | Path,
) -> pd.DataFrame:
    """Create model-ready features and persist the fitted preprocessor."""
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    df_featured = create_features(df)
    logger.info(f"Created featured dataset with shape: {df_featured.shape}")

    X = df_featured.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df_featured[TARGET_COLUMN] if TARGET_COLUMN in df_featured.columns else None

    preprocessor = create_preprocessor(X.columns)
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")

    Path(preprocessor_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    df_transformed = to_dense_frame(X_transformed, preprocessor)
    if y is not None:
        df_transformed[TARGET_COLUMN] = y.values

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")

    return df_transformed


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the feature-engineering pipeline."""
    parser = argparse.ArgumentParser(description="Feature engineering for bank marketing data.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--output", required=True, help="Path for output CSV file")
    parser.add_argument("--preprocessor", required=True, help="Path for saving the preprocessor")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_engineering(args.input, args.output, args.preprocessor)
