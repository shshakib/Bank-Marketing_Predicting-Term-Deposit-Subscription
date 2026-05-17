"""Data-cleaning pipeline for the bank marketing classifier.

This script standardizes the raw CSV, validates the expected schema, encodes
the target, and writes cleaned datasets for production modeling and diagnostic
comparison.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bank-data-processor")

TARGET_COLUMN = "deposit"
BANK_MARKETING_PRODUCTION_EXCLUSIONS: list[str] = ["duration", "pdays"]
BINARY_COLUMNS: list[str] = ["default", "housing", "loan"]
REQUIRED_COLUMNS: list[str] = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "deposit",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the raw bank marketing CSV into a DataFrame."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail early if the raw data does not match the expected source schema."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def clean_data(df: pd.DataFrame, keep_duration: bool = False) -> pd.DataFrame:
    """Clean the bank marketing dataset for serving-safe or diagnostic use.

    The production path drops ``duration`` because it is known only after a
    customer call. Keeping duration is useful only for diagnostic comparison to
    quantify how much post-call information would inflate model performance.
    """
    logger.info("Cleaning bank marketing dataset")
    _validate_schema(df)

    df_cleaned = df.copy()
    df_cleaned.columns = [column.strip().lower().replace(" ", "_") for column in df_cleaned.columns]

    string_columns = df_cleaned.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        df_cleaned[column] = df_cleaned[column].astype(str).str.strip().str.lower()

    for column in BINARY_COLUMNS:
        invalid = sorted(set(df_cleaned[column].dropna().unique()) - {"yes", "no", "unknown"})
        if invalid:
            raise ValueError(f"Unexpected values in {column}: {invalid}")

    invalid_target_values = set(df_cleaned[TARGET_COLUMN].dropna().unique()) - {"yes", "no"}
    if invalid_target_values:
        raise ValueError(
            f"Target column 'deposit' must contain only 'yes' and 'no': {invalid_target_values}"
        )

    for column in df_cleaned.columns:
        missing_count = int(df_cleaned[column].isnull().sum())
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in {column}")

            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(f"Filled missing values in {column} with median: {median_value}")
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled missing values in {column} with mode: {mode_value}")

    duplicate_count = int(df_cleaned.duplicated().sum())
    if duplicate_count:
        logger.info(f"Removing {duplicate_count} duplicate rows")
        df_cleaned = df_cleaned.drop_duplicates()

    df_cleaned[TARGET_COLUMN] = df_cleaned[TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)

    # Duration is excluded for production scoring because it is known only
    # after a call ends. It can be retained for the diagnostic comparison.
    columns_to_drop = BANK_MARKETING_PRODUCTION_EXCLUSIONS.copy()
    if keep_duration and "duration" in columns_to_drop:
        columns_to_drop.remove("duration")

    available_drop_columns = [column for column in columns_to_drop if column in df_cleaned.columns]
    if available_drop_columns:
        logger.info(
            "Dropping columns not used for production prediction: "
            f"{available_drop_columns}"
        )
        df_cleaned = df_cleaned.drop(columns=available_drop_columns)

    return df_cleaned


def process_data(
    input_file: str | Path,
    output_file: str | Path,
    keep_duration: bool = False,
) -> pd.DataFrame:
    """Load, clean, and persist one processed CSV artifact."""
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")

    df_cleaned = clean_data(df, keep_duration=keep_duration)
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df_cleaned


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the cleaning pipeline."""
    parser = argparse.ArgumentParser(description="Clean the bank marketing dataset.")
    parser.add_argument("--input", required=True, help="Path to raw bank.csv")
    parser.add_argument("--output", required=True, help="Path for cleaned output CSV")
    parser.add_argument(
        "--keep-duration",
        action="store_true",
        help="Keep duration for the diagnostic model-comparison scenario.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_data(input_file=args.input, output_file=args.output, keep_duration=args.keep_duration)
