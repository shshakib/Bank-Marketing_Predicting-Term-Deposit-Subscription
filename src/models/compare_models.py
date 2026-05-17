"""Compare candidate classifiers across production-safe and diagnostic scenarios."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    import mlflow
except ImportError:
    mlflow = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.run_processing import clean_data
from src.features.engineer import create_features, create_preprocessor, to_dense_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the compact metric set used for candidate model comparison."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity_no = tn / (tn + fp) if (tn + fp) else 0.0
    specificity_yes = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity_no),
        "specificity": float(specificity_yes),
        "balanced_accuracy": float((sensitivity_no + specificity_yes) / 2),
    }


def encode_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fit preprocessing on training data and transform train/test splits."""
    X_train = create_features(train_df.drop(columns=["deposit"]))
    X_test = create_features(test_df.drop(columns=["deposit"]))
    y_train = train_df["deposit"].astype(int)
    y_test = test_df["deposit"].astype(int)

    preprocessor = create_preprocessor(X_train.columns)
    X_train_encoded = to_dense_frame(preprocessor.fit_transform(X_train), preprocessor)
    X_test_encoded = to_dense_frame(preprocessor.transform(X_test), preprocessor)
    return X_train_encoded, X_test_encoded, y_train, y_test


def split_scenario(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified holdout split for one modeling scenario."""
    return train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df["deposit"],
    )


def run_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_folds: int,
    random_state: int,
    rf_trees: int,
) -> dict[str, Any]:
    """Tune and evaluate the Random Forest candidate model."""
    X_train, X_test, y_train, y_test = encode_train_test(train_df, test_df)
    grid = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=rf_trees, random_state=random_state, n_jobs=-1),
        param_grid={"max_features": [3, 4, 5, 6]},
        scoring="accuracy",
        cv=cv_folds,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["best_mtry"] = int(grid.best_params_["max_features"])
    metrics["n_trees"] = int(rf_trees)
    return metrics


def run_gbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_folds: int,
    random_state: int,
    full_grid: bool,
) -> dict[str, Any]:
    """Tune and evaluate the Gradient Boosting candidate model."""
    X_train, X_test, y_train, y_test = encode_train_test(train_df, test_df)
    if full_grid:
        param_grid = {
            "n_estimators": [300, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [5, 10, 20],
        }
    else:
        param_grid = {
            "n_estimators": [500],
            "learning_rate": [0.05],
            "max_depth": [7],
            "min_samples_leaf": [5],
        }

    grid = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_folds,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics.update({f"best_{key}": value for key, value in grid.best_params_.items()})
    return metrics


def run_single_baseline(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Fit and evaluate a baseline model without hyperparameter search."""
    X_train, X_test, y_train, y_test = encode_train_test(train_df, test_df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)


def run_mccv_baselines(
    df: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Summarize Logistic Regression and LDA stability with Monte Carlo CV."""
    rng = np.random.default_rng(random_state)
    n_rows = len(df)
    n_test = round(n_rows / 10)
    records = []

    for iteration in range(iterations):
        test_index = np.sort(rng.choice(n_rows, size=n_test, replace=False))
        test_mask = np.zeros(n_rows, dtype=bool)
        test_mask[test_index] = True
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()

        for model_name, model in {
            "Logistic Regression": LogisticRegression(max_iter=10000, random_state=random_state),
            "LDA": LinearDiscriminantAnalysis(),
        }.items():
            metrics = run_single_baseline(model, train_df, test_df)
            records.append({
                "iteration": iteration + 1,
                "model": model_name,
                **metrics,
            })

    mccv = pd.DataFrame(records)
    summary = (
        mccv.groupby("model")
        .agg(
            train_iterations=("iteration", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.to_dict(orient="records")


def prepare_scenario(raw_df: pd.DataFrame, keep_duration: bool) -> pd.DataFrame:
    """Build the cleaned DataFrame for one comparison scenario."""
    cleaned = clean_data(raw_df, keep_duration=keep_duration)
    return cleaned.reset_index(drop=True)


def log_results_to_mlflow(results: list[dict[str, Any]], tracking_uri: str | None) -> None:
    """Optionally log model-comparison metrics to an MLflow experiment."""
    if not tracking_uri:
        return
    if mlflow is None:
        logger.warning("MLflow is not installed; skipping comparison logging")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("bank_deposit_model_comparison")
    for result in results:
        with mlflow.start_run(run_name=f"{result['scenario']} - {result['model']}"):
            mlflow.log_params(result.get("params", {}))
            mlflow.log_metrics({
                key: value
                for key, value in result["metrics"].items()
                if isinstance(value, (int, float))
            })


def run_comparison(args: argparse.Namespace) -> None:
    """Run all candidate models and persist YAML/CSV comparison artifacts.

    The comparison intentionally evaluates both the production-safe feature set
    and the with-duration diagnostic feature set. The latter is useful for
    understanding leakage impact, but it should not be promoted to the API.
    """
    raw_df = pd.read_csv(args.raw_data)
    all_results = []

    for scenario, keep_duration in {
        "No Duration": False,
        "With Duration": True,
    }.items():
        logger.info(f"Running scenario: {scenario}")
        scenario_df = prepare_scenario(raw_df, keep_duration=keep_duration)
        train_df, test_df = split_scenario(scenario_df, random_state=args.random_state)

        model_results = {
            "Random Forest": run_rf(train_df, test_df, args.cv_folds, args.random_state, args.rf_trees),
            "GBM": run_gbm(train_df, test_df, args.cv_folds, args.random_state, args.full_gbm_grid),
            "Logistic Regression": run_single_baseline(
                LogisticRegression(max_iter=10000, random_state=args.random_state),
                train_df,
                test_df,
            ),
            "LDA": run_single_baseline(LinearDiscriminantAnalysis(), train_df, test_df),
        }

        # Monte Carlo CV is reserved for the linear baselines to estimate
        # stability without making the ensemble grid search prohibitively slow.
        mccv_summary = run_mccv_baselines(scenario_df, args.mccv_iterations, args.random_state)

        for model_name, metrics in model_results.items():
            all_results.append({
                "scenario": scenario,
                "model": model_name,
                "metrics": metrics,
                "params": {
                    "keep_duration": keep_duration,
                    "cv_folds": args.cv_folds,
                    "random_state": args.random_state,
                },
            })

        all_results.append({
            "scenario": scenario,
            "model": "Logistic/LDA MCCV",
            "metrics": {"mccv_iterations": args.mccv_iterations},
            "mccv_summary": mccv_summary,
            "params": {"keep_duration": keep_duration, "random_state": args.random_state},
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump({"results": all_results}, f, sort_keys=False)

    rows = []
    for result in all_results:
        if "accuracy" in result["metrics"]:
            rows.append({
                "scenario": result["scenario"],
                "model": result["model"],
                **result["metrics"],
            })
    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    log_results_to_mlflow(all_results, args.mlflow_tracking_uri)
    logger.info(f"Saved comparison results to {output_path}")
    logger.info(f"Saved comparison table to {csv_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the model-comparison workflow."""
    parser = argparse.ArgumentParser(description="Run the bank marketing model comparison workflow.")
    parser.add_argument("--raw-data", required=True, help="Path to raw bank.csv")
    parser.add_argument("--output", required=True, help="Path for comparison YAML")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--mccv-iterations", type=int, default=100, help="Monte Carlo CV iterations")
    parser.add_argument("--cv-folds", type=int, default=5, help="Grid-search CV folds")
    parser.add_argument("--rf-trees", type=int, default=1000, help="Random Forest tree count")
    parser.add_argument("--random-state", type=int, default=85, help="Random seed for reproducible experiments")
    parser.add_argument("--full-gbm-grid", action="store_true", help="Run the expanded GBM hyperparameter grid")
    return parser.parse_args()


if __name__ == "__main__":
    run_comparison(parse_args())
