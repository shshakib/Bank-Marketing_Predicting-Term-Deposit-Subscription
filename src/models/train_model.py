"""Final model-training script for the production scoring artifact."""

from __future__ import annotations

import argparse
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible training runs."""
    parser = argparse.ArgumentParser(description="Train the bank deposit subscription model.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()


def get_model_instance(name: str, params: dict[str, Any]) -> Any:
    """Instantiate the estimator selected in the YAML model config."""
    model_map = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate classification metrics used by notebooks and monitoring.

    Sensitivity is reported for the negative class and specificity for the
    positive subscription class to stay consistent with the project comparison
    artifacts.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall_yes = recall_score(y_test, y_pred, zero_division=0)
    precision_yes = precision_score(y_test, y_pred, zero_division=0)
    sensitivity_no = tn / (tn + fp) if (tn + fp) else 0.0
    specificity_yes = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "sensitivity": float(sensitivity_no),
        "specificity": float(specificity_yes),
        "balanced_accuracy": float((sensitivity_no + specificity_yes) / 2),
        "precision_yes": float(precision_yes),
        "recall_yes": float(recall_yes),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def log_to_mlflow(
    model: Any,
    model_cfg: dict[str, Any],
    metrics: dict[str, float],
    args: argparse.Namespace,
    save_path: Path,
) -> None:
    """Optionally log the trained model, metrics, and metadata to MLflow."""
    if not args.mlflow_tracking_uri:
        logger.info("No MLflow tracking URI provided; skipping MLflow logging")
        return
    if mlflow is None:
        logger.warning("MLflow is not installed; skipping MLflow logging")
        return

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(model_cfg["name"])

    with mlflow.start_run(run_name="final_training"):
        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "bank_deposit_model")

        if MlflowClient is None:
            return

        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/bank_deposit_model"
        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
        except Exception as exc:
            logger.debug(f"Registered model already exists or cannot be created: {exc}")

        try:
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id,
            )
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
            )
        except Exception as exc:
            logger.warning(f"Could not register model version: {exc}")

        description = (
            "Model for predicting whether a bank customer will subscribe to a term deposit. "
            "Duration and pdays are excluded from the production feature set."
        )
        try:
            client.update_registered_model(name=model_name, description=description)
            client.set_registered_model_tag(model_name, "algorithm", model_cfg["best_model"])
            client.set_registered_model_tag(model_name, "target_variable", model_cfg["target_variable"])
            client.set_registered_model_tag(model_name, "training_dataset", args.data)
            client.set_registered_model_tag(model_name, "model_path", str(save_path))
        except Exception as exc:
            logger.warning(f"Could not update model registry metadata: {exc}")


def main(args: argparse.Namespace) -> None:
    """Train, evaluate, save, and optionally register the production model.

    This script does not automatically choose the best model. It trains the
    model family and parameters declared in ``configs/model_config.yaml`` after
    the model-comparison workflow has been reviewed.
    """
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    data = pd.read_csv(args.data)
    target = model_cfg["target_variable"]

    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_cfg.get("test_size", 0.2),
        random_state=model_cfg.get("random_state", 42),
        stratify=y,
    )

    # The production artifact is intentionally controlled by config so model
    # promotion is explicit and reproducible.
    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])
    logger.info(f"Training model: {model_cfg['best_model']}")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    model_name = model_cfg["name"]
    save_path = Path(args.models_dir) / "trained" / f"{model_name}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Saved trained model to: {save_path}")

    metrics_path = save_path.parent / f"{model_name}_metrics.yaml"
    metrics_payload = {
        "model": model_name,
        "algorithm": model_cfg["best_model"],
        "target_variable": target,
        "metrics": metrics,
        "dependencies": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }
    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics_payload, f, sort_keys=False)
    logger.info(f"Saved metrics to: {metrics_path}")

    log_to_mlflow(model, model_cfg, metrics, args, save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
