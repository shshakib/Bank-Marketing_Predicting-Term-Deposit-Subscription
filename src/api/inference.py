"""Model artifact loading and inference utilities for the FastAPI service."""

import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

try:
    from .schemas import BankPredictionRequest, PredictionResponse
except ImportError:
    from schemas import BankPredictionRequest, PredictionResponse

MODEL_FILE = "bank_deposit_model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
MODEL_DIR_ENV = "MODEL_DIR"
PROBABILITY_RANGE_MARGIN = 0.10


def _artifact_roots():
    """Return artifact directories in the order they should be searched."""
    roots = []

    configured_dir = os.getenv(MODEL_DIR_ENV)
    if configured_dir:
        roots.append(Path(configured_dir))

    current_file = Path(__file__).resolve()
    roots.extend([
        Path.cwd() / "models" / "trained",
        current_file.parent / "models" / "trained",
    ])

    # Walking upward makes the same code work from the project root, src/api,
    # and the Docker image where API files are copied directly under /app.
    for parent in current_file.parents:
        roots.append(parent / "models" / "trained")

    unique_roots = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _artifact_path(file_name):
    """Find a model artifact locally, or return the first expected location."""
    candidates = [root / file_name for root in _artifact_roots()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_PATH = _artifact_path(MODEL_FILE)
PREPROCESSOR_PATH = _artifact_path(PREPROCESSOR_FILE)
MODEL_LOAD_ERROR = None

model = None
preprocessor = None


def load_artifacts():
    """Load model artifacts and store any failure for health diagnostics."""
    global model, preprocessor, MODEL_LOAD_ERROR, MODEL_PATH, PREPROCESSOR_PATH

    try:
        MODEL_PATH = _artifact_path(MODEL_FILE)
        PREPROCESSOR_PATH = _artifact_path(PREPROCESSOR_FILE)
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        model = None
        preprocessor = None
        MODEL_LOAD_ERROR = str(exc)


load_artifacts()


def artifacts_loaded():
    """Return whether both required artifacts are available in memory."""
    return model is not None and preprocessor is not None


def _artifact_metadata(path):
    """Expose basic file metadata so health checks can reveal stale artifacts."""
    metadata = {"path": str(path), "exists": path.exists()}
    if path.exists():
        metadata["modified_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return metadata


def get_model_health():
    """Return the model-serving readiness payload used by /health."""
    if not artifacts_loaded():
        load_artifacts()

    return {
        "status": "healthy" if artifacts_loaded() else "unhealthy",
        "model_loaded": artifacts_loaded(),
        "task": "binary_classification",
        "model_artifact": _artifact_metadata(MODEL_PATH),
        "preprocessor_artifact": _artifact_metadata(PREPROCESSOR_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
    }


def _require_artifacts():
    """Return loaded artifacts or raise a service-level error."""
    if not artifacts_loaded():
        load_artifacts()
    if not artifacts_loaded():
        raise RuntimeError(f"Model artifacts are not available: {MODEL_LOAD_ERROR}")
    return model, preprocessor


def create_features(df):
    """Mirror training-time feature rules before applying the preprocessor."""
    df_featured = df.copy()

    for column in df_featured.select_dtypes(include=["object", "string"]).columns:
        df_featured[column] = df_featured[column].astype(str).str.strip().str.lower()

    for column in ["default", "housing", "loan"]:
        if column in df_featured.columns:
            mapped = df_featured[column].map({"no": 0, "yes": 1})
            if mapped.isnull().any():
                raise ValueError(f"{column} must contain only yes/no values")
            df_featured[column] = mapped.astype(int)

    return df_featured


def _top_global_model_factors(loaded_model, loaded_preprocessor, limit=8):
    """Return global feature importances for tree-based models."""
    if not hasattr(loaded_model, "feature_importances_"):
        return {}
    try:
        names = loaded_preprocessor.get_feature_names_out()
    except Exception:
        names = [f"feature_{idx}" for idx in range(len(loaded_model.feature_importances_))]

    cleaned_names = [name.replace("num__", "").replace("cat__", "") for name in names]
    importances = sorted(
        zip(cleaned_names, loaded_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name: round(float(score), 4) for name, score in importances[:limit]}


def _transform_for_model(featured_data, loaded_preprocessor):
    """Apply the saved preprocessing pipeline and preserve feature names."""
    processed_features = loaded_preprocessor.transform(featured_data)
    if hasattr(processed_features, "toarray"):
        processed_features = processed_features.toarray()
    try:
        names = loaded_preprocessor.get_feature_names_out()
        names = [name.replace("num__", "").replace("cat__", "") for name in names]
        return pd.DataFrame(processed_features, columns=names)
    except Exception:
        return processed_features


def _probability_range(probability):
    """Create a simple display range around the raw probability.

    This range is intentionally named differently from a confidence interval:
    it is a user-facing sensitivity range, not an uncertainty estimate from a
    calibrated statistical procedure.
    """
    return [
        round(max(0.0, probability - PROBABILITY_RANGE_MARGIN), 4),
        round(min(1.0, probability + PROBABILITY_RANGE_MARGIN), 4),
    ]


def predict_subscription(request: BankPredictionRequest) -> PredictionResponse:
    """Predict whether a customer will subscribe to a term deposit."""
    loaded_model, loaded_preprocessor = _require_artifacts()

    input_data = pd.DataFrame([request.dict()])
    featured_data = create_features(input_data)
    processed_features = _transform_for_model(featured_data, loaded_preprocessor)

    probability = float(loaded_model.predict_proba(processed_features)[0][1])
    prediction_label = int(probability >= 0.5)
    predicted_deposit = "yes" if prediction_label else "no"

    return PredictionResponse(
        predicted_deposit=predicted_deposit,
        subscription_probability=round(probability, 4),
        prediction_label=prediction_label,
        probability_range=_probability_range(probability),
        top_model_factors=_top_global_model_factors(loaded_model, loaded_preprocessor),
        prediction_time=datetime.now().isoformat(),
    )


def batch_predict(requests: list[BankPredictionRequest]) -> list[PredictionResponse]:
    """Score a list of customers by reusing the single-record prediction path."""
    return [predict_subscription(request) for request in requests]
