"""Regression coverage for cleaning, evaluation, training, and serving."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.api import inference
from src.api.main import app
from src.data.run_processing import clean_data
from src.features.engineer import create_features, create_preprocessor
from src.models import compare_models, train_model


@pytest.fixture
def raw_data():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data/raw/bank.csv")


@pytest.fixture
def payload(raw_data):
    return raw_data.iloc[0].drop(labels=["duration", "pdays", "deposit"]).to_dict()


def test_cleaning_preserves_missing_predictors_for_training_imputation(raw_data):
    raw = raw_data.iloc[:10].copy()
    raw.loc[0, ["job", "balance", "housing"]] = np.nan
    cleaned = clean_data(raw)
    assert cleaned.loc[0, ["job", "balance", "housing"]].isna().all()
    features = create_features(cleaned)
    assert features.loc[0, ["job", "balance", "housing"]].isna().all()
    transformed = create_preprocessor(features.drop(columns="deposit").columns).fit_transform(features)
    assert np.isfinite(transformed).all()


def test_cleaning_normalizes_headers_before_validation(raw_data):
    raw = raw_data.iloc[:10].rename(columns=lambda name: f" {name.upper()} ")
    assert "deposit" in clean_data(raw)


def test_missing_target_is_rejected(raw_data):
    raw = raw_data.iloc[:10].copy()
    raw.loc[0, "deposit"] = np.nan
    with pytest.raises(ValueError, match="deposit"):
        clean_data(raw)


def test_metrics_use_subscription_as_positive_class():
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])

    class FixedModel:
        def predict(self, X):
            return y_pred

        def predict_proba(self, X):
            return np.column_stack([1 - y_pred, y_pred])

    for metrics in (
        compare_models.evaluate_predictions(y_true, y_pred),
        train_model.evaluate_model(FixedModel(), pd.DataFrame(index=y_true.index), y_true),
    ):
        assert metrics["sensitivity"] == 0.5
        assert metrics["specificity"] == 0.75
        assert metrics["balanced_accuracy"] == 0.625


@pytest.mark.parametrize("runner", ["rf", "gbm"])
def test_grid_search_fits_preprocessing_inside_each_fold(raw_data, monkeypatch, runner):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    def checked_grid(**kwargs):
        assert isinstance(kwargs["estimator"], Pipeline)
        assert "preprocessor" in kwargs["estimator"].named_steps
        kwargs["n_jobs"] = 1
        if runner == "gbm":
            kwargs["param_grid"]["model__n_estimators"] = [2]
        return GridSearchCV(**kwargs)

    monkeypatch.setattr(compare_models, "GridSearchCV", checked_grid)
    subset = raw_data.groupby("deposit", group_keys=False).head(30)
    train, test = compare_models.split_scenario(clean_data(subset), 85)
    if runner == "rf":
        metrics = compare_models.run_rf(train, test, 2, 85, 2)
        assert "best_mtry" in metrics
    else:
        metrics = compare_models.run_gbm(train, test, 2, 85, False)
        assert "best_n_estimators" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_training_saves_train_only_preprocessor_and_serves_predictions(raw_data, tmp_path, monkeypatch, payload):
    subset = raw_data.groupby("deposit", group_keys=False).head(40)
    cleaned = clean_data(subset).reset_index(drop=True)
    train, test = train_test_split(cleaned, test_size=0.2, random_state=85, stratify=cleaned.deposit)
    cleaned.loc[test.index, "job"] = "holdout-only"
    cleaned.loc[test.index, "balance"] = 999999999
    data_path = tmp_path / "cleaned.csv"
    cleaned.to_csv(data_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {
        "name": "bank_deposit_model", "target_variable": "deposit",
        "best_model": "GradientBoosting", "test_size": 0.2, "random_state": 85,
        "parameters": {"n_estimators": 5, "random_state": 85},
    }}))
    train_model.main(argparse.Namespace(config=str(config_path), data=str(data_path),
                                        models_dir=str(tmp_path), mlflow_tracking_uri=None))
    preprocessor = joblib.load(tmp_path / "trained/preprocessor.pkl")
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    assert "holdout-only" not in encoder.categories_[0]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[1] == train.balance.median()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "trained"))
    inference.load_artifacts()
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            assert 0 <= result["subscription_probability"] <= 1
            batch = client.post("/batch-predict", json=[payload, payload])
            assert batch.status_code == 200
            assert [r["subscription_probability"] for r in batch.json()] == [result["subscription_probability"]] * 2
    finally:
        inference.model = inference.preprocessor = None


def test_explicit_model_dir_does_not_fall_back(tmp_path, monkeypatch, payload):
    fallback = tmp_path / "models/trained"
    fallback.mkdir(parents=True)
    joblib.dump({"fallback": True}, fallback / inference.MODEL_FILE)
    joblib.dump({"fallback": True}, fallback / inference.PREPROCESSOR_FILE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    assert inference._artifact_path(inference.MODEL_FILE) == tmp_path / "missing" / inference.MODEL_FILE
    inference.load_artifacts()
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 503
            assert client.post("/predict", json=payload).status_code == 503
    finally:
        inference.model = inference.preprocessor = None


def test_artifacts_are_not_mixed_across_directories(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    joblib.dump({"model": True}, first / inference.MODEL_FILE)
    joblib.dump({"preprocessor": True}, second / inference.PREPROCESSOR_FILE)
    monkeypatch.setattr(inference, "_artifact_roots", lambda: [first, second])
    assert inference._artifact_path(inference.MODEL_FILE).parent == inference._artifact_path(inference.PREPROCESSOR_FILE).parent
    inference.load_artifacts()
    try:
        assert not inference.artifacts_loaded()
    finally:
        inference.model = inference.preprocessor = None


@pytest.mark.parametrize("balance", ["nan", "inf", "-inf"])
def test_api_rejects_nonfinite_balances(payload, balance):
    with TestClient(app) as client:
        response = client.post("/predict", json={**payload, "balance": balance})
    assert response.status_code == 422


def test_api_normalizes_categories_and_rejects_invalid_values(payload):
    from src.api.schemas import BankPredictionRequest
    assert BankPredictionRequest(**{**payload, "job": " Technician "}).job == "technician"
    with TestClient(app) as client:
        assert client.post("/predict", json={**payload, "job": "invalid"}).status_code == 422


def test_incompatible_artifact_feature_order_is_unhealthy(tmp_path, monkeypatch):
    from sklearn.ensemble import GradientBoostingClassifier
    from src.features.engineer import to_dense_frame

    features = pd.DataFrame({"age": [20, 30, 40, 50], "balance": [1, 2, 3, 4]})
    preprocessor = create_preprocessor(features.columns)
    encoded = to_dense_frame(preprocessor.fit_transform(features), preprocessor)
    model = GradientBoostingClassifier(n_estimators=2).fit(encoded.iloc[:, ::-1], [0, 1, 0, 1])
    joblib.dump(model, tmp_path / inference.MODEL_FILE)
    joblib.dump(preprocessor, tmp_path / inference.PREPROCESSOR_FILE)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    inference.load_artifacts()
    try:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 503
            assert "feature names or order" in response.json()["model_load_error"]
    finally:
        inference.model = inference.preprocessor = None
