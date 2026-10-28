"""Numerical, artifact, API, UI, and executable-notebook SHAP coverage."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
import yaml
from fastapi.testclient import TestClient

from src.api import inference
from src.api.explanations import BACKGROUND_FILE, load_explainer
from src.api.main import app
from src.data.run_processing import clean_data
from src.features.engineer import create_features, to_dense_frame
from src.models.train_model import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("shap-training")
    raw = pd.read_csv(ROOT / "data/raw/bank.csv").groupby("deposit", group_keys=False).sample(100, random_state=85)
    data = clean_data(raw).reset_index(drop=True)
    data_path = folder / "cleaned.csv"
    data.to_csv(data_path, index=False)
    config = folder / "config.yaml"
    config.write_text(yaml.safe_dump({"model": {
        "name": "bank_deposit_model", "target_variable": "deposit",
        "best_model": "GradientBoosting", "test_size": 0.2, "random_state": 85,
        "parameters": {"n_estimators": 12, "max_depth": 3, "random_state": 85},
    }}))
    main(argparse.Namespace(config=config, data=data_path, models_dir=folder, mlflow_tracking_uri=None))
    return folder / "trained", data_path, data, raw


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained[0]))
    for name in ("model", "preprocessor", "explainer", "EXPLANATION_ERROR", "MODEL_LOAD_ERROR", "MODEL_PATH", "PREPROCESSOR_PATH"):
        monkeypatch.setattr(inference, name, getattr(inference, name))
    inference.load_artifacts()
    with TestClient(app) as client:
        yield client


def test_background_uses_only_training_rows_and_is_reproducible(trained):
    from sklearn.model_selection import train_test_split
    folder, _, data, _ = trained
    bundle = joblib.load(folder / BACKGROUND_FILE)
    assert len(bundle["background"]) == 100
    assert set(bundle["background_indices"]).isdisjoint(bundle["test_indices"])
    train_indices, _ = train_test_split(np.arange(len(data)), test_size=0.2, random_state=85, stratify=data.deposit)
    assert bundle["background_indices"] == pd.Series(train_indices).sample(100, random_state=85).tolist()
    preprocessor = joblib.load(folder / "preprocessor.pkl")
    selected = create_features(data.iloc[bundle["background_indices"]].drop(columns="deposit"))
    expected = to_dense_frame(preprocessor.transform(selected), preprocessor)
    np.testing.assert_allclose(bundle["background"].values, expected.values)
    assert "duration" not in bundle["feature_groups"]
    assert "pdays" not in bundle["feature_groups"]


def test_api_explanation_reconstructs_single_and_batch_probabilities(client, trained):
    payloads = trained[3].head(4).drop(columns=["deposit", "duration", "pdays"]).to_dict("records")
    cached_explainer = inference.explainer
    assert client.get("/health").json()["explanations_available"]
    batch = client.post("/batch-predict", json=payloads)
    assert batch.status_code == 200
    for payload, result in zip(payloads, batch.json()):
        explanation = result["explanation"]
        assert explanation is not None
        assert set(explanation["contributions"]) == set(payload)
        assert explanation["output"] == "subscription_probability"
        reconstructed = explanation["base_probability"] + sum(explanation["contributions"].values())
        assert reconstructed == pytest.approx(explanation["prediction_probability"], abs=1e-6)
        assert reconstructed == pytest.approx(result["subscription_probability"], abs=5.1e-5)
        assert client.post("/predict", json=payload).json()["explanation"] == explanation
    assert inference.explainer is cached_explainer
    assert batch.json()[0]["explanation"]["contributions"] != batch.json()[1]["explanation"]["contributions"]
    assert client.post("/batch-predict", json=[]).json() == []


def test_grouping_matches_encoded_shap_sum(client):
    explainer = inference.explainer
    X = explainer.bundle["background"].iloc[:3]
    base, grouped = explainer.explain(X)
    raw_values = explainer.explainer.shap_values(X)
    for name in grouped:
        indices = [i for i, original in enumerate(explainer.groups) if original == name]
        np.testing.assert_allclose(grouped[name], raw_values[:, indices].sum(axis=1))
    assert base == pytest.approx(inference.model.predict_proba(explainer.bundle["background"])[:, 1].mean())


def test_invalid_attributions_are_not_shown(client, trained, monkeypatch):
    explainer = inference.explainer
    monkeypatch.setattr(explainer.explainer, "shap_values", lambda X: np.zeros(X.shape))
    with pytest.raises(ValueError, match="reconstruct"):
        explainer.explain(explainer.bundle["background"].iloc[:3])
    payload = trained[3].iloc[0].drop(labels=["deposit", "duration", "pdays"]).to_dict()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["explanation"] is None
    assert "could not be computed" in response.json()["explanation_error"]


@pytest.mark.parametrize("mode", ["missing", "stale"])
def test_unavailable_background_keeps_scoring_without_false_explanation(trained, client, tmp_path, monkeypatch, mode):
    import shutil
    folder = trained[0]
    for name in ["bank_deposit_model.pkl", "preprocessor.pkl"]:
        shutil.copyfile(folder / name, tmp_path / name)
    if mode == "stale":
        bundle = joblib.load(folder / BACKGROUND_FILE)
        bundle["model_sha256"] = "wrong-model"
        joblib.dump(bundle, tmp_path / BACKGROUND_FILE)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    inference.load_artifacts()
    health = client.get("/health").json()
    assert health["model_loaded"] and not health["explanations_available"]
    payload = trained[3].iloc[0].drop(labels=["deposit", "duration", "pdays"]).to_dict()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["explanation"] is None
    assert response.json()["explanation_error"]


def test_random_forest_positive_class_and_unsupported_linear_model(client):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from src.api.explanations import ProbabilityExplainer
    bundle = inference.explainer.bundle
    X = bundle["background"]
    y = np.arange(len(X)) % 2
    forest = RandomForestClassifier(n_estimators=5, random_state=85).fit(X, y)
    explainer = ProbabilityExplainer(forest, bundle)
    base, grouped = explainer.explain(X.iloc[:4])
    np.testing.assert_allclose(base + grouped.sum(axis=1), forest.predict_proba(X.iloc[:4])[:, 1], atol=1e-6)
    with pytest.raises(ValueError, match="supported model"):
        ProbabilityExplainer(LogisticRegression(), bundle)


def test_streamlit_explanation_chart_and_failed_rescore(client, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(requests, "get", lambda *a, **kw: client.get("/health"))
    monkeypatch.setattr(requests, "post", lambda *a, **kw: client.post("/predict", json=kw["json"]))
    ui = AppTest.from_file(str(ROOT / "streamlit_app/app.py"), default_timeout=30).run()
    assert not ui.exception
    ui.button[0].click().run()
    assert not ui.exception
    assert any(s.value == "Why this prediction?" for s in ui.subheader)
    chart = json.loads(ui.get("plotly_chart")[0].proto.spec)["data"][0]
    explanation = ui.session_state["prediction"]["explanation"]
    assert chart["measure"][-1] == "total"
    assert sum(chart["x"]) == pytest.approx(explanation["prediction_probability"] * 100, abs=1e-4)
    assert len(ui.dataframe[0].value) == 14
    ui.slider[0].set_value(50).run()
    assert len(ui.get("plotly_chart")) == 0
    ui.button[0].click().run()
    assert len(ui.get("plotly_chart")) == 1
    def fail(*args, **kwargs):
        raise requests.ConnectionError("API offline")
    monkeypatch.setattr(requests, "post", fail)
    ui.button[0].click().run()
    assert not ui.exception
    assert len(ui.get("plotly_chart")) == 0
    assert len(ui.error) == 1


def test_streamlit_missing_explanation_is_explicit(client, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(inference, "explainer", None)
    monkeypatch.setattr(inference, "EXPLANATION_ERROR", "SHAP background is missing. Retrain the model.")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: client.get("/health"))
    monkeypatch.setattr(requests, "post", lambda *a, **kw: client.post("/predict", json=kw["json"]))
    ui = AppTest.from_file(str(ROOT / "streamlit_app/app.py"), default_timeout=30).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert any("SHAP background is missing" in info.value for info in ui.info)
    assert len(ui.get("plotly_chart")) == 0


def test_explanation_notebook_executes(trained, tmp_path):
    env = dict(os.environ, BANK_SHAP_MODEL_DIR=str(trained[0]), BANK_SHAP_DATA=str(trained[1]),
               BANK_SHAP_OUTPUT_DIR=str(tmp_path), BANK_SHAP_SAMPLE_SIZE="8", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/execute_explanations.py"),
                             "--output", str(tmp_path / "executed.ipynb")],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ["global_importance.png", "beeswarm.png", "customer_waterfall.png", "executed.ipynb"]:
        assert (tmp_path / name).stat().st_size > 1000
