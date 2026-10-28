"""Probability-space TreeSHAP shared by serving and the analysis notebook."""

from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

BACKGROUND_FILE = "shap_background.pkl"
SUPPORTED_MODELS = (GradientBoostingClassifier, RandomForestClassifier)


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def feature_groups(preprocessor):
    """Map each encoded column to its original input, including dropped dummies."""
    groups = []
    originals = list(preprocessor.feature_names_in_)
    for name in preprocessor.get_feature_names_out():
        kind, encoded = name.split("__", 1)
        matches = [c for c in originals if encoded == c or (kind == "cat" and encoded.startswith(c + "_"))]
        if not matches:
            raise ValueError(f"Cannot map encoded feature {name}")
        groups.append(max(matches, key=len))
    return groups


def save_background(model, preprocessor, X_train, model_path, preprocessor_path,
                    *, dataset_hash, test_indices, training_indices, random_state):
    """Persist a bounded training-only reference, bound to the exact artifacts."""
    background = X_train.sample(n=min(100, len(X_train)), random_state=random_state)
    bundle = {
        "version": 1,
        "supported": isinstance(model, SUPPORTED_MODELS),
        "background": background,
        "feature_groups": feature_groups(preprocessor),
        "model_sha256": file_digest(model_path),
        "preprocessor_sha256": file_digest(preprocessor_path),
        "dataset_hash": dataset_hash,
        "test_indices": list(map(int, test_indices)),
        "background_indices": [int(training_indices[i]) for i in background.index],
    }
    joblib.dump(bundle, Path(model_path).parent / BACKGROUND_FILE)


class ProbabilityExplainer:
    def __init__(self, model, bundle):
        import shap

        if not isinstance(model, SUPPORTED_MODELS) or not bundle["supported"]:
            raise ValueError("SHAP explanations require GradientBoosting or RandomForest; retrain with a supported model.")
        if list(model.classes_) != [0, 1]:
            raise ValueError("SHAP requires subscription=1 as the positive class")
        self.model = model
        self.bundle = bundle
        self.groups = bundle["feature_groups"]
        self.names = list(dict.fromkeys(self.groups))
        self.explainer = shap.TreeExplainer(
            model, data=bundle["background"], model_output="probability",
            feature_perturbation="interventional",
        )

    def explain(self, X):
        """Return all original-field contributions, preserving their signed sum."""
        if list(X.columns) != list(self.bundle["background"].columns):
            raise ValueError("SHAP input columns do not match the background")
        values = np.asarray(self.explainer.shap_values(X))
        expected = np.asarray(self.explainer.expected_value)
        # RandomForest returns a separate output for each class; GBM returns
        # the positive-class output directly. Never explain class 0 by accident.
        if values.ndim == 3:
            values = values[:, :, 1]
            expected = expected[1]
        base = float(expected)
        grouped = np.column_stack([
            values[:, [i for i, group in enumerate(self.groups) if group == name]].sum(axis=1)
            for name in self.names
        ])
        probabilities = self.model.predict_proba(X)[:, 1]
        if not np.isfinite(grouped).all() or not np.isfinite(base):
            raise ValueError("SHAP produced nonfinite values")
        # SHAP's built-in additivity check does not cover every probability
        # configuration. Verify it explicitly before exposing explanations.
        if not np.allclose(base + grouped.sum(axis=1), probabilities, atol=1e-6, rtol=0):
            raise ValueError("SHAP contributions do not reconstruct the prediction")
        return base, pd.DataFrame(grouped, columns=self.names, index=X.index)


def load_explainer(model, preprocessor, model_path, preprocessor_path):
    path = Path(model_path).parent / BACKGROUND_FILE
    if not path.is_file():
        raise ValueError("SHAP background is missing. Retrain the model and restart the API.")
    bundle = joblib.load(path)
    if bundle.get("version") != 1:
        raise ValueError("Unsupported SHAP artifact version. Retrain the model.")
    if (bundle["model_sha256"] != file_digest(model_path)
            or bundle["preprocessor_sha256"] != file_digest(preprocessor_path)):
        raise ValueError("SHAP background does not match the model/preprocessor. Retrain and restart the API.")
    if bundle["feature_groups"] != feature_groups(preprocessor):
        raise ValueError("SHAP feature mapping does not match preprocessing")
    return ProbabilityExplainer(model, bundle)
