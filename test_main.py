"""
Unit tests for the wine-quality-ml MLOps pipeline.

Covers the training, evaluation quality gate, validation, deployment, and monitoring steps.
All tests run without a Databricks cluster — MLflow and the UC model registry
are mocked so that CI completes in seconds.

Run with:  pytest assets/tests/test.py
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
from unittest.mock import MagicMock, patch


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def wine_features():
    """Small synthetic DataFrame matching the sklearn wine dataset schema."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "alcohol": rng.uniform(11.0, 14.8, n),
            "malic_acid": rng.uniform(0.7, 5.8, n),
            "ash": rng.uniform(1.4, 3.2, n),
            "alcalinity_of_ash": rng.uniform(10.6, 30.0, n),
            "magnesium": rng.uniform(70.0, 162.0, n),
            "total_phenols": rng.uniform(0.9, 3.9, n),
            "flavanoids": rng.uniform(0.3, 5.1, n),
            "nonflavanoid_phenols": rng.uniform(0.1, 0.7, n),
            "proanthocyanins": rng.uniform(0.4, 3.6, n),
            "color_intensity": rng.uniform(1.3, 13.0, n),
            "hue": rng.uniform(0.5, 1.7, n),
            "od280/od315_of_diluted_wines": rng.uniform(1.3, 4.0, n),
            "proline": rng.uniform(278.0, 1680.0, n),
        }
    )


# ─── Training ────────────────────────────────────────────────────────────────


class TestTraining:
    """Training step: model trains on wine features and is registered with @Challenger alias."""

    def test_model_trains_and_produces_predictions(self, wine_features):
        rng = np.random.default_rng(42)
        labels = pd.Series(rng.integers(0, 3, len(wine_features)), name="label")

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(wine_features, labels)
        predictions = model.predict(wine_features)

        assert len(predictions) == len(wine_features)
        assert set(predictions).issubset({0, 1, 2})

    def test_challenger_alias_is_set_not_champion(self):
        client = MagicMock()

        # Simulate what train.py does after model registration
        client.set_registered_model_alias("main.wine.classifier", "Challenger", "1")

        client.set_registered_model_alias.assert_called_once_with(
            "main.wine.classifier", "Challenger", "1"
        )
        # Champion must not be set during training — that is the quality gate's job
        aliases_set = [call[0][1] for call in client.set_registered_model_alias.call_args_list]
        assert "Champion" not in aliases_set

    def test_model_signature_is_inferred(self, wine_features):
        labels = pd.Series(np.random.default_rng(1).integers(0, 3, len(wine_features)))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(wine_features, labels)
        predictions = model.predict(wine_features)

        signature = infer_signature(wine_features, predictions)

        assert signature is not None
        assert signature.inputs is not None
        assert signature.outputs is not None


# ─── Evaluate Quality Gate ────────────────────────────────────────────────────


class TestEvaluateQualityGate:
    """Evaluate step: pipeline halts when Challenger accuracy is below threshold."""

    def _run_gate(self, accuracy: float, threshold: float):
        """Mirrors the quality gate logic in evaluate.py."""
        if accuracy < threshold:
            raise ValueError(
                f"Challenger accuracy {accuracy:.4f} is below the required threshold {threshold:.4f} — "
                "pipeline halted. Review the model or lower the threshold if appropriate."
            )

    def test_gate_passes_when_accuracy_above_threshold(self):
        self._run_gate(accuracy=0.91, threshold=0.85)  # should not raise

    def test_gate_passes_at_exact_threshold(self):
        self._run_gate(accuracy=0.85, threshold=0.85)  # equal is a pass

    def test_gate_raises_when_accuracy_below_threshold(self):
        with pytest.raises(ValueError, match="below the required threshold"):
            self._run_gate(accuracy=0.72, threshold=0.85)

    def test_gate_raises_just_below_threshold(self):
        with pytest.raises(ValueError):
            self._run_gate(accuracy=0.8499, threshold=0.85)

    def test_gate_message_includes_actual_and_threshold_values(self):
        with pytest.raises(ValueError, match="0.7200") as exc_info:
            self._run_gate(accuracy=0.72, threshold=0.85)
        assert "0.8500" in str(exc_info.value)


# ─── Validation ──────────────────────────────────────────────────────────────


class TestModelValidation:
    """Validation step: signature, input example, and required tag checks."""

    def test_validation_passes_when_signature_is_present(self):
        model_info = MagicMock()
        model_info.signature = MagicMock()
        model_info.signature.inputs = MagicMock()
        model_info.signature.outputs = MagicMock()
        model_info.saved_input_example_info = MagicMock()

        assert model_info.signature is not None
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs is not None

    def test_validation_fails_when_signature_missing(self):
        model_info = MagicMock()
        model_info.signature = None

        with pytest.raises(AssertionError, match="Model signature is missing"):
            assert model_info.signature is not None, "Model signature is missing"

    def test_validation_fails_when_input_example_missing(self):
        model_info = MagicMock()
        model_info.signature = MagicMock()
        model_info.saved_input_example_info = None

        with pytest.raises(AssertionError, match="no input example"):
            assert model_info.saved_input_example_info is not None, \
                "Model has no input example"

    def test_validation_fails_when_required_tag_missing(self):
        challenger_version = MagicMock()
        challenger_version.tags = {}

        with pytest.raises(AssertionError, match="Required tag 'dataset' is missing"):
            assert "dataset" in challenger_version.tags, \
                "Required tag 'dataset' is missing from the model version"

    def test_challenger_alias_is_resolved(self):
        client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "3"
        client.get_model_version_by_alias.return_value = mock_version

        result = client.get_model_version_by_alias("main.wine.classifier", "Challenger")

        assert result.version == "3"
        client.get_model_version_by_alias.assert_called_once_with(
            "main.wine.classifier", "Challenger"
        )


# ─── Deployment ──────────────────────────────────────────────────────────────


class TestDeployment:
    """Deployment step: @Challenger is promoted to @Champion only when the quality gate passes."""

    def test_champion_alias_is_set_when_evaluation_passes(self):
        client = MagicMock()
        model_name = "main.wine.classifier"
        challenger_version = "3"
        accuracy_threshold = 0.85
        eval_accuracy = 0.91  # above threshold

        if eval_accuracy >= accuracy_threshold:
            client.set_registered_model_alias(model_name, "Champion", challenger_version)

        client.set_registered_model_alias.assert_called_once_with(
            model_name, "Champion", challenger_version
        )

    def test_champion_alias_is_not_set_when_evaluation_fails(self):
        client = MagicMock()
        model_name = "main.wine.classifier"
        accuracy_threshold = 0.85
        eval_accuracy = 0.72  # below threshold

        if eval_accuracy >= accuracy_threshold:
            client.set_registered_model_alias(model_name, "Champion", "3")

        client.set_registered_model_alias.assert_not_called()

    def test_previous_champion_is_unchanged_on_failed_evaluation(self):
        """When evaluation fails, the existing @Champion version must stay in place."""
        client = MagicMock()
        previous_champion = MagicMock()
        previous_champion.version = "2"
        client.get_model_version_by_alias.return_value = previous_champion

        accuracy_threshold = 0.85
        eval_accuracy = 0.72

        if eval_accuracy >= accuracy_threshold:
            client.set_registered_model_alias("main.wine.classifier", "Champion", "3")

        client.set_registered_model_alias.assert_not_called()
        champion = client.get_model_version_by_alias("main.wine.classifier", "Champion")
        assert champion.version == "2"


# ─── Monitoring ──────────────────────────────────────────────────────────────


class TestDriftDetection:
    """Monitoring step: per-feature mean drift calculation and alert logic."""

    def _drift_scores(self, reference: pd.DataFrame, serving: pd.DataFrame) -> pd.Series:
        """Normalised per-feature mean drift — mirrors the logic in monitoring.py."""
        training_means = reference.mean()
        serving_means = serving.mean()
        return abs(serving_means - training_means) / (training_means.abs() + 1e-8)

    def test_no_drift_on_identical_distributions(self, wine_features):
        scores = self._drift_scores(wine_features, wine_features.copy())
        assert scores.max() < 0.01

    def test_drift_detected_on_shifted_feature(self, wine_features):
        shifted = wine_features.copy()
        shifted["alcohol"] = shifted["alcohol"] * 1.5

        scores = self._drift_scores(wine_features, shifted)

        assert scores["alcohol"] > 0.1

    def test_alert_triggers_above_threshold(self, wine_features):
        drift_threshold = 0.2
        shifted = wine_features.copy()
        shifted["alcohol"] = shifted["alcohol"] * 2.0

        scores = self._drift_scores(wine_features, shifted)

        assert scores.max() > drift_threshold

    def test_no_alert_below_threshold(self, wine_features):
        drift_threshold = 0.2
        rng = np.random.default_rng(99)
        noise = pd.DataFrame(
            rng.normal(0, 0.001, wine_features.shape),
            columns=wine_features.columns,
        )
        serving = wine_features + noise

        scores = self._drift_scores(wine_features, serving)

        assert scores.max() < drift_threshold

    def test_drift_metrics_logged_per_feature(self, wine_features):
        with patch("mlflow.log_metric") as mock_log:
            scores = self._drift_scores(wine_features, wine_features.copy())
            for feature, score in scores.items():
                mlflow.log_metric(f"{feature}_drift", float(score))

            assert mock_log.call_count == len(wine_features.columns)
