import numpy as np

from boosting_glm.metrics import model_metrics, poisson_deviance


def test_poisson_deviance_is_zero_for_perfect_prediction():
    observed = np.array([0.0, 1.0, 2.0, 0.0])
    assert poisson_deviance(observed, observed) < 1e-9


def test_model_metrics_normalizes_against_reference():
    observed = np.array([0.0, 1.0, 0.0, 2.0])
    table = model_metrics(observed, {"gbm": np.array([0.1, 0.9, 0.2, 1.8]), "final": np.array([0.2, 0.8, 0.1, 1.9])})
    reference = table.set_index("model").loc["gbm"]
    assert reference["relative_mse"] == 1.0
    assert reference["relative_deviance"] == 1.0

