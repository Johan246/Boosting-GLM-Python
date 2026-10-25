import numpy as np

from boosting_glm.data import synthetic_portfolio
from boosting_glm.model import BoostedGLM


def test_model_fits_predicts_and_round_trips(tmp_path):
    frame, spec = synthetic_portfolio(700)
    X = frame[list(spec.features)]
    model = BoostedGLM(
        categorical_features=spec.categorical,
        max_iter=20,
        max_effect_grid=6,
        max_interactions=1,
        effect_sample=150,
    ).fit(X, frame["frequency"], frame["exposure"])
    prediction = model.predict(X.head(20), frame["exposure"].head(20))
    assert prediction.shape == (20,)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0)

    path = tmp_path / "model.joblib"
    model.save(path)
    restored = BoostedGLM.load(path)
    np.testing.assert_allclose(prediction, restored.predict(X.head(20), frame["exposure"].head(20)))

