from boosting_glm.data import split_dataset, synthetic_portfolio


def test_split_matches_original_proportions():
    frame, spec = synthetic_portfolio(1_000)
    split = split_dataset(frame, spec)
    assert len(split.train) == 560
    assert len(split.calibration) == 240
    assert len(split.test) == 200

