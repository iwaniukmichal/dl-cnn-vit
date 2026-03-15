import numpy as np

from archdyn.evaluation.metrics import classification_metrics, distance_ratio


def test_classification_metrics() -> None:
    metrics = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_distance_ratio_positive() -> None:
    embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
    labels = np.array([0, 0, 1, 1])
    assert distance_ratio(embeddings, labels) > 1.0
