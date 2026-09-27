import inspect

import numpy as np
from sklearn.model_selection import GroupKFold

import ml.training.train as train
from evaluation.run_all import nasa_score


def test_training_never_loads_the_official_test_set():
    # Model choices must be made on training engines only; the test set is used
    # once, by evaluation/run_all.py.
    assert "load_test" not in inspect.getsource(train)


def test_cross_validation_is_engine_wise():
    groups = np.repeat(np.arange(10), 20)
    X = np.zeros((len(groups), 1))
    for tr, va in GroupKFold(n_splits=train.N_FOLDS).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[va])


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score([50], [50]) == 0
    late = nasa_score([50], [60])   # predicted too much life left
    early = nasa_score([50], [40])
    assert late == np.exp(1) - 1 and early == np.exp(10 / 13) - 1
    assert late > early
