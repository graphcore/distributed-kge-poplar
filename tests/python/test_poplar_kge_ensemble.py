# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Any, Callable, Dict

import numpy as np
import poplar_kge_ensemble as kge_ensemble
import pytest


@pytest.mark.parametrize(
    "ensemble_fn,ensemble_args",
    [
        (kge_ensemble.mean_ensemble, dict(power=-1)),
        (kge_ensemble.mean_ensemble, dict(power=-0.5)),
        (kge_ensemble.mean_ensemble, dict(power=1)),
        (kge_ensemble.mean_ensemble, dict(power=0)),
        (kge_ensemble.mean_ensemble, dict(power=-1, model_weight=np.arange(27))),
        (kge_ensemble.median_ensemble, dict()),
    ],
    ids=str,
)
def test_ensemble(
    ensemble_fn: Callable[..., np.ndarray], ensemble_args: Dict[str, Any]
) -> None:
    random = np.random.default_rng(100)

    n_model = 27
    n_example = 23
    n_prediction = 30
    n_entity = 100

    ground_truth = random.integers(n_entity, size=n_example)
    scores = random.random((n_model, n_example, n_entity))
    scores[:, np.arange(n_example), ground_truth] += 0.25
    predictions = np.argsort(scores, axis=2)[..., ::-1][:, :, :n_prediction]

    ensemble_predictions = ensemble_fn(predictions, count=n_prediction, **ensemble_args)

    _, _, ranks = np.where(predictions == ground_truth[:, np.newaxis])
    model_mrr = np.mean(1 / (1 + ranks))

    _, ranks = np.where(ensemble_predictions == ground_truth[:, np.newaxis])
    ensemble_mrr = np.mean(1 / (1 + ranks))

    # print(ensemble_fn.__name__, ensemble_args, model_mrr, ensemble_mrr)
    assert model_mrr < ensemble_mrr
