# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Methods for ensembling ranked predictions."""

from typing import Optional

import numpy as np


def mean_ensemble(
    predictions: np.ndarray,
    count: int,
    power: float,
    default_rank: Optional[float] = None,
    model_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sort entities by mean(score), where score is -sign(power) * rank ** power.

    predictions -- uint[n_model x n_example x n_prediction]

    count -- number of predictions to return, must be <= n_prediction

    default_rank -- rank to assume when a model is not found
                    default (power < 0) = infinity
                    default (power > 0) = 1 + n_prediction

    returns -- uint[n_model x n_example x count]
    """
    n_model, n_example, n_prediction = predictions.shape

    if power == 0.0:
        default_score = -1.0
    elif default_rank is not None:
        default_score = -np.sign(power) * default_rank**power
    elif power < 0:
        default_score = 0.0
    elif power > 0:
        default_score = -((1 + n_prediction) ** power)

    if model_weight is not None and default_score != 0:
        raise NotImplementedError(
            "Model weighting only implemented for default_rank = inf."
        )

    rank_scores = (
        -np.sign(power) * (1 + np.arange(n_prediction, dtype=np.float32)) ** power
    )
    rank_scores = np.tile(rank_scores[np.newaxis, :], (n_model, 1))
    if model_weight is not None:
        rank_scores *= model_weight[:, np.newaxis]

    results = []
    for idx in range(n_example):
        ids, indices, counts = np.unique(
            predictions[:, idx, :].reshape(-1), return_inverse=True, return_counts=True
        )
        scores = (n_prediction - counts) * default_score
        np.add.at(scores, indices, rank_scores.reshape(-1))
        results.append(ids[np.argsort(scores)[::-1][:count]])
    return np.stack(results)


def median_ensemble(predictions: np.ndarray, count: int) -> np.ndarray:
    """Sort entities by median(1/rank).

    predictions -- uint[n_model x n_example x n_prediction]

    count -- number of predictions to return, must be <= n_prediction

    returns -- uint[n_model x n_example x count]
    """
    n_model, n_example, n_prediction = predictions.shape
    rrank = np.tile(1 + np.arange(n_prediction, dtype=np.float32), (n_model, 1)) ** -1

    results = []
    for idx in range(n_example):
        ids, indices = np.unique(
            predictions[:, idx, :].reshape(-1), return_inverse=True
        )
        padded_rrank = np.zeros((n_model, len(ids)), dtype=np.float32)
        padded_rrank[
            np.arange(n_model)[:, np.newaxis], indices.reshape(n_model, n_prediction)
        ] = rrank
        scores = np.median(padded_rrank, axis=0)
        results.append(ids[np.argsort(scores)[::-1][:count]])
    return np.stack(results)
