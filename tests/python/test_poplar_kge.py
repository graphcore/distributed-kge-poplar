# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import itertools as it
from functools import partial

import numpy as np
import poplar_kge as kge
import poplar_kge_dataset as kge_data
import pytest
import reference_model

# Utilities


def test_predictions_topk() -> None:
    pre = kge.Predictions(
        shard_idx=np.array(
            [
                [[10, 100], [20, 200], [30, 300], [40, 400]],
                [[50, 500], [60, 600], [70, 700], [80, 800]],
            ]
        ),
        score=np.array([[2, 1, 3, 0], [-4, -3, -2, -1]]),
    )
    post = pre.topk(k=3).sort()
    assert post.shard_idx.shape == (2, 3, 2)
    assert post.score.shape == (2, 3)
    np.testing.assert_equal(
        post.shard_idx,
        [
            [[30, 300], [10, 100], [20, 200]],
            [[80, 800], [70, 700], [60, 600]],
        ],
    )
    np.testing.assert_equal(post.score, [[3, 2, 1], [-1, -2, -3]])

    multidim = (
        kge.Predictions(pre.shard_idx[:, np.newaxis], pre.score[:, np.newaxis])
        .topk(k=3)
        .sort()
    )
    assert multidim.shard_idx.shape == (2, 1, 3, 2)
    assert multidim.score.shape == (2, 1, 3)
    np.testing.assert_equal(multidim.shard_idx[:, 0], post.shard_idx)
    np.testing.assert_equal(multidim.score[:, 0], post.score)


# Core


def _random_batch(engine: kge.Engine) -> kge.Batch:
    s = engine.settings
    n_step = s.execution.train_steps_per_program_run
    n_shard = s.model.n_shard
    bshape = (n_shard, n_step)

    # Note: the zero entity index is reserved and remote indices must not repeat
    # (within a shard)
    def sample_remote() -> np.ndarray:
        # We'd like to use random.choice(..., replace=False), but it is very slow
        # when choosing from a large set, so we use with-replacement then unique
        sample: np.ndarray = np.unique(
            engine.random.choice(
                np.arange(1, s.model.n_entity),
                size=s.n_remote * 3,
                replace=True,
            )
        )
        engine.random.shuffle(sample)
        sample = np.sort(sample[: s.n_remote])
        assert sample.size == s.n_remote
        return sample

    return kge.Batch(
        remote=(
            np.stack([sample_remote() for _ in range(n_shard * n_step)])
            .reshape(bshape + (s.n_remote,))
            .astype(np.uint32)
        ),
        a2a=engine.random_integers(s.n_remote, bshape + (n_shard, s.data.a2a_size)),
        head=engine.random_integers(s.n_remote, bshape + (s.data.batch_size,)),
        relation=engine.random_integers(
            s.model.n_relation_type, bshape + (s.data.batch_size,)
        ),
        tail=engine.random_integers(s.n_tail, bshape + (s.data.batch_size,)),
    )


def _random_prediction_query(engine: kge.Engine, n: int) -> np.ndarray:
    s = engine.settings
    return np.stack(
        [
            engine.random_integers(s.model.n_shard, (n,)),
            1 + engine.random_integers(s.model.n_entity - 1, (n,)),
            engine.random_integers(s.model.n_relation_type, (n,)),
        ],
        axis=-1,
    )


def _random_features(engine: kge.Engine) -> np.ndarray:
    s = engine.settings
    features = engine.random_normal(
        (s.model.n_shard, s.model.n_entity, s.model.entity_feature_size),
        np.float16,
    )
    features[:, 0, :] = 0  # this will be zeroed in any case
    return features


def test_engine_ops() -> None:
    # Note - we write a lot of stuff as a single test, to make it faster to run
    # (avoid recompiling)
    settings = kge.Settings.create_demo()
    settings.seed = 100
    settings.training.learning_rate = 0.01
    settings.prepare()

    engine = kge.Engine(
        settings, np.full(settings.model.n_shard, settings.model.n_entity - 1)
    )

    # ## Train
    # Check we can initialise & take training steps
    random_features = _random_features(engine)
    engine.initialise_all(random_features)

    random_batch = _random_batch(engine)
    loss_0, _ = engine.train_step_loop(random_batch)
    loss_1, _ = engine.train_step_loop(random_batch)
    assert loss_1 < loss_0

    # ## Read/write SRAM
    # Check we can write an SRAM parameter & get it back
    new_relation_embedding = engine.random_normal(
        engine.variable_to_shape["relation_embedding"]
    )
    assert not np.allclose(
        engine.read_variable("relation_embedding"), new_relation_embedding
    )
    engine.write_variable("relation_embedding", new_relation_embedding)
    np.testing.assert_allclose(
        engine.read_variable("relation_embedding"), new_relation_embedding
    )

    # ## Read DRAM
    # Check we can get our fake features back out of entity_data
    np.testing.assert_allclose(
        engine.read_entity_all()[:, :, -settings.model.entity_feature_size :],
        random_features,
    )

    # ## Predict
    # Check we can make some predictions
    n_predict = engine.settings.execution.predict_hr_batch_size + 5
    predictions = engine.predict(_random_prediction_query(engine, n_predict))
    assert predictions.shard_idx.shape == (
        n_predict,
        settings.execution.predict_n_best,
        2,
    )
    assert np.all(
        np.isin(predictions.shard_idx[:, :, 0], np.arange(settings.model.n_shard))
    )
    assert np.all(
        np.isin(predictions.shard_idx[:, :, 1], np.arange(1, settings.model.n_entity))
    )
    assert predictions.score.shape == (n_predict, settings.execution.predict_n_best)
    assert not np.any(np.isnan(predictions.score))
    np.testing.assert_equal(
        predictions.score, np.sort(predictions.score, axis=-1)[..., ::-1]
    )


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_training(dtype: str) -> None:
    settings = kge.Settings.create_demo()
    settings.seed = 100
    settings.execution.dtype = dtype
    if dtype == "float16":
        settings.training.n_step = 20  # since numerical errors compound
        settings.execution.device = "ipu"
    settings.prepare()

    data = kge_data.RawData.load(settings)
    dataset = kge_data.Dataset.load(data, settings)
    try:
        engine = kge.Engine(settings, dataset.shard_to_count)
    except RuntimeError as e:
        if "Could not attach" in str(e):
            pytest.skip("IPU not available")
        raise
    engine.initialise_all(dataset.entity_features(settings.model.entity_feature_size))
    initial_state = engine.read_all()
    reference = reference_model.Model(settings)
    reference.set_state(initial_state)

    # A difference between reference & poplar_kge is that
    # entity embeddings in poplar_kge are only updated when they are used by
    # the model, so for this test we keep 'remote' indices the same for every batch
    batches = [_random_batch(engine), _random_batch(engine), _random_batch(engine)]
    for batch in batches:
        batch.remote[...] = batches[0].remote[:, 0, np.newaxis, :]

    n_loop = settings.program_runs_per_log * settings.logs_per_training_run

    loss = []
    reference_loss = []
    mrr = []
    reference_mrr = []
    for batch in it.islice(it.cycle(batches), n_loop):
        loss.append(engine.train_step_loop(batch)[0])
        reference_loss.append(reference.train_step_loop(batch))
        mrr.append(dataset.mrr("valid", engine.predict))
        reference_mrr.append(
            dataset.mrr(
                "valid",
                partial(reference.predict, shard_to_count=dataset.shard_to_count),
            )
        )

    # Loss check
    np.testing.assert_allclose(
        loss, reference_loss, rtol=dict(float16=2e-3, float32=1e-4)[dtype]
    )

    # MRR check
    np.testing.assert_allclose(
        mrr, reference_mrr, rtol=dict(float16=2e-2, float32=1e-4)[dtype]
    )

    # Rough final value check
    final_state = engine.read_all()
    reference_final_state = reference.get_state()
    final_state.pop("step")
    assert set(final_state.keys()) == set(reference_final_state.keys())

    for key in final_state:
        # These optimiser states have particularly high error in float16 (cause unknown)
        if dtype == "float16" and key.endswith("/adam_m"):
            continue

        delta = final_state[key] - initial_state[key]
        reference_delta = reference_final_state[key] - initial_state[key]
        if key == "entity_data":
            delta = delta[:, 1:]
            reference_delta = reference_delta[:, 1:]
        # Euclidean distance, normalised by the reference
        distance = np.sqrt(
            np.sum((delta - reference_delta) ** 2) / np.sum(reference_delta**2)
        )
        tolerance = dict(float16=1e-1, float32=1e-4)[dtype]
        assert distance <= tolerance, (
            "Distance between (final - initial) and reference_(final - initial)"
            f" for parameter '{key}'"
            f" is {distance:.2g} > tolerance {tolerance:.2g}"
            f"\n\n(final - initial) = {np.array2string(delta, threshold=20)}"
            f"\n\nreference_(final - initial) = {np.array2string(reference_delta, threshold=20)}"
        )
