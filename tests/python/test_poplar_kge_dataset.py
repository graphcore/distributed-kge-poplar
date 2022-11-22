# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import itertools as it
from typing import Tuple

import numpy as np
import poplar_kge as kge
import poplar_kge_dataset as kge_data
import pytest


def test_entity_features() -> None:
    data = kge_data.RawData.generate(
        n_entity=70,
        n_relation_type=100,
        feature_size=8,
        n_train=100,
        n_eval=50,
        seed=29483,
    )

    def check_features(mapping: str, feature_size: int) -> None:
        dataset = kge_data.Dataset(
            data,
            n_shard=4,
            train_steps_per_program_run=10,
            settings=kge.DataSettings(
                seed=9842,
                batch_size=16,
                a2a_size=10,
                entity_feature_mapping=mapping,
                dataset=None,  # type:ignore[arg-type]
                sampling_strategy=None,
            ),
        )
        features = dataset.entity_features(feature_size=feature_size)
        assert features.shape == (4, int(1 + np.ceil(70 / 4)), feature_size)
        assert features.dtype == np.float16

    check_features("zero", feature_size=6)
    check_features("full", feature_size=8)
    check_features("random_projection", feature_size=6)

    with pytest.raises(ValueError):
        check_features("full", feature_size=6)


def _get_triples(
    dataset: kge_data.Dataset, batch: kge.Batch
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the positive and negative (h, r, t) triples from a batch.

    returns -- (positive[:, {h,r,t}], negatives[:])
    """
    _ = np.newaxis  # a lot of fancy indexing

    idx_to_entity = np.full((dataset.n_shard, dataset.n_entity_per_shard), -1)
    idx_to_entity[dataset.entity_to_shard, dataset.entity_to_idx] = np.arange(
        dataset.data.n_entity
    )
    shards = np.arange(dataset.n_shard)
    n_batch = dataset.train_steps_per_program_run
    batches = np.arange(n_batch)

    remote_entity = idx_to_entity[shards[:, _, _], batch.remote]
    head_entity = remote_entity[shards[:, _, _], batches[_, :, _], batch.head]
    a2a_entity = (
        remote_entity[shards[:, _, _, _], batches[_, :, _, _], batch.a2a]
        .transpose(2, 1, 0, 3)
        .reshape(dataset.n_shard, n_batch, -1)
    )
    assert not len(a2a_entity[(a2a_entity < 0) | (a2a_entity >= dataset.data.n_entity)])
    tail_entity = a2a_entity[shards[:, _, _], batches[_, :, _], batch.tail]
    mask = np.full((dataset.n_shard, n_batch, a2a_entity.shape[-1]), True)
    mask[shards[:, _, _], batches[_, :, _], batch.tail] = False
    negative_tail_entity = a2a_entity[mask]
    return (
        np.stack([head_entity, batch.relation, tail_entity], axis=-1).reshape(-1, 3),
        negative_tail_entity.flatten(),
    )


def test_sample_batch() -> None:
    data = kge_data.RawData.generate(
        n_entity=70,
        n_relation_type=1024,
        feature_size=0,
        n_train=100,
        n_eval=50,
        seed=2835,
    )
    n_shard = 4
    n_batch = 11
    s = kge.DataSettings(
        seed=12039,
        batch_size=8,
        a2a_size=5,
        entity_feature_mapping="full",
        dataset=None,  # type:ignore[arg-type]
        sampling_strategy=kge.CubicRootRelationSampling(),
    )
    dataset = kge_data.Dataset(
        data, n_shard=n_shard, train_steps_per_program_run=n_batch, settings=s
    )

    # Check basic shapes
    batch = dataset.sample_batch()
    assert batch.remote.shape == (n_shard, n_batch, s.batch_size + n_shard * s.a2a_size)
    assert batch.a2a.shape == (n_shard, n_batch, n_shard, s.a2a_size)
    for section in [batch.head, batch.relation, batch.tail]:
        assert section.shape == (n_shard, n_batch, s.batch_size)
    assert {k: v.dtype.name for k, v in batch.__dict__.items()} == dict(
        remote="uint32",
        a2a="uint32",
        head="uint32",
        relation="uint32",
        tail="uint32",
    )

    # Check contiguous
    for k, v in batch.__dict__.items():
        assert v.flags.c_contiguous, f"batch.{k} is not C-contiguous"

    # Check invariants
    all_hrt = set(map(tuple, data.train_hrt))
    missed_hrt = all_hrt.copy()
    all_negative = set(np.arange(data.n_entity))
    missed_negative = all_negative.copy()
    for batch in it.islice(dataset.batches(), 100):
        positive, negative = _get_triples(dataset, batch)
        positive_hrt = set(map(tuple, positive))
        assert not positive_hrt - all_hrt
        missed_hrt -= positive_hrt
        assert not set(negative) - all_negative
        missed_negative -= set(negative)
    # Enough samples make these tests almost certain
    assert not missed_hrt, "unlikely to omit any positive (h,r,t)"
    assert not missed_negative, "unlikely to omit any negative tail entities"
