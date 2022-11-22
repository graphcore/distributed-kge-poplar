# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Dataset loading, preparation, batching for poplar_kge."""

import dataclasses
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import ogb.lsc
import poplar_kge as kge
import torch


@dataclasses.dataclass
class RawData:
    """Wraps ogb.lsc.WikiKG90Mv2Dataset to allow testing on fake data."""

    n_entity: int
    n_relation_type: int
    entity_features: np.ndarray  # float16[n_entity x feature_size]
    train_hrt: np.ndarray  # uint32[n_train x 3]
    eval_hr_: Dict[str, np.ndarray]  # {str: uint[n_train x (2 or 3)]}

    @classmethod
    def generate(
        cls,
        n_entity: int,
        n_relation_type: int,
        feature_size: int,
        n_train: int,
        n_eval: int,
        seed: int,
    ) -> "RawData":
        random = np.random.RandomState(seed)
        # There is no generalisation is this random world
        hrt = {
            key: np.stack(
                [
                    random.randint(n_entity, size=n),
                    random.randint(n_relation_type, size=n),
                    random.randint(n_entity, size=n),
                ],
                axis=-1,
            ).astype(np.uint32)
            for key, n in {
                "train": n_train,
                "valid": n_eval,
                "test-dev": n_eval,
                "test-challenge": n_eval,
            }.items()
        }
        return cls(
            n_entity=n_entity,
            n_relation_type=n_relation_type,
            entity_features=random.randn(n_entity, feature_size).astype(np.float16),
            train_hrt=hrt["train"],
            eval_hr_=dict(
                train=hrt["train"][random.choice(n_train, size=n_eval, replace=False)],
                **{k: hrt[k] for k in ["valid", "test-dev", "test-challenge"]},
            ),
        )

    @classmethod
    def load_wikikg90mv2(
        cls, path: Path, seed: int, entity_limit: int = 1 << 63
    ) -> "RawData":
        data = ogb.lsc.WikiKG90Mv2Dataset(path)
        n_entity = data.num_entities
        train_hrt = data.train_hrt.astype(np.uint32)
        eval_hr_ = {}
        eval_hr_["valid"] = np.concatenate(
            [
                data.valid_dict["h,r->t"]["hr"],
                data.valid_dict["h,r->t"]["t"][:, np.newaxis],
            ],
            axis=1,
        ).astype(np.uint32)
        entity_features = data.entity_feat

        if entity_limit < n_entity:
            # Select a subset of entities (total <= entity_limit)
            entity_mask = np.full(data.num_entities, False)
            # Most common heads
            entity_mask[
                np.argsort(np.bincount(data.train_hrt[:, 0]))[-entity_limit // 2 :]
            ] = True
            # Most common tails
            entity_mask[
                np.argsort(np.bincount(data.train_hrt[:, 2]))[-entity_limit // 2 :]
            ] = True

            # Truncate & re-map dataset
            n_entity = np.sum(entity_mask)
            old_to_new_entity = np.full(data.num_entities, -1)
            old_to_new_entity[np.where(entity_mask)] = np.arange(n_entity)
            train_hrt = train_hrt[
                entity_mask[train_hrt[:, 0]] & entity_mask[train_hrt[:, 2]]
            ]
            train_hrt[:, 0] = old_to_new_entity[train_hrt[:, 0]]
            train_hrt[:, 2] = old_to_new_entity[train_hrt[:, 2]]
            eval_hr_["valid"] = eval_hr_["valid"][
                entity_mask[eval_hr_["valid"][:, 0]]
                & entity_mask[eval_hr_["valid"][:, 2]]
            ]
            eval_hr_["valid"][:, 0] = old_to_new_entity[eval_hr_["valid"][:, 0]]
            eval_hr_["valid"][:, 2] = old_to_new_entity[eval_hr_["valid"][:, 2]]
            entity_features = entity_features[np.where(entity_mask)]
        else:
            # Only add 'test' sets when entities haven't been truncated/remapped
            for name in ["test-dev", "test-challenge"]:
                eval_hr_[name] = data.test_dict(name)["h,r->t"]["hr"].astype(np.uint32)

        # After train_hrt has been truncated/remapped
        eval_hr_["train"] = train_hrt[
            np.random.RandomState(seed).choice(train_hrt.shape[0], size=15000)
        ]

        # Note: test first so that .astype() doesn't collapse the memmap
        if entity_features.dtype != np.float16:
            entity_features = entity_features.astype(np.float16)
        return cls(
            n_entity=n_entity,
            n_relation_type=data.num_relations,
            entity_features=entity_features,
            train_hrt=train_hrt,
            eval_hr_=eval_hr_,
        )

    @classmethod
    def load(cls, settings: kge.Settings) -> "RawData":
        if isinstance(settings.data.dataset, kge.WikiKg90Mv2Settings):
            path = Path(
                os.environ.get("OGBWIKIKG_PATH", "/localdata/research/datasets/ogb/lsc")
            )
            if not (path / "wikikg90m-v2").exists():
                raise ValueError(
                    f"Dataset 'wikikg90mv2' was not found at {path}.\n"
                    f"  On the farm, try:  mkdir -p {path}"
                    " && rsync -a --info=progress2 --chmod=D0770,F660"
                    f" /home/research-datasets/ogb/lsc/wikikg90m-v2/ {path}/wikikg90m-v2/"
                )
            return cls.load_wikikg90mv2(
                path,
                seed=settings.data.seed,
                entity_limit=settings.model.n_shard * (settings.model.n_entity - 1),
            )
        if isinstance(settings.data.dataset, kge.GeneratedDataSettings):
            return cls.generate(
                n_entity=(settings.model.n_entity - 1) * settings.model.n_shard - 1,
                n_relation_type=settings.model.n_relation_type,
                feature_size=settings.model.entity_feature_size,
                n_train=settings.data.dataset.n_train,
                n_eval=settings.data.dataset.n_eval,
                seed=settings.data.dataset.seed,
            )
        raise ValueError(f"Unknown dataset '{settings.data.dataset}'")


def unique_pad(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A zero-padded version of `np.unique(idx, return_inverse=True)`."""
    unique, inverse = np.unique(idx, return_inverse=True)
    return np.pad(unique, ((0, len(idx) - len(unique)),)), inverse.astype(np.uint32)


# Dataset


@dataclasses.dataclass
class Dataset:
    def __init__(
        self,
        data: RawData,
        n_shard: int,
        train_steps_per_program_run: int,
        settings: kge.DataSettings,
    ):
        self.data = data
        self.n_shard = n_shard
        self.train_steps_per_program_run = train_steps_per_program_run
        self.settings = settings

        # Partition entities into shards
        self.random = np.random.RandomState(settings.seed)
        self.entity_to_idx, self.entity_to_shard = np.divmod(
            self.random.permutation(data.n_entity), n_shard
        )
        self.entity_to_idx += 1  # index zero is for padding
        self.shard_to_count = np.bincount(self.entity_to_shard, minlength=n_shard)
        self.n_entity_per_shard = 1 + np.max(self.shard_to_count)
        self.shard_idx_to_entity = np.full(
            (n_shard, self.n_entity_per_shard), 1 << 31, dtype=np.uint32
        )
        self.shard_idx_to_entity[self.entity_to_shard, self.entity_to_idx] = np.arange(
            data.n_entity
        )

        # Indexed by shardpair=(head_shard, tail_shard) for sampling
        triple_to_head_shard = self.entity_to_shard[data.train_hrt[:, 0]]
        triple_to_tail_shard = self.entity_to_shard[data.train_hrt[:, 2]]

        if isinstance(self.settings.sampling_strategy, kge.CubicRootRelationSampling):
            triple_to_shardpair = (
                n_shard * data.n_relation_type * triple_to_head_shard
                + data.n_relation_type * triple_to_tail_shard
                + data.train_hrt[:, 1]
            )
            shardpair_shape = (n_shard * n_shard, data.n_relation_type)
            sort_idx = np.argsort(triple_to_shardpair)
        else:
            triple_to_shardpair = n_shard * triple_to_head_shard + triple_to_tail_shard
            shardpair_shape = (n_shard, n_shard)
            assert (
                n_shard <= 16
            ), f"cannot use uint8 to speed up sorting when n_shard ({n_shard}) > 16"
            sort_idx = np.argsort(triple_to_shardpair.astype(np.uint8))

        self.shardpair_to_count = np.bincount(
            triple_to_shardpair,
            minlength=np.prod(shardpair_shape),
        ).reshape(shardpair_shape)
        self.shardpair_to_offset = np.concatenate(
            [[0], np.cumsum(self.shardpair_to_count)[:-1]]
        ).reshape(shardpair_shape)

        if isinstance(self.settings.sampling_strategy, kge.CubicRootRelationSampling):
            count_root = np.cbrt(self.shardpair_to_count)
            self.shardpair_sample_prob = count_root / np.sum(
                count_root, axis=-1, keepdims=True
            )

        train_hrt_sorted = data.train_hrt[sort_idx]
        self.shardpair_to_flat_hrt = np.stack(
            [
                self.entity_to_idx[train_hrt_sorted[:, 0]],
                train_hrt_sorted[:, 1],
                self.entity_to_idx[train_hrt_sorted[:, 2]],
            ],
            axis=0,
        ).astype(np.uint32)

        # Derived hyperparameters
        self.entity_projection_seed = self.random.randint(1 << 32)
        if settings.batch_size % n_shard != 0:
            raise ValueError(
                f"Expected batch_size ({settings.batch_size}) to be a multiple of n_shard ({n_shard})"
            )
        self.positives_per_shardpair = settings.batch_size // n_shard
        if settings.a2a_size <= self.positives_per_shardpair:
            raise ValueError(
                f"Expected a2a_size ({settings.a2a_size}) to be >= batch_size/n_shard ({self.positives_per_shardpair})"
            )
        self.negatives_per_shardpair = settings.a2a_size - self.positives_per_shardpair

        # Precomute 'tail', possible since batch positives are always in the same place
        tail = (
            np.tile(np.arange(self.positives_per_shardpair), (n_shard, n_shard, 1))
            + (settings.a2a_size * np.arange(n_shard))[np.newaxis, :, np.newaxis]
        ).reshape(n_shard, settings.batch_size)
        self.pos_tail_idx = np.tile(
            tail[:, np.newaxis, :], (1, train_steps_per_program_run, 1)
        ).astype(np.uint32)

    def _sharded_entity_features(self, features: np.ndarray) -> np.ndarray:
        result = np.zeros(
            (self.n_shard, self.n_entity_per_shard, features.shape[-1]),
            dtype=features.dtype,
        )
        result[self.entity_to_shard, self.entity_to_idx, :] = features
        return result

    @classmethod
    def load(cls, data: RawData, settings: kge.Settings) -> "Dataset":
        return cls(
            data,
            n_shard=settings.model.n_shard,
            train_steps_per_program_run=settings.execution.train_steps_per_program_run,
            settings=settings.data,
        )

    def entity_features(self, feature_size: int) -> np.ndarray:
        mapping = self.settings.entity_feature_mapping
        if mapping == "zero":
            return np.zeros(
                (self.n_shard, self.n_entity_per_shard, feature_size), dtype=np.float16
            )

        if mapping == "full" or mapping == "random_projection":
            data_feature_size = self.data.entity_features.shape[-1]
            if mapping == "full" and feature_size != data_feature_size:
                raise ValueError(
                    "Entity_feature_mapping 'full' requires dataset "
                    f"feature_size ({data_feature_size}) == model feature_size ({feature_size})"
                )
            entity_features = self.data.entity_features
            if mapping == "random_projection":
                # Note - CPU float16 is slow, it might be faster to do this in float32
                projection = np.random.RandomState(self.entity_projection_seed).randn(
                    data_feature_size, feature_size
                ).astype(np.float16) / np.sqrt(data_feature_size)
                entity_features = entity_features @ projection

            return self._sharded_entity_features(entity_features)

        raise ValueError(f"Unknown entity_feature_mapping: {mapping}")

    # Training

    def sample_batch(self) -> kge.Batch:
        n_shard = self.n_shard
        n_batch = self.train_steps_per_program_run
        batch_size = self.settings.batch_size

        # Sample positive & negative entity indices
        if isinstance(self.settings.sampling_strategy, kge.CubicRootRelationSampling):
            num_samples = self.positives_per_shardpair * n_batch
            shardpair_relat = np.stack(
                [
                    self.random.choice(
                        self.data.n_relation_type, size=num_samples, p=prob
                    )
                    for prob in self.shardpair_sample_prob
                ]
            )
            shardpair_offset = self.shardpair_to_offset[
                np.arange(n_shard * n_shard)[:, np.newaxis], shardpair_relat
            ].reshape(n_shard, n_shard, -1)
            shardpair_count = self.shardpair_to_count[
                np.arange(n_shard * n_shard)[:, np.newaxis], shardpair_relat
            ].reshape(n_shard, n_shard, -1)
            sample_idx = (
                (
                    shardpair_offset
                    + self.random.randint(
                        1 << 63,
                        size=(n_shard, n_shard, n_batch * self.positives_per_shardpair),
                    )
                    % shardpair_count
                )
                .reshape(n_shard, n_shard, n_batch, -1)
                .transpose(0, 2, 1, 3)
            )
        else:
            sample_idx = (
                self.shardpair_to_offset[:, np.newaxis, :, np.newaxis]
                + self.random.randint(
                    1 << 63,
                    size=(n_shard, n_batch, n_shard, self.positives_per_shardpair),
                )
                % self.shardpair_to_count[:, np.newaxis, :, np.newaxis]
            )

        h, r, t = self.shardpair_to_flat_hrt[:, sample_idx]
        t_negative = 1 + (
            self.random.randint(
                1 << 63, size=(n_shard, n_batch, n_shard, self.negatives_per_shardpair)
            )
            % self.shard_to_count[np.newaxis, np.newaxis, :, np.newaxis]
        ).astype(np.uint32)

        # Remote indices, deduplication & routing indices
        remote_duplicated = np.concatenate(
            [
                h.reshape(n_shard * n_batch, -1),
                t.transpose(2, 1, 0, 3).reshape(n_shard * n_batch, -1),
                t_negative.transpose(2, 1, 0, 3).reshape(n_shard * n_batch, -1),
            ],
            axis=1,
        )
        remote, gather = map(np.stack, zip(*[unique_pad(a) for a in remote_duplicated]))
        remote = remote.reshape(n_shard, n_batch, -1)
        head = gather[:, :batch_size].reshape(n_shard, n_batch, batch_size)
        a2a = np.concatenate(
            [
                gather[:, batch_size : 2 * batch_size].reshape(
                    n_shard, n_batch, n_shard, self.positives_per_shardpair
                ),
                gather[:, 2 * batch_size :].reshape(
                    n_shard, n_batch, n_shard, self.negatives_per_shardpair
                ),
            ],
            axis=3,
        )
        relation = r.reshape(n_shard, n_batch, batch_size)

        return kge.Batch(
            remote=remote,
            head=np.ascontiguousarray(head),
            relation=np.ascontiguousarray(relation),
            a2a=a2a,
            tail=self.pos_tail_idx,
        )

    def batches(self) -> Iterable[kge.Batch]:
        while True:
            yield self.sample_batch()

    # Evaluation

    def predict(
        self, part: str, predict_fn: Callable[[np.ndarray], kge.Predictions]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictions (in plain entities) for an evaluation set.

        predictions = predict_fn(shard_idx_relation)
            shard_idx_relation: uint32[n x 3] -- (head.shard, head.index, head.relation)
            predictions: kge.Predictions[n x n_best]

        Returns: (tail, score) over all (h,r,?) in `eval_hr_[part]`.
            tail: uint32[n x n_best]
            score: float[n x n_best]
        """
        head = self.data.eval_hr_[part][:, 0]
        relation = self.data.eval_hr_[part][:, 1]
        predictions = predict_fn(
            np.stack(
                [self.entity_to_shard[head], self.entity_to_idx[head], relation], axis=1
            )
        )
        tail = self.shard_idx_to_entity[
            predictions.shard_idx[..., 0], predictions.shard_idx[..., 1]
        ]
        return (tail, predictions.score)

    def mrr(
        self, part: str, predict_fn: Callable[[np.ndarray], kge.Predictions]
    ) -> float:
        """Compute Mean Reciprocal Rank (mrr) for a labelled evaluation set."""
        true_tails = self.data.eval_hr_[part][:, 2].astype(np.int32)
        predicted_tails = self.predict(part, predict_fn)[0][:, :10].astype(np.int32)
        return ogb.lsc.WikiKG90Mv2Evaluator().eval(  # type:ignore[no-any-return]
            {"h,r->t": dict(t=true_tails, t_pred_top10=predicted_tails)}
        )["mrr"]


class DatasetWrapper(torch.utils.data.Dataset[Dict[str, np.ndarray]]):
    def __init__(self, dataset: Dataset) -> None:
        self.ds = dataset

    def __len__(self) -> int:
        return (2**31) - 1

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        sample_batch = self.ds.sample_batch()
        return {
            "remote": sample_batch.remote.astype(np.int32),
            "a2a": sample_batch.a2a.astype(np.int32),
            "head": sample_batch.head.astype(np.int32),
            "relation": sample_batch.relation.astype(np.int32),
            "tail": sample_batch.tail.astype(np.int32),
        }

    @staticmethod
    def tensors_to_batch(
        remote: torch.Tensor,
        a2a: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> kge.Batch:
        return kge.Batch(
            remote=remote.numpy().astype(np.uint32),
            head=head.numpy().astype(np.uint32),
            relation=relation.numpy().astype(np.uint32),
            a2a=a2a.numpy().astype(np.uint32),
            tail=tail.numpy().astype(np.uint32),
        )

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        worker_info = torch.utils.data.get_worker_info()  # type:ignore[no-untyped-call]
        dataset_unwrapped = worker_info.dataset.ds
        worked_seed = dataset_unwrapped.settings.seed + worker_id
        dataset_unwrapped.random = np.random.RandomState(worked_seed)
