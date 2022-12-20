# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Modelling and distributed training code for ogbl-wikikg2 on IPU.

Implements BESS, see https://arxiv.org/abs/2211.12281 for more information.
"""

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import ogb.linkproppred
import poptorch
import torch as T

# Data, sharding, batching


@dataclass
class Dataset:
    """Represents a complete knowledge graph dataset of (head, relation, tail) triples."""

    n_entity: int
    n_relation_type: int
    triples: Dict[str, np.ndarray]  # {"train|valid": int64[n_triple, {h,r,t}]}

    @classmethod
    def build_wikikg2(cls, root: Path, out: Path, seed: int) -> None:
        """Build the OGB dataset into a simple .npz file, for faster loading."""
        data = ogb.linkproppred.LinkPropPredDataset("ogbl-wikikg2", root=root)
        split = data.get_edge_split()
        parts = {}
        random = np.random.default_rng(seed)
        for part in ["train", "valid"]:
            hrt = split[part]
            parts[part] = np.stack([hrt["head"], hrt["relation"], hrt["tail"]], axis=-1)
            random.shuffle(parts[part])
        np.savez(
            out,
            n_entity=int(data[0]["num_nodes"]),
            n_relation_type=int(1 + np.max(data.graph["edge_reltype"])),
            **{f"part_{k}": v for k, v in parts.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        """Load a dataset from an .npz file saved by `Dataset.build_wikikg2`."""
        data = np.load(path)
        return cls(
            n_entity=int(data["n_entity"]),
            n_relation_type=int(data["n_relation_type"]),
            triples={
                k.replace("part_", ""): data[k] for k in data if k.startswith("part_")
            },
        )

    def sample(self, part: str, n: int, seed: Optional[int]) -> np.ndarray:
        """Draw a random sample of triples, without replacement."""
        triples = self.triples[part]
        idx = np.random.default_rng(seed).choice(
            triples.shape[0], size=n, replace=False
        )
        return triples[idx]


@dataclass
class Sharding:
    """A mapping of entities to shards (and back again).

    entity_to_shard -- int64[n_entity] -- maps entity ID to shard index

    entity_to_idx -- int64[n_entity] -- maps entity ID to index within its shard

    shard_and_idx_to_entity -- int64[n_shard, max_entity_per_shard]
                            -- maps [shard, idx] to entity ID
    """

    n_shard: int
    entity_to_shard: np.ndarray
    entity_to_idx: np.ndarray
    shard_and_idx_to_entity: np.ndarray

    @property
    def n_entity(self) -> int:
        return len(self.entity_to_shard)

    @property
    def max_entity_per_shard(self) -> int:
        return self.shard_and_idx_to_entity.shape[1]

    @classmethod
    def create(cls, n_entity: int, n_shard: int, seed: Optional[int]) -> "Sharding":
        """Construct a random balanced assignment of entities to shards."""
        # Randomly shard entities
        entity_to_idx, entity_to_shard = np.divmod(
            np.random.default_rng(seed).permutation(n_entity), n_shard
        )
        # Build a reverse mapping back to entities
        shard_and_idx_to_entity = np.zeros(
            (n_shard, int(np.ceil(n_entity / n_shard))), dtype=np.int64
        )
        shard_and_idx_to_entity[entity_to_shard, entity_to_idx] = np.arange(
            len(entity_to_idx)
        )
        return cls(
            n_shard=n_shard,
            entity_to_shard=entity_to_shard,
            entity_to_idx=entity_to_idx,
            shard_and_idx_to_entity=shard_and_idx_to_entity,
        )


class BatchSampler:
    """Sample training batches from a dataset of triples.

    Generates batches of numpy arrays containing:

        head      : int64[n_batch_per_call, n_shard, n_shard, n_positive]

        relation  : int64[n_batch_per_call, n_shard, n_shard, n_positive]

        src_tails : int64[n_batch_per_call, n_shard, n_shard, n_positive + n_negative]
    """

    def __init__(
        self,
        triples: np.ndarray,
        sharding: Sharding,
        n_positive: int,
        n_negative: int,
        n_batch_per_call: int,
        seed: Optional[int],
    ):
        self.n_shard = sharding.n_shard
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_batch_per_call = n_batch_per_call
        self.rng = np.random.default_rng(seed=seed)

        # Entity count per shard, useful for sampling negatives
        self.shard_to_count = np.bincount(
            sharding.entity_to_shard, minlength=self.n_shard
        )

        # Build a mapping (shard(head), shard(tail)) -> [(idx(h), r, idx(t))]
        triple_to_shardpair = (
            self.n_shard * sharding.entity_to_shard[triples[:, 0]]
            + sharding.entity_to_shard[triples[:, 2]]
        )
        self.shardpair_to_count = np.bincount(
            triple_to_shardpair, minlength=self.n_shard * self.n_shard
        ).reshape(self.n_shard, self.n_shard)
        self.shardpair_to_offset = np.concatenate(
            [[0], np.cumsum(self.shardpair_to_count)[:-1]]
        ).reshape(self.n_shard, self.n_shard)
        train_hrt_sorted = triples[np.argsort(triple_to_shardpair)]
        self.flat_shardpair_to_hrt = np.stack(
            [
                sharding.entity_to_idx[train_hrt_sorted[:, 0]],
                train_hrt_sorted[:, 1],
                sharding.entity_to_idx[train_hrt_sorted[:, 2]],
            ],
            axis=0,
        )

    def __iter__(self) -> "BatchSampler":
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        # Use a flattened sampling trick to draw a uniform random sample of
        # training triples, stratified by (shard(head), shard(tail))
        sample_idx = (
            self.shardpair_to_offset[None, :, :, None]
            + self.rng.integers(
                1 << 63,
                size=(
                    self.n_batch_per_call,
                    self.n_shard,
                    self.n_shard,
                    self.n_positive,
                ),
            )
            % self.shardpair_to_count[None, :, :, None]
        )
        head, relation, tail = self.flat_shardpair_to_hrt[:, sample_idx]
        # Draw negative samples, uniformly within each shard
        tail_negative = (
            self.rng.integers(
                1 << 63,
                size=(
                    self.n_batch_per_call,
                    self.n_shard,
                    self.n_shard,
                    self.n_negative,
                ),
            )
            % self.shard_to_count[None, None, :, None]
        )
        # Concatenate positive and negative tails and transpose so that the local shard index is first
        src_tails = np.concatenate([tail, tail_negative], axis=3).transpose(0, 2, 1, 3)
        return dict(head=head, relation=relation, src_tails=src_tails)


# Model


def l1_distance(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """Compute batched L1 distance between (at least 2D) tensors.

    a -- float[*group_shape, n_a, embedding_size]

    b -- float[*group_shape, n_b, embedding_size]

    returns -- float[*group_shape, n_a, n_b]
    """
    if poptorch.isRunningOnIpu():
        return T.sum(T.abs(a[..., :, None, :] - b[..., None, :, :]), dim=-1)
    return T.cdist(a, b, p=1.0)


def transe_score(head: T.Tensor, relation: T.Tensor, tail: T.Tensor) -> T.Tensor:
    """Compute a batch of scores using TransE with L1 distance.

    See: Translating Embeddings for Modeling Multi-relational Data
    https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    head -- float[n_positive, embedding_size] -- head entity embeddings

    relation -- float[n_positive, embedding_size] -- relation type embeddings

    tail -- float[n_tail, embedding_size] -- tail candidate entity embeddings

    returns -- float[n_positive, n_tail] -- scores (negative L1 distance) for each positive
                                            & candidate tail
    """
    return -l1_distance(head + relation, tail)


def all_to_all(x: T.Tensor) -> T.Tensor:
    """Cross-replica all-to-all permutation of data (IPU only).

    Each replica sends a fixed-size tensor to every other IPU. For example:

        Step   | IPU0   | IPU1
        ------ | ------ | ------
        Before | [A, B] | [C, D]
        Op     |   all_to_all
        After  | [A, C] | [B, D]

    See `gcl::allToAllCrossReplica` in the Poplar SDK documentation for more information.

    x -- float[n_replica, ...]

    returns -- float[n_replica, ...]
    """
    assert poptorch.isRunningOnIpu()
    ctypes.cdll.LoadLibrary(Path(__file__).parent / Path("build/custom_ops.so"))
    (y,) = poptorch.custom_op(
        [x],
        "AllToAll",
        "ai.graphcore",
        1,
        example_outputs=[x],
    )
    return y


def sampled_softmax_cross_entropy(
    score: T.Tensor, label: T.Tensor, total_classes: int
) -> T.Tensor:
    """A minor modification of softmax cross-entropy with an adjustment for negative sampling.

    The adjustment increases the score of negative classes to account for the fact that
    they are not updated on every step.

    This method assumes that negative classes are drawn with a flat distribution, probability
    (1/total_classes).

    score -- float[batch_size, candidate]

    label -- int[batch_size]

    total_classes -- int -- total number of classes that negative samples are drawn from

    returns -- float[] -- sum softmax cross entropy loss
    """
    # The adjustment for `class` is `log(1 / E(count(candidate == class)))`, which is constant
    # over all negative classes and zero for the target class.
    adjustment = T.tensor(
        np.log(total_classes) - np.log(score.shape[1] - 1),
        device=score.device,
        dtype=score.dtype,
    )
    nonlabel_mask = (
        T.arange(score.shape[1], device=label.device, dtype=label.dtype)[None, :]
        != label[:, None]
    )
    return T.nn.functional.cross_entropy(
        score + adjustment * nonlabel_mask, label, reduction="sum"
    )


class Model(T.nn.Module):
    """A basic knowledge graph embedding (KGE) model using TransE and BESS.

    Parameters:
        entity_embedding -- float[n_shard, max_entity_per_shard, embedding_size]
        relation_embedding -- float[n_relation_type, embedding_size]

    Forward (IPU):
        head -- int[1, n_shard, n_positive]
        relation -- int[1, n_shard, n_positive]
        src_tails -- int[1, n_shard, n_tail]

        Note that this corresponds to the shape on each replica. There should be
        `n_shard` replicas.

    Forward (CPU):
        head -- int[n_shard, n_shard, n_positive]
        relation -- int[n_shard, n_shard, n_positive]
        src_tails -- int[n_shard, n_shard, n_tail]
    """

    def __init__(self, sharding: Sharding, n_relation_type: int, embedding_size: int):
        super().__init__()
        self.sharding = sharding
        self.embedding_size = embedding_size
        self.score = transe_score
        self.entity_embedding = T.nn.Parameter(
            T.FloatTensor(
                sharding.n_shard,
                sharding.max_entity_per_shard,
                embedding_size,
            )
        )
        T.nn.init.normal_(self.entity_embedding, std=1 / embedding_size)
        self.relation_embedding = T.nn.Parameter(
            T.FloatTensor(n_relation_type, embedding_size)
        )
        T.nn.init.normal_(self.relation_embedding, std=1 / embedding_size)

    def forward(
        self, head: T.Tensor, relation: T.Tensor, src_tails: T.Tensor
    ) -> T.Tensor:
        if poptorch.isRunningOnIpu():
            return self.forward_ipu(head, relation, src_tails)
        return self.forward_cpu(head, relation, src_tails)

    def forward_cpu(
        self, head: T.Tensor, relation: T.Tensor, src_tails: T.Tensor
    ) -> T.Tensor:
        n_shard, _, n_positive = head.shape
        _, _, n_tails = src_tails.shape
        shards = T.arange(n_shard)[:, None, None]
        score = self.score(
            head=self.entity_embedding[shards, head]
            .float()
            .view(n_shard, -1, self.embedding_size),
            relation=self.relation_embedding[relation, :]
            .float()
            .view(n_shard, -1, self.embedding_size),
            tail=self.entity_embedding[shards, src_tails]
            .float()
            .permute(1, 0, 2, 3)
            .reshape(n_shard, -1, self.embedding_size),
        )
        true_tail_idx = (
            (T.arange(n_shard)[:, None] * n_tails + T.arange(n_positive)[None, :])
            .view(-1)
            .repeat(n_shard)
        )
        return sampled_softmax_cross_entropy(
            score.view(-1, n_shard * n_tails),
            true_tail_idx,
            total_classes=self.sharding.n_entity,
        )

    def forward_ipu(
        self, head: T.Tensor, relation: T.Tensor, src_tails: T.Tensor
    ) -> T.Tensor:
        head = head.squeeze(0)
        relation = relation.squeeze(0)
        src_tails = src_tails.squeeze(0)
        head_embedding, src_tail_embedding = T.split(
            self.entity_embedding[T.concat([head, src_tails], dim=1)],
            [head.shape[1], src_tails.shape[1]],
            dim=1,
        )
        relation_embedding = self.relation_embedding[relation]
        score = self.score(
            head=head_embedding.float().view(-1, self.embedding_size),
            relation=relation_embedding.float().view(-1, self.embedding_size),
            tail=all_to_all(src_tail_embedding).float().view(-1, self.embedding_size),
        )
        true_tail_idx = (
            T.arange(self.sharding.n_shard, device=score.device, dtype=T.int)[:, None]
            * src_tails.shape[1]
            + T.arange(head.shape[1], device=score.device, dtype=T.int)[None, :]
        ).view(-1)
        return sampled_softmax_cross_entropy(
            score, true_tail_idx, total_classes=self.sharding.n_entity
        )

    # Persistence

    def save(self, path: Path) -> None:
        """Save model parameters and entity sharding metadata to disk."""
        T.save(dict(**self.state_dict(), sharding=self.sharding.__dict__), path)

    def load(self, path: Path) -> None:
        """Load model parameters and entity sharding metadata from disk.

        Note that any `BatchSampler` must be updated to use `model.sharding` after calling
        this method.
        """
        state = T.load(path)
        self.sharding = Sharding(**state.pop("sharding"))
        self.load_state_dict(state)


# Evaluation


def mrr(predictions: np.ndarray, target: np.ndarray) -> float:
    """Compute the mean reciprocal rank (MRR) of targets in a ranked list.

    Missing targets have rank infinity (no contribution).

    predictions -- int[batch_size, n_predictions]

    target -- int[batch_size]

    returns -- float -- mean over the batch
    """
    rows, cols = np.nonzero(predictions == target[:, None])
    assert len(rows) == len(
        np.unique(rows)
    ), "target should never appear twice in predictions"
    return np.sum(1 / (1 + cols)) / predictions.shape[0]


def evaluate_mrr(model: Model, triples: np.ndarray, n_best: int) -> float:
    """Compute MRR for a batch of evaluation triples.

    Note: runs on CPU, for sake of simplicity.

    triples -- int[batch_size, {h, r, t}]

    n_best -- int -- number of tail predictions per query (fix this for a fair comparison)

    returns -- float -- MRR over all triples
    """
    head, relation, true_tail = triples.T
    score = model.score(
        head=model.entity_embedding[
            model.sharding.entity_to_shard[head], model.sharding.entity_to_idx[head]
        ].float(),
        relation=model.relation_embedding[relation].float(),
        tail=model.entity_embedding.view(-1, model.embedding_size).float(),
    )
    _, idx = T.topk(score, k=n_best)
    shards, indices = np.divmod(idx, model.entity_embedding.shape[1])
    return mrr(model.sharding.shard_and_idx_to_entity[shards, indices], true_tail)


def predict(head: int, relation: int, model: Model, n_best: int) -> List[int]:
    """Make the n_best most likely tail predictions for a single `(head, relation, ?)` query.

    Note: runs on CPU, for sake of simplicity.
    """
    score = model.score(
        head=model.entity_embedding[
            None,
            model.sharding.entity_to_shard[head],
            model.sharding.entity_to_idx[head],
        ].float(),
        relation=model.relation_embedding[None, relation].float(),
        tail=model.entity_embedding.view(-1, model.embedding_size).float(),
    )
    _, idx = T.topk(score, k=n_best)
    shards, indices = np.divmod(idx[0], model.entity_embedding.shape[1])
    return list(model.sharding.shard_and_idx_to_entity[shards, indices])


# Training


def create_train_step(
    model: Model,
    optimiser: str,
    lr: float,
    sgdm_momentum: Optional[float],
    weight_decay: float,
    device: str,
    device_iterations: int,
) -> Callable[..., T.Tensor]:
    """Create a 'stepper function' for training, with the same interface across {CPU, IPU}.

    Note: returns the final loss (not summed over device_iterations).

    Usage:

        stepper = create_train_step(...)
        for batch in batches:
            stepper(batch)

    optimiser -- {"adamw" | "sgdm"}

    device -- {"ipu" | "cpu"} -- note that the CPU implementation is slow, only included for testing

    device_iterations -- int -- the number of optimiser steps to take for each call
                                (this must match the batch shape passed to `step()`)

    returns -- fn(head, relation, src_tails) -- training `step()` function
                  -- head, relation, src_tails are numpy arrays, as generated by `BatchSampler`
    """
    if device == "cpu":
        if optimiser == "adamw":
            opt = T.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimiser == "sgdm":
            opt = T.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=sgdm_momentum,
                weight_decay=weight_decay,
            )
        else:
            assert False, f"unexpected optimiser {optimiser!r}"

        def step(batch: Dict[str, np.ndarray]) -> T.Tensor:
            opt.zero_grad()
            for i in range(device_iterations):
                loss = model(**{k: T.tensor(v[i]) for k, v in batch.items()})
                loss.backward()
                opt.step()
            return loss

        return step

    if device == "ipu":
        options = poptorch.Options()
        options.replication_factor = model.sharding.n_shard
        options.deviceIterations(device_iterations)
        options.Precision.enableStochasticRounding(True)
        if "POPLAR_EXECUTABLE_CACHE_DIR" in os.environ:
            options.enableExecutableCaching(
                str(Path(os.environ["POPLAR_EXECUTABLE_CACHE_DIR"]) / "kge_training")
            )

        # Add a memory saving optimisation pattern. This removes an unnecessary
        # entity_embedding gradient all-reduce, which is a no-op since it is fully
        # sharded across replicas.
        ctypes.cdll.LoadLibrary(Path(__file__).parent / "build/custom_ops.so")
        options._popart.setPatterns(dict(RemoveAllReducePattern=True))

        (dtype,) = {p.dtype for p in model.parameters()}
        if optimiser == "adamw":
            opt = poptorch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                accum_type=T.float32,
                first_order_momentum_accum_type=dtype,
                second_order_momentum_accum_type=T.float32,
            )
        elif optimiser == "sgdm":
            opt = poptorch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=sgdm_momentum,
                weight_decay=weight_decay,
                accum_type=T.float32,
                velocity_accum_type=dtype,
            )
        else:
            assert False, f"unexpected optimiser {optimiser!r}"

        ipu_model = poptorch.trainingModel(model, options=options, optimizer=opt)

        # Set `entity_embedding` as a fully sharded parameter across replicas. This
        # disables the gradient all-reduce, and allows the parameters to be initialised
        # and read separately, so that they are treated as separate parameters.
        #
        # (Compare with `relation_embedding`, which keeps default settings, a replicated
        #  parameter across replicas. This means that the gradients across replica are
        #  all-reduced together, such that the parameter value is kept synchronised.)
        ipu_model.entity_embedding.replicaGrouping(
            poptorch.CommGroupType.NoGrouping,
            0,
            poptorch.VariableRetrievalMode.OnePerGroup,
        )

        def step(batch: Dict[str, np.ndarray]) -> T.Tensor:
            # PopTorch expects device_iterations & shard to be flattened
            # into a single dimension
            return ipu_model(
                **{
                    k: T.tensor(np.ascontiguousarray(v), dtype=T.int32).flatten(
                        end_dim=1
                    )
                    for k, v in batch.items()
                }
            )

        return step

    assert False, f"device '{device}' unexpected, expected {{'cpu', 'ipu'}}"


def train(
    model: Model,
    batch_sampler: BatchSampler,
    n_step: int,
    optimiser: str,
    lr: float,
    sgdm_momentum: Optional[float],
    weight_decay: float,
    valid_triples: np.ndarray,
    valid_interval: int,
    device: str,
) -> Iterable[Dict[str, Any]]:
    """Wraps `create_train_step` into a full training loop, with interleaved validation.

    n_step -- int -- total number of optimiser update steps; an upper bound if `n_step` is not a
                     multiple of `batch_sampler.n_batch_per_call`

    valid_triples -- int[n_valid, {h, r, t}] -- triples for interleaved validation
                     (we recommend limiting this in the range of 1000s, as validation runs on CPU)

    valid_interval -- int -- number of optimiser update steps before re-running validation

    See `create_train_step()` for the description of remaining parameters.

    yields dict(
        step    -- int -- number of optimiser update steps taken
        example -- int -- number of positive examples consumed so far
        loss    -- Optional[float] -- mean loss for this step (missing in final validation dict)
        mrr     -- Optional[float] -- validation MRR (periodically, based on `valid_interval`)
        elapsed -- float -- elapsed seconds since last dict (includes training and validaiton time)
    )
    """
    n_best = 10
    t0 = time.time()
    step = create_train_step(
        model=model,
        lr=lr,
        optimiser=optimiser,
        sgdm_momentum=sgdm_momentum,
        weight_decay=weight_decay,
        device=device,
        device_iterations=batch_sampler.n_batch_per_call,
    )
    n_loop = n_step // batch_sampler.n_batch_per_call
    valid_interval_loop = valid_interval // batch_sampler.n_batch_per_call
    examples_per_step = batch_sampler.n_positive * batch_sampler.n_shard**2
    for n in range(n_loop + 1):
        record = dict(step=n * batch_sampler.n_batch_per_call)
        record["example"] = record["step"] * examples_per_step
        if n and (n % valid_interval_loop == 0) or n == n_loop:
            record["mrr"] = evaluate_mrr(model, valid_triples, n_best=n_best)
        if n < n_loop:
            record["loss"] = float(T.sum(step(next(batch_sampler)))) / examples_per_step
        record["elapsed"] = time.time() - t0
        t0 += record["elapsed"]
        yield record
