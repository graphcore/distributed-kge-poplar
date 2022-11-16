# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""A reference implementation in plain PyTorch, to match poplar_kge."""

import typing
from typing import Dict

import numpy as np
import poplar_kge as kge
import torch as T
from poplar_kge import Predictions


class Model(T.nn.Module):
    """A reference implementation that should match `poplar_kge.Engine`."""

    @typing.no_type_check
    def __init__(self, settings: kge.Settings):
        super().__init__()
        self.settings: kge.Settings = settings
        m = self.settings.model
        self.register_parameter(
            "entity_embedding",
            T.nn.Parameter(T.FloatTensor(m.n_shard, m.n_entity, m.embedding_size)),
        )
        self.entity_features = T.zeros(m.n_shard, m.n_entity, m.entity_feature_size)
        self.register_parameter(
            "feature_projection",
            T.nn.Parameter(T.FloatTensor(m.entity_feature_size, m.embedding_size)),
        )
        self.register_parameter(
            "relation_embedding",
            T.nn.Parameter(T.FloatTensor(m.n_relation_type, m.embedding_size)),
        )
        T.nn.init.normal_(self.entity_embedding)
        T.nn.init.xavier_normal_(self.feature_projection)
        T.nn.init.normal_(self.relation_embedding)
        s = self.settings.training
        self.opt: T.optim.Optimizer = T.optim.Adam(
            self.parameters(),
            lr=s.learning_rate,
            betas=(s.adam_beta_m, s.adam_beta_v),
            eps=s.adam_epsilon,
            weight_decay=s.weight_decay,
        )

    @typing.no_type_check
    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        with T.no_grad():
            (step,) = set(state["step"])
            for key, param in self.named_parameters():
                if key != "entity_embedding":
                    param[...] = T.tensor(state[key])
                    self.opt.state[param] = dict(
                        step=step,
                        exp_avg=T.tensor(state[f"{key}/adam_m"]),
                        exp_avg_sq=T.tensor(state[f"{key}/adam_v"]),
                    )
            size = self.settings.model.embedding_size
            data = T.tensor(state["entity_data"]).to(T.float)
            self.entity_embedding[...] = data[:, :, 0:size]
            self.opt.state[self.entity_embedding] = dict(
                step=step,
                exp_avg=data[:, :, size : 2 * size],
                exp_avg_sq=data[:, :, 2 * size : 3 * size] ** 2,
            )
            self.entity_features[...] = data[:, :, 3 * size :]

    @typing.no_type_check
    def get_state(self) -> Dict[str, np.ndarray]:
        state = {}
        for key, param in self.named_parameters():
            if key != "entity_embedding":
                state[key] = param.detach().numpy()
                state[f"{key}/adam_m"] = (
                    self.opt.state[param]["exp_avg"].detach().numpy()
                )
                state[f"{key}/adam_v"] = (
                    self.opt.state[param]["exp_avg_sq"].detach().numpy()
                )
        state["entity_data"] = (
            T.cat(
                [
                    self.entity_embedding,
                    self.opt.state[self.entity_embedding]["exp_avg"],
                    T.sqrt(self.opt.state[self.entity_embedding]["exp_avg_sq"]),
                    self.entity_features,
                ],
                dim=2,
            )
            .detach()
            .numpy()
        )
        return state

    @staticmethod
    def all_to_all(tensor: T.Tensor) -> T.Tensor:
        return tensor.transpose(0, 1)

    @staticmethod
    def one_hot_sign(indices: T.Tensor, num_classes: int) -> T.Tensor:
        return (  # type:ignore[no-any-return]
            2 * T.nn.functional.one_hot(indices, num_classes) - 1
        )

    @staticmethod
    def to_tensor(array: np.ndarray) -> T.Tensor:
        if array.dtype == np.uint32:
            array = array.astype(np.int64)
        return T.tensor(array)

    @typing.no_type_check
    def compute_embedding(self, shard: T.Tensor, idx: T.Tensor) -> T.Tensor:
        return (
            self.entity_embedding[shard, idx]
            + self.entity_features[shard, idx] @ self.feature_projection
        )

    @typing.no_type_check
    def compute_broadcast_score(
        self, pred_tails: T.Tensor, tails: T.Tensor
    ) -> T.Tensor:
        return self.settings.model.gamma - T.cdist(pred_tails, tails, p=1)

    @typing.no_type_check
    def loss(self, batch: kge.Batch) -> T.Tensor:
        batch = kge.Batch(**{k: self.to_tensor(v) for k, v in batch.__dict__.items()})
        shards = T.arange(self.settings.model.n_shard)
        entity_embeddings = self.compute_embedding(shards[:, None], batch.remote)
        heads = entity_embeddings[shards[:, None], batch.head]
        predicted_tails = heads + self.relation_embedding[batch.relation, :]
        tails = self.all_to_all(
            entity_embeddings[shards[:, None, None], batch.a2a]
        ).reshape(
            self.settings.model.n_shard,
            self.settings.model.n_shard * self.settings.data.a2a_size,
            self.settings.model.embedding_size,
        )
        scores = self.compute_broadcast_score(predicted_tails, tails)
        return -T.mean(
            batch.weight
            * T.nn.functional.logsigmoid(
                self.one_hot_sign(batch.tail, scores.shape[2]) * scores
            )
        )

    @typing.no_type_check
    def train_step_loop(self, batch: kge.Batch) -> float:
        losses = []
        batches = [
            kge.Batch(**{k: v[:, i] for k, v in batch.__dict__.items()})
            for i in range(self.settings.execution.train_steps_per_program_run)
        ]
        for batch in batches:
            self.opt.zero_grad()
            loss = self.loss(batch)
            loss.backward(T.tensor(self.settings.training.loss_scale))
            self.opt.step()
            losses.append(float(loss))
        return np.mean(losses)

    @typing.no_type_check
    def predict(
        self, shard_idx_relation: np.ndarray, shard_to_count: np.ndarray
    ) -> Predictions:
        with T.no_grad():
            mask = T.zeros((self.settings.model.n_shard, self.settings.model.n_entity))
            for i, l in enumerate(shard_to_count):
                mask[i, 0] = -1e4
                mask[i, 1 + l :] = -1e4
            mask = mask.reshape(-1)

            shard_idx_relation = self.to_tensor(shard_idx_relation)
            all_entity_embs = self.compute_embedding(
                np.arange(self.settings.model.n_shard)[:, None],
                np.arange(self.settings.model.n_entity),
            )
            predicted_tails = (
                all_entity_embs[shard_idx_relation[:, 0], shard_idx_relation[:, 1]]
                + self.relation_embedding[shard_idx_relation[:, 2]]
            )

            query_scores = self.compute_broadcast_score(
                predicted_tails,
                all_entity_embs.reshape(-1, self.settings.model.embedding_size),
            )
            query_scores += mask

            predictions_flat = T.topk(
                query_scores, k=self.settings.execution.predict_n_best
            )

            return Predictions(
                shard_idx=np.stack(
                    np.divmod(predictions_flat.indices, self.settings.model.n_entity),
                    axis=-1,
                ),
                score=predictions_flat.values,
            )
