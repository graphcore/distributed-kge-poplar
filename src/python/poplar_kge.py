# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Python interface to the Poplar KGE model."""

import dataclasses
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar, Union

import libpoplar_kge
import libpvti as pvti
import numpy as np

Shape = Tuple[int, ...]
R = TypeVar("R")
CHANNEL = pvti.createTraceChannel("poplar_kge")


def instrument_fn(fn: Callable[..., R]) -> Callable[..., R]:
    """Wraps pvti.instrrument_fn(CHANNEL) with a type annotation."""
    return pvti.instrument_fn(CHANNEL)(fn)  # type:ignore[no-any-return]


# Please prepend names of changes *that do not appear in logged settings*
# to this list, for sake of identifying code fixes in experiment logs.
CODE_CHANGES = (
    "FIX-hidden-regularisation-weight",
    "SOFTMAX_STABLE",
    "AKG-22-push-predictions",
    "AKG-43-shift-logged-step-count",
    "AKG-24-MRR-fix",
    "AKG-23-scaled-initialisation",
)


@dataclasses.dataclass
class ModelSettings:
    seed: int
    score_fn: str
    distance_fn: str
    n_shard: int
    n_entity: int  # per-shard
    n_relation_type: int
    embedding_size: int
    entity_feature_size: int
    feature_mlp_size: int
    feature_dropout: float
    negative_adversarial_scale: float
    share_feature_networks: bool
    gamma: float
    init_scale: float


@dataclasses.dataclass
class WikiKg90Mv2Settings:
    name: str = "wikikg90mv2"


@dataclasses.dataclass
class GeneratedDataSettings:
    n_train: int
    n_eval: int
    seed: int
    name: str = "generated"


@dataclasses.dataclass
class CubicRootRelationSampling:
    type: str = "cubic_root"


@dataclasses.dataclass
class DataSettings:
    seed: int
    dataset: Union[WikiKg90Mv2Settings, GeneratedDataSettings]
    batch_size: int  # per-shard
    a2a_size: int  # per-shard
    entity_feature_mapping: str  # "zero" | "full" | "random_projection"
    sampling_strategy: Optional[CubicRootRelationSampling]


@dataclasses.dataclass
class LinearLearningRateDecay:
    type: str = "linear"


@dataclasses.dataclass
class SteppedLearningRateDecay:
    step: int
    multiplier: float
    type: str = "stepped"


@dataclasses.dataclass
class ExponentialLearningRateDecay:
    half_life_steps: int
    type: str = "exponential"


@dataclasses.dataclass
class LogSigmoidLoss:
    type: str = "logsigmoid"


@dataclasses.dataclass
class SoftmaxLoss:
    correction_weight: float
    type: str = "softmax"


@dataclasses.dataclass
class NormRegularisation:
    power: float
    weight: float


@dataclasses.dataclass
class TrainingSettings:
    n_step: int
    validation_interval: int
    loss: Union[LogSigmoidLoss, SoftmaxLoss]
    embedding_regularisation: NormRegularisation
    feature_regularisation: NormRegularisation
    hidden_regularisation: NormRegularisation
    learning_rate: float
    learning_rate_decay: Union[
        None,
        LinearLearningRateDecay,
        SteppedLearningRateDecay,
        ExponentialLearningRateDecay,
    ]
    learning_rate_modifiers: Dict[str, float]
    adam_beta_m: float
    adam_beta_v: float
    adam_epsilon: float
    weight_decay: float
    loss_scale: float


@dataclasses.dataclass
class ExecutionSettings:
    device: str  # "ipu" | "cpu"
    dtype: str  # "float16"| "float32"
    train_steps_per_program_run: int
    rw_batch_size: int  # per-shard
    predict_hr_batch_size: int  # per-shard
    predict_tail_batch_size: int  # per-shard
    predict_n_best: int


@dataclasses.dataclass
class LogSettings:
    path: Optional[Path]
    wandb: bool
    steps_per_log: int
    predict_at: Tuple[float, ...]  # proportion of training run, 0.9 = 90%


@dataclasses.dataclass
class Settings:
    seed: int
    model: ModelSettings
    data: DataSettings
    training: TrainingSettings
    execution: ExecutionSettings
    logging: LogSettings
    code_changes: Tuple[str, ...] = CODE_CHANGES

    @property
    def program_runs_per_log(self) -> int:
        return int(
            np.ceil(
                self.logging.steps_per_log / self.execution.train_steps_per_program_run
            )
        )

    @property
    def logs_per_training_run(self) -> int:
        return int(
            np.ceil(
                self.training.n_step
                / self.execution.train_steps_per_program_run
                / self.program_runs_per_log
            )
        )

    @property
    def logs_per_validation(self) -> int:
        return int(
            np.ceil(
                self.training.validation_interval
                / self.execution.train_steps_per_program_run
                / self.program_runs_per_log,
            )
        )

    @property
    def n_tail(self) -> int:
        return self.model.n_shard * self.data.a2a_size

    @property
    def n_remote(self) -> int:
        return self.data.batch_size + self.n_tail

    @property
    def entity_data_r_size(self) -> int:
        return 3 * self.model.embedding_size + self.model.entity_feature_size

    @property
    def entity_data_r_shape(self) -> Shape:
        return (
            self.model.n_shard,
            self.execution.rw_batch_size,
            self.entity_data_r_size,
        )

    @property
    def entity_data_w_shape(self) -> Shape:
        return (
            self.model.n_shard,
            self.execution.rw_batch_size,
            self.model.entity_feature_size,
        )

    @property
    def predict_at_log(self) -> Set[int]:
        steps_per_log = (
            self.execution.train_steps_per_program_run * self.program_runs_per_log
        )
        return {
            int(round(t * self.training.n_step / steps_per_log))
            for t in self.logging.predict_at
        }

    @property
    def relation_embedding_size(self) -> int:
        if self.model.score_fn == "RotatE":
            return self.model.embedding_size // 2
        return self.model.embedding_size

    def prepare(self) -> "Settings":
        """Choose random seeds."""
        if self.seed is None:
            self.seed = np.random.randint(1 << 30)
        model_seed, data_seed, raw_data_seed = [
            int(s.generate_state(1)) for s in np.random.SeedSequence(self.seed).spawn(3)
        ]
        if self.model.seed is None:
            self.model.seed = model_seed
        if self.data.seed is None:
            self.data.seed = data_seed
        if (
            isinstance(self.data.dataset, GeneratedDataSettings)
            and self.data.dataset.seed is None
        ):
            self.data.dataset.seed = raw_data_seed
        if isinstance(self.logging.path, str):
            self.logging.path = Path(self.logging.path)
        if self.training.loss_scale is None:
            self.training.loss_scale = float(
                self.data.batch_size * self.model.n_shard * self.data.a2a_size
            )
        return self

    def flatten(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        self._flatten(self, (), result)
        return result

    @classmethod
    def _flatten(
        cls, obj: Any, prefix: Tuple[str, ...], result: Dict[str, Any]
    ) -> None:
        if dataclasses.is_dataclass(obj):
            for k, v in obj.__dict__.items():
                cls._flatten(v, prefix + (k,), result)
        else:
            result[".".join(prefix)] = obj

    @classmethod
    def create_demo(cls) -> "Settings":
        return cls(
            seed=None,  # type:ignore[arg-type]
            model=ModelSettings(
                seed=None,  # type:ignore[arg-type]
                score_fn="TransE",
                distance_fn="L1",
                n_shard=2,
                n_entity=90,
                n_relation_type=12,
                embedding_size=8,
                entity_feature_size=6,
                feature_mlp_size=0,
                feature_dropout=0.0,
                negative_adversarial_scale=0.0,
                share_feature_networks=True,
                gamma=2.0,
                init_scale=1.0,
            ),
            data=DataSettings(
                seed=None,  # type:ignore[arg-type]
                batch_size=6,
                a2a_size=5,
                entity_feature_mapping="full",
                dataset=GeneratedDataSettings(
                    n_train=50,
                    n_eval=50,
                    seed=None,  # type:ignore[arg-type]
                ),
                sampling_strategy=CubicRootRelationSampling(),
            ),
            training=TrainingSettings(
                n_step=100,
                validation_interval=50,
                loss=LogSigmoidLoss(),
                embedding_regularisation=NormRegularisation(power=3.0, weight=0.0),
                feature_regularisation=NormRegularisation(power=3.0, weight=0.0),
                hidden_regularisation=NormRegularisation(power=3.0, weight=0.0),
                learning_rate=0.1,
                learning_rate_decay=None,
                learning_rate_modifiers={},
                adam_beta_m=0.9,
                adam_beta_v=0.999,
                adam_epsilon=1e-8,
                weight_decay=0.0,
                loss_scale=None,  # type:ignore[arg-type]
            ),
            execution=ExecutionSettings(
                device="cpu",
                dtype="float32",
                train_steps_per_program_run=5,
                rw_batch_size=50,
                predict_hr_batch_size=30,
                predict_tail_batch_size=50,
                predict_n_best=11,
            ),
            logging=LogSettings(
                path=None,
                wandb=False,
                steps_per_log=20,
                predict_at=(0.5,),
            ),
        )

    @classmethod
    def create_wikikg90mv2(cls) -> "Settings":
        n_shard = 16
        return cls(
            seed=None,  # type:ignore[arg-type]
            model=ModelSettings(
                seed=None,  # type:ignore[arg-type]
                score_fn="TransE",
                distance_fn="L1",
                n_shard=n_shard,
                n_entity=int(np.ceil(91230610 / n_shard) + 1),
                n_relation_type=int(n_shard * np.ceil(1387 / n_shard)),
                embedding_size=256,
                entity_feature_size=768,
                feature_mlp_size=0,
                feature_dropout=0.0,
                negative_adversarial_scale=0.0,
                share_feature_networks=True,
                gamma=10.0,
                init_scale=1.0,
            ),
            data=DataSettings(
                seed=None,  # type:ignore[arg-type]
                batch_size=512,
                a2a_size=96,
                entity_feature_mapping="full",
                sampling_strategy=CubicRootRelationSampling(),
                dataset=WikiKg90Mv2Settings(),
            ),
            training=TrainingSettings(
                n_step=int(5e6),
                validation_interval=int(2.5e5),
                loss=SoftmaxLoss(correction_weight=1.0),
                embedding_regularisation=NormRegularisation(power=3.0, weight=0.0),
                feature_regularisation=NormRegularisation(power=3.0, weight=1e-6),
                hidden_regularisation=NormRegularisation(power=3.0, weight=0.0),
                learning_rate=1e-3,
                learning_rate_decay=LinearLearningRateDecay(),
                learning_rate_modifiers={},
                adam_beta_m=0.9,
                adam_beta_v=0.999,
                adam_epsilon=1e-8,
                weight_decay=0.0,
                loss_scale=None,  # type:ignore[arg-type]
            ),
            execution=ExecutionSettings(
                device="ipu",
                dtype="float16",
                train_steps_per_program_run=20,
                rw_batch_size=512,
                predict_hr_batch_size=128,
                predict_tail_batch_size=1024,
                predict_n_best=100,
            ),
            logging=LogSettings(
                path=None,
                wandb=True,
                steps_per_log=int(1e3),
                predict_at=(0.8, 0.9),
            ),
        )

    @classmethod
    def create_wikikg90mv2_short(cls) -> "Settings":
        settings = cls.create_wikikg90mv2()
        settings.data.batch_size = 256
        settings.data.a2a_size = 32
        settings.execution.train_steps_per_program_run = 100
        return settings


@dataclasses.dataclass
class Batch:
    """A batch of input data for training.

    remote -- uint32[n_shard x steps x (batch_size + n_shard * a2a_size)]
              - Indices must be unique within a shard
              - Zero is reserved for padding
    """

    # noqa: E262
    remote: np.ndarray  # . uint32[n_shard x steps x (batch_size + n_shard * a2a_size)]
    a2a: np.ndarray  # .... uint32[n_shard x steps x n_shard x a2a_size] -- into remote
    head: np.ndarray  # ... uint32[n_shard x steps x batch_size] -- into remote
    relation: np.ndarray  # uint32[n_shard x steps x batch_size]
    tail: np.ndarray  # ... uint32[n_shard x steps x batch_size] -- into remote[a2a].T.reshape(n_shard, -1)


@dataclasses.dataclass
class Predictions:
    """A batch of top predictions of sharded entities."""

    shard_idx: np.ndarray  # uint32[(*) x n_best x 2]
    score: np.ndarray  # ... float32[(*) x n_best]

    def topk(self, k: int) -> "Predictions":
        """Top-k for each item in the batch (after: n_best == k)."""
        # Fiddly reshaping to deal with arbitary leading axes
        n_best = self.score.shape[-1]
        flat_shard_idx = self.shard_idx.reshape(-1, n_best, 2)
        flat_score = self.score.reshape(-1, n_best)
        batch_idx = np.arange(flat_score.shape[0])[:, np.newaxis]
        idx = np.argpartition(flat_score, -k, axis=1)[:, -k:]
        return Predictions(
            shard_idx=flat_shard_idx[batch_idx, idx].reshape(
                self.score.shape[:-1] + (k, 2)
            ),
            score=flat_score[batch_idx, idx].reshape(self.score.shape[:-1] + (k,)),
        )

    def sort(self) -> "Predictions":
        """Sort within each item (descending) in the batch."""
        # Fiddly reshaping to deal with arbitary leading axes
        n_best = self.score.shape[-1]
        flat_shard_idx = self.shard_idx.reshape(-1, n_best, 2)
        flat_score = self.score.reshape(-1, n_best)
        batch_idx = np.arange(flat_score.shape[0])[:, np.newaxis]
        idx = np.argsort(flat_score, axis=1)[..., ::-1]
        return Predictions(
            shard_idx=flat_shard_idx[batch_idx, idx].reshape(self.shard_idx.shape),
            score=flat_score[batch_idx, idx].reshape(self.score.shape),
        )


class Engine:
    """Wraps libpoplar_kge.Engine to make it slightly more palatable."""

    @instrument_fn
    def __init__(self, settings: Settings, shard_to_entity_count: np.ndarray):
        """
        Create the model, compile, and load onto the target device.

        shard_to_entity_count: int[n_shard] -- the number of actual entities in each shard (NOT including padding)
        """
        self.settings = settings
        assert shard_to_entity_count.shape == (settings.model.n_shard,)
        assert np.all(shard_to_entity_count < self.settings.model.n_entity)
        self.shard_to_entity_count = shard_to_entity_count
        self.random = np.random.RandomState(settings.model.seed)

        engine_settings = self.settings.flatten()
        del engine_settings["logging.path"]
        del engine_settings["logging.predict_at"]
        del engine_settings["code_changes"]
        for k in list(engine_settings):
            if engine_settings[k] is None:
                del engine_settings[k]
        self.engine = libpoplar_kge.Engine(
            engine_settings, gp_folder=str(Path(libpoplar_kge.__file__).parent)
        )

        self.variable_to_shape = dict(self.engine.run("variables", {})["variables"])
        self.variable_to_dtype = {
            name: np.dtype("uint32" if name == "step" else "float32")
            for name in self.variable_to_shape
        }
        self.uninitialized_variables = set(self.variable_to_shape)
        self.uninitialized_entity_data = np.full(self.settings.model.n_entity, True)
        self.entity_data_initialized: bool = False
        self.init_scale = (
            self.settings.model.init_scale / self.settings.model.embedding_size
        )
        self.step = 0
        self.current_learning_rate = settings.training.learning_rate
        if isinstance(settings.training.learning_rate_decay, SteppedLearningRateDecay):
            self.decay_step = float(settings.training.learning_rate_decay.step)

    # Initialisation

    @staticmethod
    def _check_shape(name: str, array: np.ndarray, expected: Shape) -> None:
        if array.shape != tuple(expected):
            raise ValueError(
                f"'{name}' has incorrect shape, expected {expected}, actual {array.shape}"
            )

    @instrument_fn
    def write_variable(
        self, name: str, value_or_fn: Union[Callable[[Shape], np.ndarray], np.ndarray]
    ) -> None:
        """Write an SRAM variable, from explicit value or initialisation function."""
        if isinstance(value_or_fn, np.ndarray):
            value = value_or_fn
        else:
            value = value_or_fn(self.variable_to_shape[name])
        self._check_shape(f"write({name})", value, self.variable_to_shape[name])
        self.engine.run("write", dict(name=name, value=value))
        self.uninitialized_variables -= {name}

    @instrument_fn
    def read_variable(self, name: str) -> np.ndarray:
        """Read an SRAM variable."""
        shape = self.variable_to_shape[name]
        dtype = self.variable_to_dtype[name]
        value = np.full(shape, np.nan if dtype.kind == "f" else 0, dtype=dtype)
        self.engine.run("read", dict(name=name, value=value))
        return value

    @instrument_fn
    def _rw_indices(self, offset: int) -> np.ndarray:
        indices = np.arange(
            offset, offset + self.settings.execution.rw_batch_size, dtype=np.uint32
        )
        # read/write overflow data at index 0
        indices *= indices < self.settings.model.n_entity
        return np.tile(indices[np.newaxis], (self.settings.model.n_shard, 1))

    @staticmethod
    def _int16_if_float16(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.float16:
            return array.view(np.int16)
        return array

    @instrument_fn
    def write_entity(self, offset: int, data: np.ndarray) -> None:
        """Write a contiguous chunk of entity data [offset, offset + rw_batch_size)."""
        self._check_shape("write_entity(data)", data, self.settings.entity_data_w_shape)
        indices = self._rw_indices(offset)
        self.engine.run(
            "write_entity",
            dict(write_indices=indices, write_features=self._int16_if_float16(data)),
        )
        self.uninitialized_entity_data[indices[0]] = False

    @instrument_fn
    def read_entity(self, offset: int) -> np.ndarray:
        """Read a contiguous chunk of entity data from [offset, min(offset + rw_batch_size, n_entity))."""
        indices = self._rw_indices(offset)
        dtype = self.settings.execution.dtype
        value = np.zeros(self.settings.entity_data_r_shape, dtype=dtype)
        self.engine.run(
            "read_entity",
            dict(read_indices=indices, read_data=self._int16_if_float16(value)),
        )
        return value[:, : (self.settings.model.n_entity - offset), :]

    @instrument_fn
    def read_entity_all(self) -> np.ndarray:
        """Read the whole entity data tensor.

        WARNING - for full-size models, this may blow up host RAM.
        """
        return np.concatenate(
            [
                self.read_entity(offset)
                for offset in range(
                    0,
                    self.settings.model.n_entity,
                    self.settings.execution.rw_batch_size,
                )
            ],
            axis=1,
        )

    @instrument_fn
    def random_normal(
        self, shape: Shape, dtype: Union[str, np.dtype] = "float32"
    ) -> np.ndarray:
        """Utility for generating seeded random normal data."""
        return self.random.normal(size=shape).astype(dtype)

    @instrument_fn
    def random_uniform(
        self,
        shape: Shape,
        low: float = 0,
        high: float = 1,
        dtype: Union[str, np.dtype] = "float32",
    ) -> np.ndarray:
        """Utility for generating seeded random normal data."""
        return self.random.uniform(low=low, high=high, size=shape).astype(dtype)

    @instrument_fn
    def random_integers(self, limit: int, shape: Shape) -> np.ndarray:
        """Utility for generating seeded random integer data in the range [0, limit)."""
        return self.random.randint(limit, size=shape).astype(np.uint32)

    @instrument_fn
    def initialise_variables(self) -> None:
        """Reset SRAM variables to initial values."""

        self.write_variable("step", np.full(self.settings.model.n_shard, 0, np.uint32))
        for name, shape in self.variable_to_shape.items():
            if name.endswith("/adam_m") or name.endswith("/adam_v"):
                self.write_variable(name, np.zeros(shape, dtype=np.float32))

        init_fn = self.random_normal
        if self.settings.model.score_fn == "RotatE":
            init_fn = functools.partial(self.random_uniform, low=-np.pi, high=np.pi)
        init_relation_embedding = self.init_scale * init_fn(
            (
                self.settings.model.n_relation_type,
                self.settings.relation_embedding_size,
            )
        )
        self.write_variable("relation_embedding", init_relation_embedding)
        if self.settings.model.score_fn == "TransH":
            init_relation_normal = self.init_scale * self.random_normal(
                (
                    self.settings.model.n_relation_type,
                    self.settings.relation_embedding_size,
                )
            )
            self.write_variable("relation_normal", init_relation_normal)

        for prefix in (
            [""] if self.settings.model.share_feature_networks else ["head_", "tail_"]
        ):
            init_feature_projection = (
                self.init_scale / np.sqrt(self.settings.model.entity_feature_size)
            ) * self.random_normal(
                (
                    self.settings.model.entity_feature_size,
                    self.settings.model.embedding_size,
                )
            )
            self.write_variable(f"{prefix}feature_projection", init_feature_projection)

            if self.settings.model.feature_mlp_size > 0:
                init_mlp_up_projection = (
                    self.init_scale / np.sqrt(2 * self.settings.model.embedding_size)
                ) * self.random_normal(
                    (
                        2 * self.settings.model.embedding_size,
                        self.settings.model.feature_mlp_size,
                    )
                )
                self.write_variable(
                    f"{prefix}mlp_up_projection", init_mlp_up_projection
                )

                init_mlp_down_projection = (
                    self.init_scale / np.sqrt(self.settings.model.feature_mlp_size)
                ) * self.random_normal(
                    (
                        self.settings.model.feature_mlp_size,
                        self.settings.model.embedding_size,
                    )
                )
                self.write_variable(
                    f"{prefix}mlp_down_projection", init_mlp_down_projection
                )

    @instrument_fn
    def initialise_all(self, entity_features: np.ndarray) -> None:
        """Reset all variables to initial values."""

        self.initialise_variables()

        dtype = self.settings.execution.dtype
        rw_batch_size = self.settings.execution.rw_batch_size
        for offset in range(0, self.settings.model.n_entity, rw_batch_size):
            # Note: np.array() so that we definitely load any memmap
            features_chunk = entity_features[:, offset : (offset + rw_batch_size), :]
            features_chunk = np.pad(
                features_chunk,
                [(0, 0), (0, rw_batch_size - features_chunk.shape[1]), (0, 0)],
            )
            self.write_entity(offset, features_chunk.astype(dtype))

    @instrument_fn
    def read_all(self) -> Dict[str, np.ndarray]:
        """Read all of variables, including entity data into numpy arrays.

        WARNING - for full-size models, this may blow up host RAM.
        """
        return dict(
            **{name: self.read_variable(name) for name in self.variable_to_shape},
            entity_data=self.read_entity_all(),
        )

    # Training

    @instrument_fn
    def train_step_loop(self, batch: Batch) -> Tuple[float, float]:
        """Take `self.settings.execution.train_steps_per_program_run` training steps, returning the average loss."""

        if not self.entity_data_initialized and np.any(self.uninitialized_entity_data):
            raise ValueError(
                "Calling train_step() with uninitialised entity data"
                f" ({np.sum(self.uninitialized_entity_data)} entities)"
            )
        else:
            self.entity_data_initialized = True
        if self.uninitialized_variables:
            raise ValueError(
                f"Calling train_step() with uninitialised variables {list(self.uninitialized_variables)}"
            )
        loss = np.full(self.settings.model.n_shard, np.nan, dtype=np.float32)

        if isinstance(
            self.settings.training.learning_rate_decay, LinearLearningRateDecay
        ):
            self.current_learning_rate = self.settings.training.learning_rate * (
                1 - self.step / self.settings.training.n_step
            )
        elif isinstance(
            self.settings.training.learning_rate_decay, SteppedLearningRateDecay
        ):
            if self.step >= self.decay_step:
                self.current_learning_rate *= (
                    self.settings.training.learning_rate_decay.multiplier
                )
                self.decay_step *= 1.5
        elif isinstance(
            self.settings.training.learning_rate_decay, ExponentialLearningRateDecay
        ):
            decay = (
                self.step / self.settings.training.learning_rate_decay.half_life_steps
            )
            self.current_learning_rate = (
                self.settings.training.learning_rate * 0.5**decay
            )

        self.step += self.settings.execution.train_steps_per_program_run
        self.engine.run(
            "train_step_loop",
            dict(
                learning_rate=np.full(
                    self.settings.model.n_shard,
                    self.current_learning_rate,
                    dtype=np.float32,
                ),
                loss=loss,
                **{
                    k: self._int16_if_float16(np.ascontiguousarray(v))
                    for k, v in batch.__dict__.items()
                },
            ),
        )
        return float(np.sum(loss)), self.current_learning_rate

    # Prediction

    @instrument_fn
    def _predict_hr_chunk(self, idx_relation: np.ndarray) -> Predictions:
        n_shard = self.settings.model.n_shard
        hr_batch_size = self.settings.execution.predict_hr_batch_size
        n_best = self.settings.execution.predict_n_best

        # (hr_shard, tail_shard, hr_batch, tail_batch)
        scores = np.full(
            (n_shard, n_shard, hr_batch_size, n_best), np.nan, dtype=np.float32
        )
        indices = np.full(scores.shape, 0, dtype=np.uint32)
        self.engine.run(
            "predict",
            dict(
                predict_head=np.ascontiguousarray(idx_relation[:, :, 0]),
                predict_relation=np.ascontiguousarray(idx_relation[:, :, 1]),
                predict_entity_count=self.shard_to_entity_count.astype(
                    np.uint32, order="C"
                ),
                predict_scores=scores,
                predict_indices=indices,
            ),
        )
        shard_idx = np.zeros(indices.shape + (2,), dtype=np.uint32)
        shard_idx[..., 0] = np.arange(n_shard)[np.newaxis, :, np.newaxis, np.newaxis]
        shard_idx[..., 1] = indices
        return (
            Predictions(
                shard_idx=shard_idx.transpose(0, 2, 1, 3, 4).reshape(
                    n_shard, hr_batch_size, n_shard * n_best, 2
                ),
                score=scores.transpose(0, 2, 1, 3).reshape(
                    n_shard, hr_batch_size, n_shard * n_best
                ),
            )
            .topk(n_best)
            .sort()
        )

    @instrument_fn
    def predict(self, shard_idx_relation: np.ndarray) -> Predictions:
        """Make sharded tail predictions given (head, relation).

        shard_idx_relation: uint32[n x 3] -- (head.shard, head.index, head.relation)

        returns: Predictions[n x n_best]
        """
        if shard_idx_relation.shape[1:] != (3,):
            raise ValueError("shard_idx_relation should be of shape (n, 3)")

        n_shard = self.settings.model.n_shard
        chunk_size = self.settings.execution.predict_hr_batch_size
        n_best = self.settings.execution.predict_n_best

        shard_to_dataidx = [
            np.where(shard_idx_relation[:, 0] == i)[0] for i in range(n_shard)
        ]
        predictions = Predictions(
            shard_idx=np.zeros((shard_idx_relation.shape[0], n_best, 2), np.uint32),
            score=np.zeros((shard_idx_relation.shape[0], n_best), np.float32),
        )
        for offset in range(0, max(x.size for x in shard_to_dataidx), chunk_size):
            chunk_shard_to_dataidx = [
                x[offset : offset + chunk_size] for x in shard_to_dataidx
            ]
            chunk_idx_relation = np.zeros((n_shard, chunk_size, 2), dtype=np.uint32)
            for shard, dataidx in enumerate(chunk_shard_to_dataidx):
                chunk_idx_relation[shard, : len(dataidx)] = shard_idx_relation[
                    dataidx, 1:3
                ]

            chunk_predictions = self._predict_hr_chunk(chunk_idx_relation)

            for shard, dataidx in enumerate(chunk_shard_to_dataidx):
                predictions.shard_idx[dataidx] = chunk_predictions.shard_idx[
                    shard, : len(dataidx)
                ]
                predictions.score[dataidx] = chunk_predictions.score[
                    shard, : len(dataidx)
                ]
        return predictions
