# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""General support utilities."""

import dataclasses
import datetime
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import poplar_kge as kge
import wandb


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    raise ValueError(f"Value {obj} of type {type(obj)} is not JSON-seralisable")


def recursive_replace(d: kge.Settings, u: Dict[str, Any]) -> kge.Settings:
    for k, v in u.items():
        if dataclasses.is_dataclass(getattr(d, k)):
            setattr(d, k, recursive_replace(getattr(d, k), v))
        else:
            setattr(d, k, v)
    return d


class Logger:
    """Log to wandb, file and stderr."""

    def __init__(
        self, settings: kge.Settings, code_path: str, truncate_train: bool = True
    ):
        self.settings = settings
        self.truncate_train = truncate_train
        self.last_t = time.time()
        self.last_event = "start"
        self.step = 0
        self.log_file = None

        if settings.logging.path:
            settings.logging.path.mkdir(parents=True, exist_ok=True)
            (settings.logging.path / "app.json").write_text(
                json.dumps(settings.flatten(), default=_json_default)
            )
            self.log_file = (settings.logging.path / "log.jsonl").open("w")
            self._write_log(
                dict(
                    event="start",
                    start_time=datetime.datetime.now().isoformat(),
                    settings=settings.flatten(),
                )
            )
        if settings.logging.wandb:
            wandb.init(
                entity="ogb-wikiwiki",
                project="poplar-kge-v2",
                config=dataclasses.asdict(settings),
                dir=tempfile.gettempdir(),
            )
            # If being run from a sweep agent, this will update the settings
            # to those specified in the sweep. If being run from a regular
            # training run, this is a no-op
            self.settings = recursive_replace(settings, wandb.config)
            wandb.run.log_code(code_path)  # type:ignore[union-attr]
        print(f"[start] {self.settings.flatten()}", file=sys.stderr, flush=True)

        self.steps_per_log = (
            self.settings.execution.train_steps_per_program_run
            * self.settings.program_runs_per_log
        )
        self.samples_per_step = settings.model.n_shard * settings.data.batch_size

    def _write_log(self, data: Dict[str, Any]) -> None:
        assert self.log_file
        self.log_file.write(json.dumps(data, default=_json_default) + "\n")
        self.log_file.flush()

    def log(self, event: str, data: Dict[str, Any]) -> None:
        data = data.copy()
        elapsed = time.time() - self.last_t
        self.last_t += elapsed
        if event == "train_step_loop":
            self.step += self.steps_per_log
        sample = self.samples_per_step * self.step

        if self.settings.logging.wandb:
            wandb.log(
                {**data, f"{event}_time": elapsed, "step": self.step, "sample": sample},
                step=self.step,
            )

        if self.log_file:
            self._write_log(
                dict(
                    event=event,
                    step=self.step,
                    time=elapsed,
                    sample=sample,
                    data=data,
                )
            )

        if not self.truncate_train or not (
            event == self.last_event == "train_step_loop"
        ):
            print(
                f"[#{self.step // 1000:>06d}k {event} : {elapsed:.3f} s] {json.dumps(data)}",
                file=sys.stderr,
                flush=True,
            )

        self.last_event = event

    def savez(self, name: str, data: Dict[str, Union[int, np.ndarray]]) -> None:
        if self.settings.logging.wandb:
            np.savez(Path(wandb.run.dir) / name, **data)  # type:ignore[union-attr]
        if self.settings.logging.path:
            np.savez(self.settings.logging.path / name, **data)
