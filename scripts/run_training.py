# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict, Union

import numpy as np
import poplar_kge as kge
import poplar_kge_dataset as kge_data
import poplar_kge_utility as kge_utility
import torch

# settings = kge.Settings.create_demo()

settings = kge.Settings.create_wikikg90mv2()
# settings.model.score_fn = "RotatE"  # modify settings directly here


# Main script

settings.prepare()
logger = kge_utility.Logger(settings, __file__)
# Grab settings from `logger` in case they've been changed for a sweep
settings = logger.settings

data = kge_data.RawData.load(settings)
logger.log("load_data", {})

dataset = kge_data.Dataset.load(data, settings)
logger.log("build_index", {})

engine = kge.Engine(settings, dataset.shard_to_count)
logger.log("compile", {})

engine.initialise_all(dataset.entity_features(settings.model.entity_feature_size))
logger.log("initialise", {})

ds = kge_data.DatasetWrapper(dataset)
dl = torch.utils.data.DataLoader(
    ds, batch_size=None, num_workers=10, worker_init_fn=ds.worker_init_fn
)
dl_iter = iter(dl)


def validate() -> None:
    for part in ["train", "valid"]:
        logger.log(f"eval_{part}", {f"{part}_mrr": dataset.mrr(part, engine.predict)})


def predict(name: str) -> None:
    results: Dict[str, Union[int, np.ndarray]] = dict(step=logger.step)
    for part in ["valid", "test-dev", "test-challenge"]:
        entity, score = dataset.predict(part, engine.predict)
        results[part] = entity.astype(np.uint32)
        results[f"{part}-score"] = score.astype(np.float16)
    logger.savez(f"predictions_{name}.npz", results)


for n in range(settings.logs_per_training_run):
    if n % settings.logs_per_validation == 0:
        validate()
    if n in settings.predict_at_log:
        predict(str(logger.step))
    loss, lr = np.mean(
        [
            engine.train_step_loop(ds.tensors_to_batch(**next(dl_iter)))
            for _ in range(settings.program_runs_per_log)
        ],
        axis=0,
    )
    logger.log("train_step_loop", dict(loss=loss, learning_rate=lr))
validate()
predict("final")
