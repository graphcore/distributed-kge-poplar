# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Profiling script."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import poplar_kge as kge
import poplar_kge_utility as kge_utility
import test_poplar_kge


def profile(settings: kge.Settings) -> None:
    if settings.logging.path:
        settings.logging.path.mkdir(parents=True, exist_ok=True)
        assert (
            "POPLAR_ENGINE_OPTIONS" not in os.environ
        ), "POPLAR_ENGINE_OPTIONS should not be set outside the run_profile script"
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(
            {
                "autoReport.directory": str(settings.logging.path),
                "autoReport.all": True,
                "autoReport.outputArchive": False,
                "autoReport.executionProfileProgramRunCount": 1,
                "profiler.replicaToProfile": 0,
            }
        )

    logger = kge_utility.Logger(settings, __file__, truncate_train=False)
    engine = kge.Engine(
        settings, np.full(settings.model.n_shard, settings.model.n_entity - 1)
    )
    logger.log("compile", {})

    engine.initialise_variables()
    # hack to avoid having to run additional 'fill' programs
    engine.uninitialized_entity_data[:] = False
    logger.log("initialise", {})

    train_batch = test_poplar_kge._random_batch(engine)
    predict_query = test_poplar_kge._random_prediction_query(
        engine, engine.settings.execution.predict_hr_batch_size
    )
    logger.log("sample", {})

    for _ in range(5):
        engine.train_step_loop(train_batch)
        logger.log("train_step_loop", {})

    engine.predict(predict_query)
    logger.log("predict_step", {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()

    settings = kge.Settings.create_wikikg90mv2()

    settings.logging.wandb = False
    settings.logging.path = args.output
    settings.prepare()
    profile(settings)
