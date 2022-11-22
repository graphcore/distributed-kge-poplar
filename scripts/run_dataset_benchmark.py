# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import cProfile
import itertools as it
import pstats

import poplar_kge as kge
import poplar_kge_dataset as kge_data
import poplar_kge_utility as kge_utility

profile = cProfile.Profile()

settings = kge.Settings.create_wikikg90mv2()
settings.logging.wandb = False

settings.prepare()
logger = kge_utility.Logger(settings, __file__)

data = kge_data.RawData.load(settings)
logger.log("load_data", {})

dataset = kge_data.Dataset.load(data, settings)
logger.log("build_index", {})

profile.enable()

count = 10
list(it.islice(dataset.batches(), count))
logger.log("sample_batch", dict(count=count))

profile.disable()
pstats.Stats(profile).sort_stats("cumtime").print_stats()
