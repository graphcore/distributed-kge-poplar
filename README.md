# Distributed KGE (C++)

IPU implementation of a sharded knowledge graph embedding (KGE) model, implemented in Poplar for execution using DRAM on an IPU-POD16.

Note that this is a low-level implementation for advanced IPU usage.


## Usage

### First-time setup

0. Ensure `clang++` and `ninja` are installed.
1. Clone this repository with `--recurse-submodules`.
2. Install Poplar SDK and activate with `source $POPLAR_SDK_DIR/enable`.
3. Create and activate a Python virtual environment.
4. Install Python requirements `pip install -r requirements-dev.txt`
5. Check everything is working by running `./dev` (see also `./dev --help`).

For example:

```sh
sudo apt-get install clang++ ninja
git clone --recurse-submodules REPO
source $POPLAR_SDK_DIR/enable
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
./dev --help
./dev
```

### Training

Our standard training script is in [scripts/run_training.py](scripts/run_training.py). To build the core C++ code, add it to the path and run training,

```sh
./dev train
```

This trains a TransE model with embedding size 256.

Note:
 - Build and develelopment automation is provided by the `./dev` script, which generates a ninja build file (`build/build.ninja`).
 - You may wish to change C++ compiler, e.g. `env CXX=g++ ./dev ...`
 - The training script expects the OGB WikiKG90Mv2 dataset to be downloaded to `$OGBWIKIKG_PATH`; see the [OGB WikiKG90Mv2 page](https://ogb.stanford.edu/docs/lsc/wikikg90mv2/) for instructions.


## About

The application is a self-contained research platform for KGE models, using Poplar/PopLibs directly for execution on IPU, PyTorch for data loading and numpy for batching and interchange. Since model checkpoints would be very large, all training, evaluation and prediction tasks are run in a single job via `run_training.py`.

The main components are:

 - [scripts/{run_training.py, run_profile.py}](scripts/) - top-level entry points, note that we use Python configuration in place of a command line interface
 - Core model & training
   - [src/poplar_kge.cpp](src/poplar_kge.cpp) - core model and training step definition
   - [src/python/poplar_kge.py](src/python/poplar_kge.py) - Python glue code & experiment settings
   - [src/python/poplar_kge_dataset.py](src/python/poplar_kge_dataset.py) - data sampling & batching
 - Library-like components
   - [src/pag/](src/pag/) - Poplar AutoGrad (PAG), a self-contained mini-library for adding automatic differentiation to PopLibs programs
   - [src/fructose/](src/fructose/) - Fructose, a self-contained mini-library for a friendly, noise-free interface to PAG
   - [src/poplar_extensions/](src/poplar_extensions/) - custom device codelets, with a PopLibs-like interface, for efficient L1/L2 distance

See also [doc/design.md](doc/design.md) for a more detailed description of the design of the application.

### Poplar remote buffers

We rely on Poplar's access to streaming memory in this code (see [IPU memory architecture](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html#memory-architecture)), which enables sparse access to a much larger memory store. This is accessed via the [remote memory buffers](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html#remote-memory-buffers) API.

One implementation detail of interest is that we stack all remote embedding state (consisting of entity features, embeddings and optimiser state) into a single remote buffer, which helps to minimise memory overhead due to padding.

### References & license

The included code is released under a MIT license (see [LICENSE](LICENSE)).

Copyright (c) 2022 Graphcore Ltd. Licensed under the MIT License

Our dependencies are:

| Component | Type | About | License |
| --- | --- | --- | --- |
| pybind11 | submodule | C++/Python interop library ([github](https://github.com/pybind/pybind11)) | BSD 3-Clause |
| Catch2 | submodule | C++ unit testing framework ([github](https://github.com/catchorg/Catch2)) | Boost |
| OGB | `requirements.txt` | Open Graph Benchmark dataset and task definition ([paper](https://arxiv.org/abs/2103.09430), [website](https://ogb.stanford.edu/)) | MIT |
| PyTorch | `requirements.txt` | Machine learning framework ([website](https://pytorch.org/)) | BSD 3-Clause |
| WandB | `requirements.txt` | Weights and Biases client library ([website](https://wandb.ai/)), for optional logging to wandb servers | MIT |

We also use ninja ([website](https://ninja-build.org/)) with clang++ from LLVM ([website](https://clang.llvm.org/)) to build C++ code and additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).

The OGB WikiKG90Mv2 dataset is licenced under CC-0.
