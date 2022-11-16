// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPLAR_EXTENSIONS_DISTANCE_HPP
#define POPLAR_EXTENSIONS_DISTANCE_HPP

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poplar_extensions {

poplar::Tensor l1distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext);

poplar::Tensor l1distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext);

poplar::Tensor l2distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext);

poplar::Tensor l2distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& dist,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext);

}  // namespace poplar_extensions

#endif  // POPLAR_EXTENSIONS_DISTANCE_HPP
