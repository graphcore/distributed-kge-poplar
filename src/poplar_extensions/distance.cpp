// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "distance.hpp"

#include <cmath>
#include <map>
#include <sstream>

#include <poputil/VertexTemplates.hpp>

namespace {

void mapTensor2Dblocks(poplar::Graph& graph, poplar::Tensor& t) {
    assert(t.rank() == 2 && "only 2D tensors can use mapTensor2Dblocks");
    auto nTiles = graph.getTarget().getNumTiles();
    auto blockSize0 =
        std::max<unsigned>(std::ceil(t.dim(0) / nTiles),
                           std::ceil(std::sqrt(static_cast<float>(t.numElements()) / nTiles)));
    auto nBlocks0 = (t.dim(0) + blockSize0 - 1) / blockSize0;
    auto nBlocks1 = std::max<unsigned>(1u, nTiles / nBlocks0);
    auto blockSize1 = (t.dim(1) + nBlocks1 - 1) / nBlocks1;
    for (auto i = 0u; i < nBlocks0; ++i) {
        for (auto j = 0u; j < nBlocks1; ++j) {
            auto tile = nBlocks1 * i + j;
            graph.setTileMapping(t.slice({std::min<unsigned>(i * blockSize0, t.dim(0)),
                                          std::min<unsigned>(j * blockSize1, t.dim(1))},
                                         {std::min<unsigned>((i + 1) * blockSize0, t.dim(0)),
                                          std::min<unsigned>((j + 1) * blockSize1, t.dim(1))}),
                                 tile);
        }
    }
}

poplar::Tensor getCachedCopy(std::map<std::pair<size_t, size_t>, poplar::Tensor>& cache,
                             poplar::Graph& graph,
                             size_t tile,
                             size_t index,
                             const poplar::Tensor& t,
                             poplar::program::Sequence& prog) {
    std::pair<size_t, size_t> key = {tile, index};
    auto iter = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }
    poplar::Tensor copy = graph.addVariable(t.elementType(), t.shape());
    graph.setTileMapping(copy, tile);
    cache[key] = copy;
    prog.add(poplar::program::Copy(t, copy));
    return copy;
}

}  // namespace

namespace poplar_extensions {

poplar::Tensor l1distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        std::ostringstream msg;
        msg << "Bad arguments to l1distance, expected a.shape (M, K), b.shape (N, K), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t n = b.dim(0);
    poplar::Tensor out =
        graph.addVariable(a.elementType(), {a.dim(0), b.dim(0)}, {debugContext, "l1dist_out"});
    mapTensor2Dblocks(graph, out);
    const auto& mapping = graph.getTileMapping(out);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l1dist"});
    const auto vertexName = poputil::templateVertex("L1DistanceSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a_index = j / n;
                size_t b_index = j % n;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a_index]);
                graph.connect(v["b"], b[b_index]);
                graph.connect(v["out"], out[a_index][b_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return out;
}

poplar::Tensor l1distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(1) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0)) {
        std::ostringstream msg;
        msg << "Bad arguments to l1distancegrad, expected"
            << " a.shape (M, K), b.shape (N, K), gradOutput.shape (M, N), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString()
            << ", gradOutput.shape = " << gradOutput.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t k = a.dim(1);
    poplar::Tensor grad =
        graph.addVariable(a.elementType(), a.shape(), {debugContext, "l1dist_grad"});
    mapTensor2Dblocks(graph, grad);
    const auto& mapping = graph.getTileMapping(grad);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l1dist_grad"});
    const auto vertexName = poputil::templateVertex("L1DistanceGradSingleVertex", a.elementType());
    std::map<std::pair<size_t, size_t>, poplar::Tensor> bCache, gradCache;
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a1_index = j / k;
                size_t a2_index = j % k;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a1_index][a2_index]);
                graph.connect(
                    v["b"], getCachedCopy(bCache, graph, i, a2_index,
                                          b.slice({a2_index, a2_index + 1}, 1).squeeze({1}), prog));
                graph.connect(v["gradOutput"], getCachedCopy(gradCache, graph, i, a1_index,
                                                             gradOutput[a1_index], prog));
                graph.connect(v["grad"], grad[a1_index][a2_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return grad;
}

poplar::Tensor l2distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        std::ostringstream msg;
        msg << "Bad arguments to l2distance, expected a.shape (M, K), b.shape (N, K), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t n = b.dim(0);
    poplar::Tensor out =
        graph.addVariable(a.elementType(), {a.dim(0), b.dim(0)}, {debugContext, "l2dist_out"});
    mapTensor2Dblocks(graph, out);
    const auto& mapping = graph.getTileMapping(out);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l2dist"});
    const auto vertexName = poputil::templateVertex("L2DistanceSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a_index = j / n;
                size_t b_index = j % n;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a_index]);
                graph.connect(v["b"], b[b_index]);
                graph.connect(v["out"], out[a_index][b_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return out;
}

poplar::Tensor l2distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& dist,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || dist.rank() != 2 || gradOutput.rank() != 2 ||
        a.dim(1) != b.dim(1) || gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0) ||
        dist.dim(0) != a.dim(0) || dist.dim(1) != b.dim(0)) {
        std::ostringstream msg;
        msg << "Bad arguments to l2distancegrad, expected"
            << " a.shape (M, K), b.shape (N, K), gradOutput.shape (M, N), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString()
            << ", gradOutput.shape = " << gradOutput.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t k = a.dim(1);
    poplar::Tensor grad =
        graph.addVariable(a.elementType(), a.shape(), {debugContext, "l2dist_grad"});
    mapTensor2Dblocks(graph, grad);
    const auto& mapping = graph.getTileMapping(grad);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l2dist_grad"});
    const auto vertexName = poputil::templateVertex("L2DistanceGradSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a1_index = j / k;
                size_t a2_index = j % k;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a1_index][a2_index]);
                graph.connect(v["b"], b.slice({a2_index, a2_index + 1}, 1).squeeze({1}));
                graph.connect(v["dist"], dist[a1_index]);
                graph.connect(v["gradOutput"], gradOutput[a1_index]);
                graph.connect(v["grad"], grad[a1_index][a2_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return grad;
}

}  // namespace poplar_extensions
