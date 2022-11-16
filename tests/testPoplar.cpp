// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <catch2/catch.hpp>

#include <poplar/Engine.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

/**
 * Not a 'real' test case, but somewhere to play around with Poplar directly.
 */
TEST_CASE("Manual poplar", "[poplar]") {
    auto device = poplar::Device::createCPUDevice();

    poplar::Graph graph(device.getTarget());
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    poplar::program::Sequence prog;

    auto a =
        graph.addConstant<float>(poplar::FLOAT, {1, 2, 5}, std::vector<float>(1 * 2 * 5, 1.0f));
    graph.setTileMapping(a, 0);
    auto b =
        graph.addConstant<float>(poplar::FLOAT, {1, 5, 7}, std::vector<float>(1 * 5 * 7, 1.0f));
    graph.setTileMapping(b, 0);
    auto c = poplin::matMulGrouped(graph, a, b, prog, poplar::FLOAT);

    // prog.add(poplar::program::PrintTensor("c", c));

    poplar::Engine engine(graph, prog);
    engine.loadAndRun(device);
}
