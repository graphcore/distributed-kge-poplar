// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <catch2/catch.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <popops/codelets.hpp>

#include "fructose/fructose.hpp"
#include "poplar_kge.hpp"

namespace {
struct HostTensor {
    std::vector<size_t> shape;
    std::vector<float> data;

    HostTensor() = default;
    HostTensor(const std::vector<size_t>& shape)
        : shape(shape), data(fr::util::numElements(shape)) {}

    float scalar() const {
        assert(shape.size() == 0);
        return data.front();
    }
};
struct TestHelper {
    poplar::Device device;
    fr::RootFrame rootFrame;
    std::unique_ptr<poplar::Engine> engine;

    explicit TestHelper(unsigned nIpus = 1)
        : device(poplar::Device::createCPUDevice(nIpus)), rootFrame(device.getTarget()) {}

    void load() {
        engine.reset(new poplar::Engine(rootFrame.graph.poplar(), rootFrame.tape.prog()));
        engine->load(device);
    }
    std::unordered_map<std::string, HostTensor> run() {
        std::unordered_map<std::string, HostTensor> result;
        for (auto& item : rootFrame.streams) {
            assert(item.second.spec().dtype == poplar::FLOAT);
            result.insert({item.first, HostTensor(item.second.spec().shape)});
            engine->connectStream(item.first, result[item.first].data.data());
        }
        engine->run();
        return result;
    }
    std::unordered_map<std::string, HostTensor> loadAndRun() {
        load();
        return run();
    }
};
}  // namespace

TEST_CASE("poplar_kge::detachedSoftmax", "[poplar_kge]") {
    TestHelper test;
    popops::addCodelets(test.rootFrame.graph.poplar());

    auto xent =
        poplar_kge::detachedSoftmax(fr::ops::constant<float>({1, 1, 2, 2, 4, 6}).reshape({2, 3}));
    fr::ops::output("xent", xent);

    using Catch::Matchers::Approx;
    auto result = test.loadAndRun();
    auto norm1 = std::exp(1.0f) + std::exp(1.0f) + std::exp(2.0f);
    auto norm2 = std::exp(2.0f) + std::exp(4.0f) + std::exp(6.0f);
    REQUIRE_THAT(
        result["xent"].data,
        Approx<float>({std::exp(1.0f) / norm1, std::exp(1.0f) / norm1, std::exp(2.0f) / norm1,
                       std::exp(2.0f) / norm2, std::exp(4.0f) / norm2, std::exp(6.0f) / norm2}));
}
