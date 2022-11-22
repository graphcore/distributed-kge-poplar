// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <catch2/catch.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <popops/codelets.hpp>

#include "fructose/frnn.hpp"
#include "fructose/fructose.hpp"

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

TEST_CASE("fr::nn::softmaxCrossEntropy", "[fr]") {
    TestHelper test;
    popops::addCodelets(test.rootFrame.graph.poplar());

    auto xent = fr::nn::softmaxCrossEntropy(fr::ops::constant<float>({-100, 2, 2, 2, 2}),
                                            fr::ops::constant(1u));
    fr::ops::output("xent", xent);

    auto result = test.loadAndRun();
    REQUIRE(result["xent"].scalar() == Approx(std::log(4)));
}

TEST_CASE("fructose basics", "[fr]") {
    TestHelper test;

    auto a = fr::ops::variable("a", {{}, poplar::FLOAT});
    auto b = a + a;
    auto c = b + b;
    c.backward();
    fr::ops::output("c", c);
    fr::ops::output("grad_a", a.grad());
    fr::ops::output("grad_b", b.grad());
    a.hostAccess();

    test.load();
    test.engine->writeTensor<float>(a.name(), {10.0f});
    auto result = test.run();
    REQUIRE(result["c"].scalar() == Approx(40.0f));
    REQUIRE(result["grad_a"].scalar() == Approx(4.0f));
    REQUIRE(result["grad_b"].scalar() == Approx(2.0f));
}

namespace {
std::vector<unsigned> getTensorToTileMapping(const fr::Tensor& tensor) {
    auto& frame = fr::Environment::frame();
    auto tileMapping = frame.graph.poplar().getTileMapping(frame.graph.unwrap(tensor.pag()));
    std::vector<unsigned> tensorMapping(tensor.numElements());
    for (auto tile = 0u; tile < tileMapping.size(); ++tile) {
        for (auto interval : tileMapping[tile]) {
            for (auto i = interval.begin(); i < interval.end(); ++i) {
                tensorMapping[i] = tile;
            }
        }
    }
    return tensorMapping;
}
}  // namespace

TEST_CASE("fr::mapping::OneTile", "[fr]") {
    fr::RootFrame frame(poplar::Target::createIPUTarget(1, 6, "IPU-POD16"));

    auto a = fr::ops::variable("a", {{2}, poplar::FLOAT});
    fr::mapping::setDefault(fr::mapping::OneTile(), {a});  // default = -1
    REQUIRE(getTensorToTileMapping(a) == std::vector<unsigned>{5, 5});

    auto b = fr::ops::variable("b", {{2}, poplar::FLOAT});
    fr::mapping::setDefault(fr::mapping::OneTile(9), {b});  // wraparound
    REQUIRE(getTensorToTileMapping(b) == std::vector<unsigned>{3, 3});

    auto c = fr::ops::variable("c", {{2}, poplar::FLOAT});
    fr::mapping::setDefault(fr::mapping::OneTile(-8), {c});  // negative wraparound
    REQUIRE(getTensorToTileMapping(c) == std::vector<unsigned>{4, 4});
}

TEST_CASE("fr::Buffer", "[fr]") {
    TestHelper test;

    fr::Buffer buf("buf", {{3, 1, 4}, poplar::FLOAT});
    buf.write(fr::ops::constant<float>({100, 101, 102, 103,  //
                                        200, 201, 202, 203,  //
                                        300, 301, 302, 303},
                                       {{3, 1, 4}}),
              fr::ops::constant<unsigned>({2, 0, 1}));

    auto out = buf.read(fr::ops::constant<unsigned>({0u, 2u}));
    REQUIRE(out.shape() == std::vector<size_t>({2, 1, 4}));
    fr::ops::output("out", out);

    using Catch::Matchers::Approx;
    auto result = test.loadAndRun();
    REQUIRE_THAT(result["out"].data, Approx<float>({200, 201, 202, 203,  //
                                                    100, 101, 102, 103}));
}
