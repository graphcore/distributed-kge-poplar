// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <unordered_map>

#include <catch2/catch.hpp>

#include <poplar/CSRFunctions.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplin/codelets.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include <poputil/TileMapping.hpp>

#include "pag/pag.hpp"

namespace {
struct HostTensor {
    std::vector<size_t> shape;
    std::vector<float> data;

    float scalar() const {
        assert(shape.size() == 0);
        return data.front();
    }
};

unsigned numElements(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<size_t>());
}

template <class Collection, class Func>
auto mapVector(const Collection& items, Func&& func)
    -> std::vector<decltype(func(*items.begin()))> {
    std::vector<decltype(func(*items.begin()))> result;
    result.reserve(items.size());
    std::transform(items.begin(), items.end(), std::back_inserter(result),
                   std::forward<Func>(func));
    return result;
}

poplar::Device attach(unsigned nDevice, poplar::TargetType preferredType) {
    if (preferredType != poplar::TargetType::CPU) {
        auto devices = poplar::DeviceManager().getDevices(preferredType, nDevice);
        for (auto&& device : devices) {
            if (device.attach()) {
                return std::move(device);
            }
        }
        WARN("Cannot attach to device, falling back to CPU");
    }
    return poplar::Device::createCPUDevice(nDevice);
}

struct TestHelper {
    poplar::Device device;
    poplar::Graph poplarGraph;
    pag::Graph graph;
    pag::Tape tape;
    std::unordered_map<std::string, std::vector<size_t>> readTensors;
    std::unique_ptr<poplar::Engine> engine;

    // Enables exceptions by default, but these will only occur on IPU/IPU_MODEL
    explicit TestHelper(unsigned nDevice = 1u,
                        poplar::TargetType preferredType = poplar::TargetType::IPU_MODEL)
        : device(attach(nDevice, preferredType)),
          poplarGraph(device, poplar::replication_factor(nDevice)),
          graph(poplarGraph) {
        popops::addCodelets(poplarGraph);
        poplin::addCodelets(poplarGraph);
        poprand::addCodelets(poplarGraph);
        poplarGraph.addCodelets("build/poplar_extensions.gp");
        poplar::setFloatingPointBehaviour(
            poplarGraph, tape.prog(),
            {/*inv*/ true, /*div*/ true, /*oflo*/ true, /*esr*/ false, /*nanoo*/ false});
    }

    template <class T>
    pag::Tensor tensor(const std::vector<size_t>& shape,
                       const std::vector<T>& data,
                       poplar::Type type = poplar::equivalent_device_type<T>().value) {
        auto tensor = poplarGraph.addConstant<T>(type, shape, data);
        poplarGraph.setTileMapping(tensor, 0u);
        return graph.wrap(tensor, true);
    }
    template <class T>
    pag::Tensor ones(const std::vector<size_t>& shape) {
        return tensor<T>(shape, std::vector<T>(numElements(shape), 1));
    }
    pag::Tensor scalar(float value) { return tensor<float>({}, {value}); }

    void read(const std::string& name, const pag::Tensor& tensor) {
        return read(name, graph.unwrap(tensor));
    }
    void read(const std::string& name, poplar::Tensor tensor) {
        if (tensor.elementType() == poplar::HALF) {
            tensor = popops::cast(poplarGraph, tensor, poplar::FLOAT, tape.prog());
        }
        assert(tensor.elementType() == poplar::FLOAT);
        poplarGraph.createHostRead(name, tensor);
        readTensors[name] = tensor.shape();
    }

    std::unordered_map<std::string, HostTensor> run() {
        engine = std::make_unique<poplar::Engine>(poplarGraph, tape.prog());
        engine->loadAndRun(device);

        std::unordered_map<std::string, HostTensor> reads;
        for (auto& item : readTensors) {
            auto& hostTensor = reads[item.first];
            if (poplarGraph.getReplicationFactor() == 1) {
                hostTensor.shape = item.second;
            } else {
                hostTensor.shape.resize(1 + item.second.size());
                hostTensor.shape[0] = poplarGraph.getReplicationFactor();
                std::copy(item.second.begin(), item.second.end(), hostTensor.shape.begin() + 1);
            }
            hostTensor.data.resize(numElements(hostTensor.shape));
            engine->readTensor(item.first, hostTensor.data.data(),
                               hostTensor.data.data() + hostTensor.data.size());
        }
        return reads;
    }
};
}  // namespace

TEST_CASE("pag::ops::identity", "[pag]") {
    TestHelper test;

    auto poplarConstant = test.poplarGraph.addConstant<float>(poplar::FLOAT, {}, 10.0f);
    test.poplarGraph.setTileMapping(poplarConstant, 0u);
    auto constant = test.graph.wrap(poplarConstant, false);
    auto withGrad = pag::ops::identity(test.graph, constant, /*requiresGrad*/ true, test.tape);
    test.tape.backward(test.graph, pag::ops::neg(test.graph, withGrad, test.tape));

    test.read("value", withGrad);
    test.read("grad", test.graph.grad(withGrad));

    auto out = test.run();

    REQUIRE(out["value"].scalar() == Approx(10.0f));
    REQUIRE(out["grad"].scalar() == Approx(-1.0f));
}

TEST_CASE("pag::ops::add", "[pag]") {
    TestHelper test;

    auto a = test.scalar(10.0f);
    auto b = pag::ops::add(test.graph, a, a, test.tape, "add1");
    auto c = pag::ops::add(test.graph, b, a, test.tape, "add2");
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_b", test.graph.grad(b));
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    REQUIRE(out["c"].scalar() == Approx(30.0f));
    REQUIRE(out["grad_b"].scalar() == Approx(1.0f));
    REQUIRE(out["grad_a"].scalar() == Approx(3.0f));
}

TEST_CASE("pag::ops::add(broadcast)", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({2}, {10.0f, 20.0f});
    auto b = test.tensor<float>({1, 2}, {0.5f, 0.5f});
    auto c = test.tensor<float>({3, 2}, {0, 1, 2, 3, 4, 5});
    auto y = pag::ops::add(test.graph, a, c, test.tape, "ac");
    y = pag::ops::add(test.graph, b, y, test.tape, "bc");
    test.tape.backward(test.graph, y);

    test.read("y", y);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));
    test.read("grad_c", test.graph.grad(c));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["y"].data, Approx<float>({10.5, 21.5, 12.5, 23.5, 14.5, 25.5}));

    REQUIRE(out["grad_a"].shape == std::vector<size_t>({2}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({3, 3}));

    REQUIRE(out["grad_b"].shape == std::vector<size_t>({1, 2}));
    REQUIRE_THAT(out["grad_b"].data, Approx<float>({3, 3}));

    REQUIRE(out["grad_c"].shape == std::vector<size_t>({3, 2}));
    REQUIRE_THAT(out["grad_c"].data, Approx<float>({1, 1, 1, 1, 1, 1}));
}

TEST_CASE("pag::ops::sub", "[pag]") {
    TestHelper test;

    auto a = test.scalar(10.0f);
    auto b = test.scalar(1.0f);
    auto c = pag::ops::sub(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c, test.graph.unwrap(test.scalar(3.0f)));

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    REQUIRE(out["c"].scalar() == Approx(9.0f));
    REQUIRE(out["grad_a"].scalar() == Approx(3.0f));
    REQUIRE(out["grad_b"].scalar() == Approx(-3.0f));
}

TEST_CASE("pag::ops::mul", "[pag]") {
    TestHelper test;

    auto a = test.scalar(2.0f);
    auto b = test.scalar(3.0f);
    auto c = pag::ops::mul(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c, test.graph.unwrap(test.scalar(5.0f)));

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    REQUIRE(out["c"].scalar() == Approx(6.0f));
    REQUIRE(out["grad_a"].scalar() == Approx(15.0f));
    REQUIRE(out["grad_b"].scalar() == Approx(10.0f));
}

TEST_CASE("pag::ops::div", "[pag]") {
    TestHelper test;

    auto a = test.scalar(2.0f);
    auto b = test.scalar(3.0f);
    auto c = pag::ops::div(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c, test.graph.unwrap(test.scalar(5.0f)));

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    REQUIRE(out["c"].scalar() == Approx(2.0f / 3));
    REQUIRE(out["grad_a"].scalar() == Approx(5.0f / 3));
    REQUIRE(out["grad_b"].scalar() == Approx(-5 * 2.0f / 9));
}

TEST_CASE("pag::ops::neg", "[pag]") {
    TestHelper test;

    auto a = test.scalar(5.0f);
    auto b = pag::ops::neg(test.graph, a, test.tape);
    test.tape.backward(test.graph, b, test.graph.unwrap(test.scalar(-3.0f)));

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    REQUIRE(out["b"].scalar() == Approx(-5.0f));
    REQUIRE(out["grad_a"].scalar() == Approx(3.0f));
}

TEST_CASE("pag::ops::abs", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {-2, -3, 2});
    auto b = pag::ops::abs(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({2, 3, 2}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({-1, -1, 1}));
}

TEST_CASE("pag::ops::l1dist", "[pag]") {
    auto dtype = GENERATE(as<poplar::Type>(), poplar::FLOAT, poplar::HALF);

    TestHelper test(1, poplar::TargetType::IPU);
    // Note - it's important to test zero distance
    std::vector<float> aData(2 * 5);
    std::vector<float> bData(4 * 5);
    std::iota(aData.begin(), aData.end(), 0.0);
    std::iota(bData.begin(), bData.end(), 0.0f);
    auto a = test.tensor<float>({2, 5}, aData, dtype);
    auto b = test.tensor<float>({4, 5}, bData, dtype);
    auto c = pag::ops::l1distance(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["c"].data, Approx<float>({0.0f, 25.0f, 50.0f, 75.0f,  //
                                               25.0f, 0.0f, 25.0f, 50.0f}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({-3.0f, -3.0f, -3.0f, -3.0f, -3.0f,  //
                                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}));
    REQUIRE_THAT(out["grad_b"].data, Approx<float>({-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  //
                                                    1.0f,  1.0f,  1.0f,  1.0f,  1.0f,   //
                                                    2.0f,  2.0f,  2.0f,  2.0f,  2.0f,   //
                                                    2.0f,  2.0f,  2.0f,  2.0f,  2.0f}));
}

TEST_CASE("pag::ops::l1dist/benchmark", "[pag][benchmark]") {
    TestHelper test(1, poplar::TargetType::IPU);
    for (auto dtype : {poplar::FLOAT, poplar::HALF}) {
        auto poplarA = test.poplarGraph.addConstant(dtype, {512, 256}, 0.0f);
        poputil::mapTensorLinearly(test.poplarGraph, poplarA);
        auto poplarB = test.poplarGraph.addConstant(dtype, {16 * 96, 256}, 2.0f);
        poputil::mapTensorLinearly(test.poplarGraph, poplarB);
        pag::ops::l1distance(test.graph, test.graph.wrap(poplarA, true),
                             test.graph.wrap(poplarB, true), test.tape);
    }
    test.run();
}

TEST_CASE("pag::ops::l2dist", "[pag]") {
    auto dtype = GENERATE(as<poplar::Type>(), poplar::FLOAT, poplar::HALF);

    TestHelper test(1, poplar::TargetType::IPU);
    // Note - it's important to test zero distance
    std::vector<float> aData(2 * 5);
    std::vector<float> bData(4 * 5);
    std::iota(aData.begin(), aData.end(), 0.0f);
    std::iota(bData.begin(), bData.end(), 0.0f);
    auto a = test.tensor<float>({2, 5}, aData, dtype);
    auto b = test.tensor<float>({4, 5}, bData, dtype);
    auto c = pag::ops::l2distance(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    using Catch::Matchers::Approx;
    float margin = dtype == poplar::HALF ? 1e-2f : 1e-4f;
    REQUIRE_THAT(out["c"].data, Approx<float>({0.f, 11.1803f, 22.3607f, 33.5410f,  //
                                               11.1803f, 0.f, 11.1803f, 22.3607f})
                                    .margin(margin));
    REQUIRE_THAT(out["grad_a"].data,
                 Approx<float>({-1.3416f, -1.3416f, -1.3416f, -1.3416f, -1.3416f,  //
                                -0.4472f, -0.4472f, -0.4472f, -0.4472f, -0.4472f})
                     .margin(margin));
    REQUIRE_THAT(out["grad_b"].data,
                 Approx<float>({-0.4472f, -0.4472f, -0.4472f, -0.4472f, -0.4472f,  //
                                0.4472f,  0.4472f,  0.4472f,  0.4472f,  0.4472f,   //
                                0.8944f,  0.8944f,  0.8944f,  0.8944f,  0.8944f,   //
                                0.8944f,  0.8944f,  0.8944f,  0.8944f,  0.8944f})
                     .margin(margin));
}

TEST_CASE("pag::ops::l2dist-large", "[pag]") {
    TestHelper test(1, poplar::TargetType::IPU);

    std::vector<float> aData(2 * 5);
    std::vector<float> bData(4 * 5);
    std::iota(aData.begin(), aData.end(), 0.0f);
    std::iota(bData.begin(), bData.end(), 0.0f);
    auto a = test.tensor<float>({1, 5}, {100.0f, 200.0f, 300.0f, 400.0f, 500.0f}, poplar::HALF);
    auto b = test.tensor<float>({1, 5}, {600.0f, 600.0f, 600.0f, 600.0f, 600.0f}, poplar::HALF);
    auto c = pag::ops::l2distance(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["c"].data, Approx<float>({741.62f}).margin(0.5f));
    REQUIRE_THAT(out["grad_a"].data,
                 Approx<float>({-0.6742, -0.5394, -0.4045, -0.2697, -0.1348}).margin(1e-3f));
}

TEST_CASE("pag::ops::cast", "[pag]") {
    TestHelper test;

    auto a = test.scalar(5.0f);
    auto b = pag::ops::cast(test.graph, a, poplar::HALF, test.tape);
    REQUIRE(test.graph.unwrap(a).elementType() == poplar::FLOAT);
    REQUIRE(test.graph.unwrap(b).elementType() == poplar::HALF);
    // fails with wrong gradient type
    CHECK_THROWS(test.tape.backward(
        test.graph, b, test.graph.unwrap(test.tensor<float>({}, {-3.0f}, poplar::FLOAT))));

    test.tape.backward(test.graph, b,
                       test.graph.unwrap(test.tensor<float>({}, {-3.0f}, poplar::HALF)));
    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));
    REQUIRE(test.graph.grad(a).elementType() == poplar::FLOAT);
    REQUIRE(test.graph.grad(b).elementType() == poplar::HALF);

    auto out = test.run();

    REQUIRE(out["b"].scalar() == Approx(5.0f));
    REQUIRE(out["grad_a"].scalar() == Approx(-3.0f));
}

TEST_CASE("pag::ops::max", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({2, 3}, {10, 20, 30, 40, 50, 60});
    auto b = test.tensor<float>({3}, {5, 35, 65});
    auto c = pag::ops::max(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["c"].data, Approx<float>({10, 35, 65, 40, 50, 65}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({1, 0, 0, 1, 1, 0}));
    REQUIRE_THAT(out["grad_b"].data, Approx<float>({0, 1, 2}));
}

TEST_CASE("pag::ops::square", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {1, -2, 3});
    auto b = pag::ops::square(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({1, 4, 9}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({2, -4, 6}));
}

TEST_CASE("pag::ops::pow-exponent>1", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {1, -2, 3});
    auto b = pag::ops::pow(test.graph, a, 3, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({1, -8, 27}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({3, 12, 27}));
}

TEST_CASE("pag::ops::pow-exponent<1", "[pag]") {
    TestHelper test(1, poplar::TargetType::IPU);

    auto a = test.tensor<float>({4}, {0, 8, 27, 64});
    auto b = pag::ops::pow(test.graph, a, 1. / 3., test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({0, 2, 3, 4}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({0, 1. / 12., 1. / 27., 1. / 48.}));
}

TEST_CASE("pag::ops::sqrt", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {1, 4, 9});
    auto b = pag::ops::sqrt(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({1, 2, 3}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({0.5, 0.25, 1. / 6.}));
}

TEST_CASE("pag::ops::cbrt", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {8, 27, 64});
    auto b = pag::ops::cbrt(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({2, 3, 4}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({1. / 12., 1. / 27., 1. / 48.}));
}

TEST_CASE("pag::ops::sin", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {0, M_PI_4, M_PI_2});
    auto b = pag::ops::sin(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({0.0f, M_SQRT1_2, 1.0f}));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({1.0f, M_SQRT1_2, -0.0f}).margin(1e-7));
}

TEST_CASE("pag::ops::cos", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {0, M_PI_4, M_PI_2});
    auto b = pag::ops::cos(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx<float>({1.0f, M_SQRT1_2, -0.0f}).margin(1e-7));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>({-0.0, -M_SQRT1_2, -1.0f}).margin(1e-12));
}

TEST_CASE("pag::ops::dropout", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({10}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
    auto b = pag::ops::dropout(test.graph, a, 0.25, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Equals;

    auto out_vec = out["b"].data;
    sort(out_vec.begin(), out_vec.end());
    out_vec.erase(unique(out_vec.begin(), out_vec.end()), out_vec.end());
    REQUIRE_THAT(out_vec, Equals<float>({0, 4}));

    auto grad_vec = out["grad_a"].data;
    sort(grad_vec.begin(), grad_vec.end());
    grad_vec.erase(unique(grad_vec.begin(), grad_vec.end()), grad_vec.end());
    REQUIRE_THAT(grad_vec, Equals<float>({0, 4. / 3.}));
}

TEST_CASE("pag::ops::matMul", "[pag]") {
    TestHelper test;

    auto a = test.ones<float>({2, 3});
    auto b = test.ones<float>({3, 5});
    auto c = pag::ops::matMul(test.graph, a, b, test.tape);
    test.tape.backward(test.graph, c);

    test.read("c", c);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["c"].data, Approx<float>(std::vector<float>(2 * 5, 3.0f)));
    REQUIRE_THAT(out["grad_a"].data, Approx<float>(std::vector<float>(2 * 3, 5.0f)));
    REQUIRE_THAT(out["grad_b"].data, Approx<float>(std::vector<float>(3 * 5, 2.0f)));
}

TEST_CASE("pag::ops::multiSlice", "[pag]") {
    TestHelper test;

    auto table = test.tensor<float>({3, 2}, {10, 11, 20, 21, 30, 31});
    auto idx = test.tensor<uint32_t>({4, 1}, {1, 1, 0, 1});
    auto resultGrad = test.tensor<float>({4, 1, 2}, {0.1, 1, 0.2, 2, 0.3, 3, 0.4, 4});

    auto plan = popops::embedding::plan(
        test.poplarGraph, poplar::FLOAT, test.graph.unwrap(table).dim(0),
        test.graph.unwrap(table).dim(1), {test.graph.unwrap(idx).dim(0)}, {});

    auto result = pag::ops::multiSlice(test.graph, table, idx, {0u}, {1u}, test.tape, plan, {});

    test.tape.backward(test.graph, result, test.graph.unwrap(resultGrad));

    test.read("result", result);
    test.read("grad_table", test.graph.grad(table));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["result"].shape == std::vector<size_t>({4, 1, 2}));
    REQUIRE_THAT(out["result"].data, Approx<float>({20, 21, 20, 21, 10, 11, 20, 21}));

    REQUIRE(out["grad_table"].shape == test.graph.unwrap(table).shape());
    REQUIRE_THAT(out["grad_table"].data, Approx<float>({0.3, 3, 0.7, 7, 0, 0}));
}

TEST_CASE("pag::ops::reduce", "[pag]") {
    TestHelper test;

    auto a = test.ones<float>({2, 3, 4, 5});
    auto b = pag::ops::reduce(test.graph, a, {3, 1}, popops::Operation::ADD, test.tape);
    auto gradB = test.graph.unwrap(test.tensor<float>({2, 4}, {1, 2, 3, 4, 10, 20, 30, 40}));
    test.tape.backward(test.graph, b, gradB);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    REQUIRE_THAT(out["b"].data, Catch::Matchers::Approx<float>({15, 15, 15, 15, 15, 15, 15, 15}));
    auto gradA = out["grad_a"];
    REQUIRE(gradA.shape == test.graph.unwrap(a).shape());
    REQUIRE(gradA.data[0] == Approx(1.0f));
    REQUIRE(gradA.data[5] == Approx(2.0f));
    REQUIRE(gradA.data[2 * 3 * 4 * 5 - 1] == Approx(40.0f));
}

TEST_CASE("pag::ops::transpose", "[pag]") {
    TestHelper test;

    auto tensor = test.tensor<float>({2, 3}, {11, 12, 13, 21, 22, 23});
    auto transposed = pag::ops::transpose(test.graph, tensor, test.tape);
    test.tape.backward(test.graph, transposed);

    test.read("transposed", transposed);
    test.read("grad", test.graph.grad(tensor));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["transposed"].shape == std::vector<size_t>({3, 2}));
    REQUIRE_THAT(out["transposed"].data, Approx<float>({11, 21, 12, 22, 13, 23}));
    REQUIRE(out["grad"].shape == std::vector<size_t>({2, 3}));
}

TEST_CASE("pag::ops::reshape", "[pag]") {
    TestHelper test;

    auto tensor = test.tensor<float>({2, 1, 2}, {10, 11, 20, 21});
    auto reshaped = pag::ops::reshape(test.graph, tensor, {1, 2, 2}, test.tape);
    test.tape.backward(test.graph, reshaped,
                       test.graph.unwrap(test.tensor<float>({1, 2, 2}, {1, 2, 3, 4})));

    test.read("reshaped", reshaped);
    test.read("grad_tensor", test.graph.grad(tensor));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["reshaped"].shape == std::vector<size_t>({1, 2, 2}));
    REQUIRE_THAT(out["reshaped"].data, Approx<float>({10, 11, 20, 21}));

    REQUIRE(out["grad_tensor"].shape == std::vector<size_t>({2, 1, 2}));
    REQUIRE_THAT(out["grad_tensor"].data, Approx<float>({1, 2, 3, 4}));
}

TEST_CASE("pag::ops::slice", "[pag]") {
    TestHelper test;

    auto tensor = test.tensor<float>({1, 5, 1}, {100, 200, 300, 400, 500});
    auto sliced = pag::ops::slice(test.graph, tensor, 1u, {3, 5}, test.tape);
    test.tape.backward(test.graph, sliced);

    test.read("sliced", sliced);
    test.read("grad", test.graph.grad(tensor));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["sliced"].shape == std::vector<size_t>({1, 2, 1}));
    REQUIRE_THAT(out["sliced"].data, Approx<float>({400, 500}));

    REQUIRE(out["grad"].shape == std::vector<size_t>({1, 5, 1}));
    REQUIRE_THAT(out["grad"].data, Approx<float>({0, 0, 0, 1, 1}));
}

TEST_CASE("pag::ops::concat", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({1, 2, 1}, {100, 200});
    auto b = test.tensor<float>({1, 1, 1}, {300});
    auto c = test.tensor<float>({1, 2, 1}, {400, 500});
    auto concat = pag::ops::concat(test.graph, {a, b, c}, 1u, test.tape);

    auto gradConcat = test.graph.unwrap(test.tensor<float>({1, 5, 1}, {10, 20, 30, 40, 50}));
    test.tape.backward(test.graph, concat, gradConcat);

    test.read("concat", concat);
    test.read("grad_a", test.graph.grad(a));
    test.read("grad_b", test.graph.grad(b));
    test.read("grad_c", test.graph.grad(c));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["concat"].shape == std::vector<size_t>({1, 5, 1}));
    REQUIRE_THAT(out["concat"].data, Approx<float>({100, 200, 300, 400, 500}));

    REQUIRE_THAT(out["grad_a"].data, Approx<float>({10, 20}));
    REQUIRE_THAT(out["grad_b"].data, Approx<float>({30}));
    REQUIRE_THAT(out["grad_c"].data, Approx<float>({40, 50}));
}

TEST_CASE("pag::ops::split", "[pag]") {
    TestHelper test;

    auto tensor = test.tensor<float>({1, 5, 1}, {100, 200, 300, 400, 500});
    auto parts = pag::ops::split(test.graph, tensor, 1, {2, 1, 2}, test.tape);
    test.graph.addGrad(parts[0], test.graph.unwrap(test.tensor<float>({1, 2, 1}, {600, 700})),
                       test.tape.prog(), {});
    test.graph.addGrad(parts[1], test.graph.unwrap(test.tensor<float>({1, 1, 1}, {800})),
                       test.tape.prog(), {});
    test.graph.addGrad(parts[2], test.graph.unwrap(test.tensor<float>({1, 2, 1}, {900, 1000})),
                       test.tape.prog(), {});

    test.tape.backward(test.graph);

    test.read("parts[0]", parts[0]);
    test.read("parts[1]", parts[1]);
    test.read("parts[2]", parts[2]);
    test.read("grad", test.graph.grad(tensor));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE(out["parts[0]"].shape == std::vector<size_t>({1, 2, 1}));
    REQUIRE_THAT(out["parts[0]"].data, Approx<float>({100, 200}));

    REQUIRE(out["parts[1]"].shape == std::vector<size_t>({1, 1, 1}));
    REQUIRE_THAT(out["parts[1]"].data, Approx<float>({300}));

    REQUIRE(out["parts[2]"].shape == std::vector<size_t>({1, 2, 1}));
    REQUIRE_THAT(out["parts[2]"].data, Approx<float>({400, 500}));

    REQUIRE(out["grad"].shape == std::vector<size_t>({1, 5, 1}));
    REQUIRE_THAT(out["grad"].data, Approx<float>({600, 700, 800, 900, 1000}));
}

///////////////////////////////////////////////////////////////////////////////
// Neural Networks

TEST_CASE("pag::ops::logSoftmax", "[pag]") {
    TestHelper test;

    auto a = test.tensor<float>({3}, {1, 1, 2});
    auto b = pag::ops::logSoftmax(test.graph, a, test.tape);
    auto gradB = test.graph.unwrap(test.tensor<float>({3}, {10, 15, 5}));
    test.tape.backward(test.graph, b, gradB);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    auto norm = std::log(std::exp(1.0f) + std::exp(1.0f) + std::exp(2.0f));
    REQUIRE_THAT(out["b"].data, Approx<float>({1 - norm, 1 - norm, 2 - norm}));
    REQUIRE_THAT(out["grad_a"].data,
                 Approx<float>({10 - 30 * std::exp(1 - norm), 15 - 30 * std::exp(1 - norm),
                                5 - 30 * std::exp(2 - norm)}));
}

TEST_CASE("pag::ops::sigmoid", "[pag]") {
    std::vector<float> input({-100, -1, 0, 1, 100});
    std::vector<float> expectedOutput({0, 0.26894143f, 0.5f, 0.7310586f, 1.0f});
    std::vector<float> expectedGrad({0, 0.19661194f, 0.25f, 0.19661193f, 0.0f});

    TestHelper test;

    auto a = test.tensor({input.size()}, input);
    auto b = pag::ops::sigmoid(test.graph, a, test.tape);
    test.tape.backward(test.graph, b);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx(expectedOutput).margin(1e-3f));
    REQUIRE_THAT(out["grad_a"].data, Approx(expectedGrad).margin(1e-3f));
}

TEST_CASE("pag::ops::logSigmoid", "[pag]") {
    auto dtype = GENERATE(as<poplar::Type>(), poplar::FLOAT, poplar::HALF);

    std::vector<float> input({-100, -12, -1, 0, 1, 12, 100});
    auto expectedOutput = mapVector(input, [](float x) {
        return static_cast<float>(-std::log(1 + std::exp(static_cast<double>(-x))));
    });
    std::vector<float> outputGrad(input.size(), 10.0f);
    auto expectedGrad = mapVector(input, [](float x) {
        return static_cast<float>(10.0 / (1 + std::exp(static_cast<double>(x))));
    });

    // Half-precision behaviour is most reliable on IPU
    TestHelper test(1u, poplar::TargetType::IPU);

    auto a = test.tensor({input.size()}, input, dtype);
    auto b = pag::ops::logSigmoid(test.graph, a, test.tape);
    auto gradB = test.graph.unwrap(test.tensor({outputGrad.size()}, outputGrad, dtype));
    test.tape.backward(test.graph, b, gradB);

    test.read("b", b);
    test.read("grad_a", test.graph.grad(a));

    auto out = test.run();

    using Catch::Matchers::Approx;
    REQUIRE_THAT(out["b"].data, Approx(expectedOutput).margin(1e-2f));
    REQUIRE_THAT(out["grad_a"].data, Approx(expectedGrad).margin(1e-2f));
}

///////////////////////////////////////////////////////////////////////////////
// Collectives

namespace {
/**
 * Create a tensor with contents like out[i, j, k] = 100*(1+i) + 10*(1+j) + (1+k).
 *
 * Also includes the replia index (with an extra *10), e.g. on replica 3, out[0, 2] == 4013
 */
pag::Tensor indexTensor(TestHelper& test, const std::vector<size_t>& shape) {
    std::vector<float> data(numElements(shape));
    for (auto i = 0u; i < data.size(); ++i) {
        auto stride = 1u;
        for (auto j = 0u; j < shape.size(); ++j) {
            auto dim = shape[shape.size() - j - 1];
            data[i] += std::pow(10.0f, j) * (1 + (i / stride) % dim);
            stride *= dim;
        }
    }
    auto poplarData = test.poplarGraph.addConstant<float>(poplar::FLOAT, shape, data);
    test.poplarGraph.setTileMapping(poplarData, 0u);
    auto replicaIndex = test.poplarGraph.addReplicationIndexConstant();
    test.poplarGraph.setTileMapping(replicaIndex, 0u);

    namespace pe = popops::expr;
    auto result =
        popops::map(test.poplarGraph,
                    pe::_1 + (pe::Cast(pe::_2, poplar::FLOAT) + pe::Const(1.0f)) *
                                 pe::Const(std::pow<float, float>(10.0f, shape.size() + 1)),
                    {poplarData, replicaIndex}, test.tape.prog());
    return test.graph.wrap(result, /*requiresGrad*/ true);
}
}  // namespace

TEST_CASE("pag::ops::allToAllCrossReplica", "[pag]") {
    TestHelper test(/*nDevice*/ 2, poplar::TargetType::CPU);

    auto input = indexTensor(test, {2, 3});
    auto output = pag::ops::allToAllCrossReplica(test.graph, input, test.tape, {});
    test.tape.backward(test.graph, output,
                       popops::neg(test.poplarGraph, test.graph.unwrap(input), test.tape.prog()));

    test.read("output", output);
    test.read("grad(input)", test.graph.grad(input));
    auto out = test.run();

    REQUIRE(out["output"].shape == std::vector<size_t>({2, 2, 3}));
    REQUIRE(out["output"].data == std::vector<float>{1011, 1012, 1013, 2011, 2012, 2013,  //
                                                     1021, 1022, 1023, 2021, 2022, 2023});
    REQUIRE(out["grad(input)"].shape == std::vector<size_t>({2, 2, 3}));
    REQUIRE(out["grad(input)"].data ==
            std::vector<float>{-1011, -1012, -1013, -2011, -2012, -2013,  //
                               -1021, -1022, -1023, -2021, -2022, -2023});
}

TEST_CASE("pag::ops::reduceScatterCrossReplica", "[pag]") {
    TestHelper test(/*nDevice*/ 2, poplar::TargetType::CPU);

    auto input = indexTensor(test, {6});
    auto output = pag::ops::reduceScatterCrossReplica(test.graph, input,
                                                      gcl::CollectiveOperator::ADD, test.tape, {});
    test.tape.backward(test.graph, output, test.graph.unwrap(indexTensor(test, {3})));

    test.read("output", output);
    test.read("grad(input)", test.graph.grad(input));
    auto out = test.run();

    REQUIRE(out["output"].shape == std::vector<size_t>({2, 3}));
    REQUIRE(out["output"].data == std::vector<float>{302, 304, 306,  //
                                                     308, 310, 312});
    REQUIRE(out["grad(input)"].shape == std::vector<size_t>({2, 6}));
    REQUIRE(out["grad(input)"].data == std::vector<float>{101, 102, 103, 201, 202, 203,  //
                                                          101, 102, 103, 201, 202, 203});
}

TEST_CASE("pag::ops::allGatherCrossReplica", "[pag]") {
    TestHelper test(/*nDevice*/ 2, poplar::TargetType::CPU);

    auto input = indexTensor(test, {3, 1});
    auto output = pag::ops::allGatherCrossReplica(test.graph, input, test.tape, {});
    test.tape.backward(test.graph, output, test.graph.unwrap(indexTensor(test, {2, 3, 1})));

    test.read("output", output);
    test.read("grad(input)", test.graph.grad(input));
    auto out = test.run();

    REQUIRE(out["output"].shape == std::vector<size_t>({2, 2, 3, 1}));
    REQUIRE(out["output"].data == std::vector<float>{1011, 1021, 1031, 2011, 2021, 2031,  //
                                                     1011, 1021, 1031, 2011, 2021, 2031});
    REQUIRE(out["grad(input)"].shape == std::vector<size_t>({2, 3, 1}));
    REQUIRE(out["grad(input)"].data == std::vector<float>{30222, 30242, 30262,  //
                                                          30422, 30442, 30462});
}
