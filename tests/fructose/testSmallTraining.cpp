// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <catch2/catch.hpp>

#include <poplar/Engine.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>

#include "fructose/frnn.hpp"
#include "fructose/fructose.hpp"

TEST_CASE("Small training example", "[fr]") {
    auto N = 10u;
    auto hidden_size = 64u;
    std::mt19937 rng(3248924);

    // Dataset - modulo addition
    std::vector<uint32_t> dataA, dataB, dataY;
    for (auto i = 0u; i < N * N; ++i) {
        auto a = i % N;
        auto b = i / N;
        auto y = (a + b) % N;
        dataA.push_back(a);
        dataB.push_back(b);
        dataY.push_back(y);
    }

    // Build graph
    auto device = poplar::Device::createCPUDevice();
    fr::RootFrame rootFrame(device.getTarget());
    popops::addCodelets(rootFrame.graph.poplar());
    poplin::addCodelets(rootFrame.graph.poplar());

    auto a = fr::ops::input("a", {{N * N}, poplar::UNSIGNED_INT});
    auto b = fr::ops::input("b", {{N * N}, poplar::UNSIGNED_INT});
    auto y = fr::ops::input("y", {{N * N}, poplar::UNSIGNED_INT});

    auto embedding = fr::ops::variable("embedding", {{N, hidden_size}, poplar::FLOAT});
    auto z = fr::nn::relu(fr::ops::gather(embedding, a) + fr::ops::gather(embedding, b));

    auto hiddenW = fr::ops::variable("hidden/W", {{hidden_size, hidden_size}, poplar::FLOAT});
    auto hiddenB = fr::ops::variable("hidden/b", {{hidden_size}, poplar::FLOAT});
    z = fr::nn::relu(fr::ops::matMul(z, hiddenW) + hiddenB);

    auto projectionW = fr::ops::variable("projection/W", {{hidden_size, N}, poplar::FLOAT});
    auto logits = fr::ops::matMul(z, projectionW);
    auto loss = fr::ops::mean(fr::nn::softmaxCrossEntropy(logits, y));

    loss.backward();
    fr::ops::output("loss", loss);

    fr::nn::AdamParams adamParams{/*betaM*/ 0.9f, /*betaV*/ 0.999f, /*epsilon*/ 1e-8f,
                                  /*weightDecay*/ 0.0f};
    auto step = fr::ops::variable("step", {{}, poplar::UNSIGNED_INT});
    auto adamStepSize =
        fr::nn::adamStepSizeAutoIncrement(step, fr::ops::constant(0.01f), adamParams);
    for (auto& tensor : {embedding, hiddenW, hiddenB, projectionW}) {
        auto momentum =
            fr::ops::variable(tensor.name() + "/adam_m", tensor.spec(), /*requiresGrad*/ false);
        auto variance =
            fr::ops::variable(tensor.name() + "/adam_v", tensor.spec(), /*requiresGrad*/ false);
        fr::nn::adam(tensor, momentum, variance, adamStepSize, adamParams);
        tensor.hostAccess();
        momentum.hostAccess();
        variance.hostAccess();
    }
    step.hostAccess();

    // Initialise
    poplar::Engine engine(rootFrame.graph.poplar(), rootFrame.tape.prog());
    engine.load(device);
    engine.writeTensor<uint32_t>("step", {0u});
    for (auto& tensor : {embedding, hiddenW, hiddenB, projectionW}) {
        std::vector<float> init(tensor.numElements());
        auto scale = std::unordered_map<std::string, float>{
            {"embedding", 1.0f},
            {"hidden/W", 1.0f / std::sqrt(hidden_size)},
            {"hidden/b", 0.0f},
            {"projection/W", 1.0f / std::sqrt(hidden_size)}}[tensor.name()];
        std::generate(init.begin(), init.end(),
                      [&rng, scale] { return scale * std::normal_distribution<float>()(rng); });
        engine.writeTensor<float>(tensor.name(), init);
        engine.writeTensor<float>(tensor.name() + "/adam_m", std::vector<float>(init.size()));
        engine.writeTensor<float>(tensor.name() + "/adam_v", std::vector<float>(init.size()));
    }

    // Run
    float hostLoss;
    engine.connectStream("loss", &hostLoss);
    engine.connectStream<uint32_t>("a", dataA);
    engine.connectStream<uint32_t>("b", dataB);
    engine.connectStream<uint32_t>("y", dataY);
    for (auto i = 0u; i < 30; ++i) {
        engine.run();
    }
    REQUIRE(hostLoss < 0.03f);
}
