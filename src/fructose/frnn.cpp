// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "frnn.hpp"

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace fr::nn {

///////////////////////////////////////////////////////////////////////////////
// Ops

Tensor relu(const Tensor& tensor) {
    Frame f("frnn::relu");
    return ops::max(tensor, ops::constant(0.0f, tensor.dtype()));
}

Tensor softmaxCrossEntropy(const Tensor& logits, const Tensor& labels) {
    Frame f("frnn::softmaxCrossEntropy");
    auto logp = ops::logSoftmax(logits);
    auto oneHotLabels = ops::oneHot(labels, logp.shape().back(), logp.dtype());
    return -ops::sum(logp * oneHotLabels, {logp.rank() - 1});
}

Tensor dropout(const Tensor& a, float dropProbability) {
    Frame f("frnn::dropout");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::dropout(f.graph, a.pag(), dropProbability, f.tape, f.di));
}

///////////////////////////////////////////////////////////////////////////////
// Optimisers

void sgd(const Tensor& tensor, const Tensor& learningRate) {
    Frame f("frnn::sgd");
    auto grad = tensor.grad();
    popops::scaledSubtractFrom(f.graph.poplar(), f.graph.unwrap(tensor.pag()),
                               f.graph.unwrap(grad.pag()), f.graph.unwrap(learningRate.pag()),
                               f.tape.prog(), f.di);
}

Tensor adamStepSizeAutoIncrement(const Tensor& step,
                                 const Tensor& learningRate,
                                 const AdamParams& params) {
    Frame f("frnn::adamStepSizeAutoIncrement");
    mapping::setDefault(mapping::OneTile(), {step, learningRate});
    popops::addInPlace(f.graph.poplar(), f.graph.unwrap(step.pag()),
                       f.graph.unwrap(ops::constant(1u, step.dtype()).pag()), f.tape.prog(), f.di);

    namespace pe = popops::expr;
    auto numerator =
        pe::Sqrt(1 - pe::Pow(pe::Const(params.betaV), pe::Cast(pe::_1, poplar::FLOAT)));
    auto denominator = 1 - pe::Pow(pe::Const(params.betaM), pe::Cast(pe::_1, poplar::FLOAT));
    return Tensor::wrap(
        f.graph.wrap(popops::map(f.graph.poplar(), (numerator / denominator) * pe::_2,
                                 {f.graph.unwrap(step.pag()), f.graph.unwrap(learningRate.pag())},
                                 f.tape.prog(), f.di),
                     /*requiresGrad*/ false));
}

void adam(const Tensor& tensor,
          const Tensor& momentum,
          const Tensor& variance,
          const Tensor& stepSize,
          const AdamParams& params) {
    Frame f("frnn::adam");
    assert(tensor.valid() && "`tensor` should have a grad, so must have been mapped");

    // {tensor, momentum, variance} are used elementwise, so give them the same default mapping
    mapping::setDefault(mapping::Copy(f.graph.unwrap(tensor.pag())), {momentum, variance});

    namespace pe = popops::expr;

    auto grad = tensor.grad();
    popops::mapInPlace(
        f.graph.poplar(), pe::Const(params.betaM) * pe::_1 + pe::Const(1 - params.betaM) * pe::_2,
        {f.graph.unwrap(momentum.pag()), f.graph.unwrap(grad.pag())}, f.tape.prog(), f.di);

    popops::mapInPlace(
        f.graph.poplar(),
        pe::Const(params.betaV) * pe::_1 + pe::Const(1 - params.betaV) * pe::_2 * pe::_2,
        {f.graph.unwrap(variance.pag()), f.graph.unwrap(grad.pag())}, f.tape.prog(), f.di);

    auto update = pe::_1 - pe::_2 * (pe::_3 / (pe::Sqrt(pe::_4) + pe::Const(params.epsilon)) +
                                     pe::Const(params.weightDecay) * pe::_1);
    popops::mapInPlace(f.graph.poplar(), update,
                       {f.graph.unwrap(tensor.pag()), f.graph.unwrap(stepSize.pag()),
                        f.graph.unwrap(momentum.pag()), f.graph.unwrap(variance.pag())},
                       f.tape.prog(), f.di);
}

}  // namespace fr::nn
