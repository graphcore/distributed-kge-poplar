// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef FRNN_HPP
#define FRNN_HPP

#include "fructose.hpp"

/**
 * Fructose-Neural-Networks (fr::nn), additional functions for implementing neural nets.
 */
namespace fr::nn {

// Ops
Tensor relu(const Tensor& tensor);
Tensor softmaxCrossEntropy(const Tensor& logits, const Tensor& labels);
Tensor dropout(const Tensor& a, float dropProbability);

// Optimisers
void sgd(const Tensor& tensor, const Tensor& learningRate);

struct AdamParams {
    float betaM;
    float betaV;
    float epsilon;
    float weightDecay;
};
Tensor adamStepSizeAutoIncrement(const Tensor& step,
                                 const Tensor& learningRate,
                                 const AdamParams& params);
void adam(const Tensor& tensor,
          const Tensor& momentum,
          const Tensor& variance,
          const Tensor& stepSize,
          const AdamParams& params);

}  // namespace fr::nn

#endif  // FRNN_HPP
