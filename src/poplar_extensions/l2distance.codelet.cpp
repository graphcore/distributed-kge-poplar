// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <cmath>

#ifdef __IPU__
#include <ipu_vector_math>
#endif
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

#ifdef __IPU__

float l2dist(const float* a, const float* b, size_t size) {
    float2* a2 = (float2*)a;
    float2* b2 = (float2*)b;

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        float2 diff = a2[i] - b2[i];
        sum += diff * diff;
    }
    float res = sum[0] + sum[1];
    if (size % 2) {
        float diff = a[size - 1] - b[size - 1];
        res += diff * diff;
    }
    return ipu::sqrt(res);
}

float l2dist(const half* a, const half* b, size_t size) {
    half4* a4 = (half4*)a;
    half4* b4 = (half4*)b;

    half4 sum = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        half4 diff = a4[i] - b4[i];
        sum += diff * diff;
    }
    float res = float(sum[0]) + float(sum[1]) + float(sum[2]) + float(sum[3]);
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            float diff = float(a[i] - b[i]);
            res += diff * diff;
        }
    }
    return ipu::sqrt(res);
}

#else  // !__IPU__

template <typename T>
float l2dist(const T* a, const T* b, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = float(a[i] - b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

#endif  // __IPU__

template <typename T>
class L2DistanceSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<poplar::Vector<T, SPAN, 8>> a;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> b;
    poplar::Output<T> out;

    bool compute() {
        *out = T(l2dist(&a[0], &b[0], a.size()));
        return true;
    }
};
template class L2DistanceSingleVertex<float>;
template class L2DistanceSingleVertex<half>;

template <typename T>
float l2distgrad(const T& a, const T* b, const T* dist, const T* grad, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = float(a - b[i]);
        float dist_i = float(dist[i]);
        float val = dist_i == 0.0f ? 0.0f : float(grad[i]) * diff / dist_i;
        sum += val;
    }
    return sum;
}

template <typename T>
class L2DistanceGradSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<T> a;
    poplar::Input<poplar::Vector<T, SPAN, 8>> b;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> dist;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> gradOutput;
    poplar::Output<T> grad;

    bool compute() {
        *grad = l2distgrad(*a, &b[0], &dist[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L2DistanceGradSingleVertex<float>;
template class L2DistanceGradSingleVertex<half>;
