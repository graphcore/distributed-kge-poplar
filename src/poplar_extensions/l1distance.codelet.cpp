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

float l1dist(const float* a, const float* b, size_t size) {
    float2* a2 = (float2*)a;
    float2* b2 = (float2*)b;

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        sum += ipu::fabs(a2[i] - b2[i]);
    }
    float res = sum[0] + sum[1];
    if (size % 2) res += ipu::fabs(a[size - 1] - b[size - 1]);
    return res;
}

float l1dist(const half* a, const half* b, size_t size) {
    half4* a4 = (half4*)a;
    half4* b4 = (half4*)b;

    half4 sum = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        sum += ipu::fabs(a4[i] - b4[i]);
    }
    float res = float(sum[0]) + float(sum[1]) + float(sum[2]) + float(sum[3]);
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            res += ipu::fabs(float(a[i] - b[i]));
        }
    }
    return res;
}

#else  // !__IPU__

template <typename T>
float l1dist(const T* a, const T* b, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += std::fabs(float(a[i] - b[i]));
    }
    return sum;
}

#endif  // __IPU__

template <typename T>
class L1DistanceSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<poplar::Vector<T, SPAN, 8>> a;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> b;
    poplar::Output<T> out;

    bool compute() {
        *out = T(l1dist(&a[0], &b[0], a.size()));
        return true;
    }
};
template class L1DistanceSingleVertex<float>;
template class L1DistanceSingleVertex<half>;

template <typename T>
static inline T signum(T x) {
    return T((T(0.0) < x) - (x < T(0.0)));
}

#ifdef __IPU__

template <class T>
T copysign(T x, T signValue) {
    return x * signum(signValue);
}
float2 copysign(float2 x, float2 signValue) {
    return x * float2{signum(signValue[0]), signum(signValue[1])};
}
half4 copysign(half4 x, half4 signValue) {
    return x * half4{signum(signValue[0]), signum(signValue[1]), signum(signValue[2]),
                     signum(signValue[3])};
}

float l1distgrad(const float& a, const float* b, const float* grad, size_t size) {
    float2 sum = {0.0, 0.0};
    float2* grad2 = (float2*)grad;
    float2* b2 = (float2*)b;
    float2 a2 = {a, a};

    for (size_t i = 0; i < size / 2; ++i) {
        sum += copysign(grad2[i], a2 - b2[i]);
    }
    float res = sum[0] + sum[1];
    if (size % 2) {
        res += copysign(grad[size - 1], a - b[size - 1]);
    }
    return res;
}

float l1distgrad(const half& a, const half* b, const half* grad, size_t size) {
    half4 sum = {0.0, 0.0};
    half4* grad4 = (half4*)grad;
    half4* b4 = (half4*)b;
    half4 a4 = {a, a, a, a};

    for (size_t i = 0; i < size / 4; ++i) {
        sum += copysign(grad4[i], a4 - b4[i]);
    }
    float res = float(sum[0]) + float(sum[1]) + float(sum[2]) + float(sum[3]);
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            res += float(copysign(grad[i], a - b[i]));
        }
    }
    return res;
}

#else  // !__IPU__

template <typename T>
float l1distgrad(const T& a, const T* b, const T* grad, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += float(grad[i]) * float(signum(a - b[i]));
    }
    return sum;
}

#endif  // __IPU__

template <typename T>
class L1DistanceGradSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<T> a;
    poplar::Input<poplar::Vector<T, SPAN, 8>> b;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> gradOutput;
    poplar::Output<T> grad;

    bool compute() {
        *grad = l1distgrad(*a, &b[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L1DistanceGradSingleVertex<float>;
template class L1DistanceGradSingleVertex<half>;
