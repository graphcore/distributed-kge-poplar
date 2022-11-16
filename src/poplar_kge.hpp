// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPLAR_KGE_HPP
#define POPLAR_KGE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "fructose/fructose.hpp"

namespace poplar_kge {

fr::Tensor detachedSoftmax(const fr::Tensor& tensor);

/**
 * A C-contiguous N-dimensional array view (does not own 'data').
 */
template <class T>
struct ArrayView {
    ArrayView(const std::vector<size_t>& shape, T* data);

    const std::vector<size_t>& shape() const;
    const T* data() const;
    T* data();

   private:
    std::vector<size_t> m_shape;
    T* m_data;
};

struct float16 {
    uint16_t value;
};
static_assert(sizeof(float16) == 2);

using Batch =
    std::unordered_map<std::string,
                       std::variant<std::string,
                                    // Important for ArrayView to come before scalar
                                    // types to avoid implicit conversions to scalars
                                    ArrayView<float>,
                                    ArrayView<uint32_t>,
                                    ArrayView<float16>,
                                    bool,
                                    float,
                                    unsigned,
                                    std::vector<std::tuple<std::string, std::vector<unsigned>>>,
                                    std::unordered_map<std::string, float>>>;

struct EngineImpl;

/**
 * Interface to training engine.
 */
struct Engine {
    /**
     * Construct an engine to train & evaluate the KGE model.
     *
     * See lib.cpp:engineDocstring for full details.
     */
    Engine(const Batch& settings, const std::string& gpFolder);
    ~Engine();

    /**
     * Run a command, e.g. "train_step_loop", "read", etc.
     *
     * See lib.cpp:runDocstring for full details.
     */
    Batch run(const std::string& command, Batch& data);

   private:
    std::unique_ptr<EngineImpl> m_impl;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation

template <class T>
ArrayView<T>::ArrayView(const std::vector<size_t>& shape, T* data) : m_shape(shape), m_data(data) {}
template <class T>
const std::vector<size_t>& ArrayView<T>::shape() const {
    return m_shape;
}
template <class T>
const T* ArrayView<T>::data() const {
    return m_data;
}
template <class T>
T* ArrayView<T>::data() {
    return m_data;
}

}  // namespace poplar_kge

#endif  // POPLAR_KGE_HPP
