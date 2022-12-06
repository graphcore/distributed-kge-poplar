// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef FRUCTOSE_HPP
#define FRUCTOSE_HPP

#include <memory>
#include <optional>
#include <vector>

#include <poplar/Type.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>

#include <pag/pag.hpp>

/**
 * FR = Fructose
 *
 * A simple "sugar" interface to Poplar/PAG. Sugar means:
 *  - No need to pass `pag::Graph` or `pag::Tape` around, they're stashed in globals.
 *  - Lazy Tensor layout - variables get tile mapping upon first use.
 *  - Tensor operator overloads, friendly op definitions.
 */
namespace fr {

///////////////////////////////////////////////////////////////////////////////
// Graph construction

/**
 * A fr::Tensor wraps either a grad-able pag::Tensor, or metadata, such that
 * it supports 'lazy layout' by the first op that uses it.
 */
struct Tensor {
    using ID = unsigned;
    using Shape = std::vector<size_t>;
    static constexpr ID Invalid = 0u;
    struct Spec {
        Shape shape;
        poplar::Type dtype;
    };

    Tensor();
    static Tensor declare(const Spec&, bool requiresGrad, const std::string& name);
    static Tensor wrap(const pag::Tensor&);

    void set(const poplar::Tensor&) const;
    void hostAccess(bool read = true, bool write = true) const;
    void backward(const Tensor& rootGrad = {}) const;

    Tensor transpose() const;
    Tensor reshape(const Shape&) const;
    Tensor slice(size_t dim, poplar::Interval region) const;
    std::vector<Tensor> split(size_t dim, const std::vector<size_t>& sizes) const;

    ID id() const;
    const Spec& spec() const;
    const Shape& shape() const;
    size_t rank() const;
    size_t numElements() const;
    poplar::Type dtype() const;
    const std::string& name() const;
    bool valid() const;
    pag::Tensor pag() const;
    Tensor grad() const;
    Tensor astype(poplar::Type) const;
    Tensor operator[](const Tensor& index) const;

   private:
    explicit Tensor(ID id);
    ID m_id;
};

bool operator==(const Tensor::Spec&, const Tensor::Spec&);
bool operator!=(const Tensor::Spec&, const Tensor::Spec&);
std::ostream& operator<<(std::ostream&, const Tensor::Spec&);

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& tensor);
Tensor operator~(const Tensor& tensor);
Tensor operator<(const Tensor& lhs, const Tensor& rhs);

namespace ops {

// Sources
template <class T>
Tensor constant(T value, poplar::Type type = poplar::equivalent_device_type<T>().value);
template <class T>
Tensor constant(const std::vector<T>& value,
                const std::optional<Tensor::Shape>& shape = std::nullopt,
                poplar::Type type = poplar::equivalent_device_type<T>().value);
template <class T>
Tensor full(const Tensor::Shape& shape,
            const T value,
            poplar::Type type = poplar::equivalent_device_type<T>().value);
Tensor variable(const std::string& name,
                const Tensor::Spec& spec,
                std::optional<bool> requiresGrad = std::nullopt);
Tensor input(const std::string& handle, const Tensor::Spec& spec);
Tensor randomNormal(float mean,
                    float stdDev,
                    const Tensor::Shape& shape,
                    unsigned seed = 0,
                    poplar::Type type = poplar::FLOAT);

// Sinks
void output(const std::string& handle, const Tensor& tensor);
void print(const std::string& message, const Tensor& tensor);

// Transformations
Tensor abs(const Tensor& a);
Tensor max(const Tensor& a, const Tensor& b);
Tensor square(const Tensor& a);
Tensor pow(const Tensor& a, float exponent);
Tensor sqrt(const Tensor& a);
Tensor cbrt(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor l1distance(const Tensor& a, const Tensor& b);
Tensor l2distance(const Tensor& a, const Tensor& b);
Tensor gather(const Tensor& tensor, const Tensor& indices);
Tensor sum(const Tensor& tensor, const std::vector<size_t>& dims = {});
Tensor mean(const Tensor& tensor, const std::vector<size_t>& dims = {});
Tensor matMul(const Tensor& lhs, const Tensor& rhs);
Tensor logSoftmax(const Tensor& tensor);
Tensor sigmoid(const Tensor& tensor);
Tensor logSigmoid(const Tensor& tensor);
Tensor oneHot(const Tensor& tensor, size_t N, poplar::Type type);

Tensor startGrad(const Tensor& tensor);
Tensor concat(const std::vector<Tensor>& tensors, size_t dim);
// Only in the fwd pass
Tensor copyToLinearTensor(const Tensor& tensor,
                          std::optional<unsigned> minElementsPerTile = std::nullopt,
                          std::optional<unsigned> grainSize = std::nullopt);

// Collectives
Tensor allGather(const Tensor& tensor);
Tensor allToAll(const Tensor& tensor);

// Other
void forN(unsigned n, const std::function<void(const fr::Tensor&)>& body);

}  // namespace ops

/**
 * Reading from or writing to host memory.
 */
struct Stream {
    Stream() = default;
    Stream(const std::string& handle, const Tensor::Spec&, poplar::DataStreamType);

    std::string handle() const;
    Tensor::Spec spec() const;

    Tensor read() const;
    void write(const Tensor&);

   private:
    poplar::DataStream m_stream;
    std::vector<size_t> m_shape;
};

/**
 * Reading from and writing to remote buffers.
 */
struct Buffer {
    Buffer() = default;
    Buffer(const std::string& name, const Tensor::Spec&);

    Tensor read(const Tensor& indices) const;
    void write(const Tensor& data, const Tensor& indices);

    size_t totalBytes(const poplar::Target&) const;

   private:
    std::vector<size_t> rwShape(const Tensor& indices) const;
    poplar::RemoteBuffer m_buffer;
    std::vector<size_t> m_rowShape;
};

///////////////////////////////////////////////////////////////////////////////
// Helpers & utilities

namespace util {

template <class T>
struct Seq {
    const T& sequence;
};

template <class T>
Seq<T> seq(const T& sequence);

template <class T>
std::ostream& operator<<(std::ostream&, const Seq<T>&);

template <class T>
bool operator==(const Seq<T>&, const Seq<T>&);
template <class T>
bool operator!=(const Seq<T>&, const Seq<T>&);

void checkArgument(const Tensor&,
                   const std::string& message,
                   const Tensor::Shape& shape,
                   const std::vector<poplar::Type>& types = {});

template <class T>
std::vector<T> arange(T start, T end);

template <class T>
std::vector<T> arange(T end);

unsigned numElements(const fr::Tensor::Shape& shape);

template <class Collection, class Func>
auto mapVector(const Collection& items, Func&& func) -> std::vector<decltype(func(*items.begin()))>;

}  // namespace util

namespace mapping {

struct Method {
    virtual void apply(poplar::Graph&, const poplar::Tensor&) const = 0;
};

struct Linear : Method {
    void apply(poplar::Graph&, const poplar::Tensor&) const;
};

struct OneTile : Method {
    /**
     * Tile number, can be negative, counting back from the last tile.
     *
     * Note that the tile number will wrap around modulo target.getNumTiles().
     */
    int tile;
    OneTile();
    explicit OneTile(int tile);
    void apply(poplar::Graph&, const poplar::Tensor&) const;
};

/**
 * Copy a tile mapping from another tensor.
 */
struct Copy : Method {
    poplar::Tensor base;
    Copy(const poplar::Tensor&);
    void apply(poplar::Graph&, const poplar::Tensor&) const;
};

/**
 * For each tensor, set a mapping (if not already set).
 */
void setDefault(const Method& method, const std::vector<Tensor>& tensors);

}  // namespace mapping

///////////////////////////////////////////////////////////////////////////////
// State management

struct TensorPool;

/**
 * A frame adds a new frame to the stack (e.g. debugContext), and gives access to the global
 * `pag::Graph`, `pag::Tape` etc.
 *
 * Usage:
 *     Frame f("myOps::foo");
 *     foo(f.graph, f.tape, f.di);
 */
struct Frame {
    pag::Graph& graph;
    poplin::PlanningCache& matMulCache;
    pag::Tape& tape;
    std::unordered_map<std::string, Stream>& streams;
    poplar::DebugInfo di;

    explicit Frame(const std::string& name = "",
                   poplar::SourceLocation loc = poplar::SourceLocation::Current());
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    ~Frame();

    unsigned replicationFactor() const;

   protected:
    Frame(pag::Graph& graph,
          poplin::PlanningCache& matMulCache,
          pag::Tape& tape,
          std::unordered_map<std::string, Stream>& streams,
          const poplar::DebugInfo& di,
          const std::string& name,
          const poplar::SourceLocation& loc);
};

struct SubProgramFrame : Frame {
    explicit SubProgramFrame(const std::string& name = "",
                             poplar::SourceLocation loc = poplar::SourceLocation::Current());

   private:
    pag::Tape m_tape;
    std::unordered_map<std::string, Stream> m_streams;
};

struct RootFrame : Frame {
    std::unordered_map<std::string, Tensor> variables;
    std::unique_ptr<TensorPool> pool;

    explicit RootFrame(const poplar::Target&,
                       poplar::SourceLocation loc = poplar::SourceLocation::Current());
    ~RootFrame();

   private:
    poplar::Graph m_poplarGraph;
    pag::Graph m_pagGraph;
    poplin::PlanningCache m_matMulCache;
    pag::Tape m_tape;
    std::unordered_map<std::string, Stream> m_streams;
};

/**
 * Global state that defines the current `poplar::Graph` and `poplar::program::Sequence`
 * being built.
 */
struct Environment {
    static Frame& frame();
    static RootFrame& rootFrame();

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

   private:
    friend struct Frame;
    friend struct RootFrame;
    Environment();
    static Environment& instance();
    RootFrame* m_root;
    std::vector<Frame*> m_stack;
};

}  // namespace fr

///////////////////////////////////////////////////////////////////////////////
// Template implementations

namespace fr {

namespace ops {

template <class T>
Tensor constant(T value, poplar::Type type) {
    Frame f("fr::ops::constant");
    auto poplarTensor = f.graph.poplar().addConstant(type, {}, value, f.di);
    // Most ops fill from zero, so constants on N-1 seems somewhat reasonable.
    mapping::OneTile().apply(f.graph.poplar(), poplarTensor);
    return Tensor::wrap(f.graph.wrap(poplarTensor, /*requiresGrad*/ false));
}

template <class T>
Tensor constant(const std::vector<T>& value,
                const std::optional<Tensor::Shape>& shape,
                poplar::Type type) {
    Frame f("fr::ops::constant");
    auto actualShape = shape.value_or(Tensor::Shape{value.size()});
    auto poplarTensor = f.graph.poplar().addConstant<T>(type, actualShape, value);
    mapping::Linear().apply(f.graph.poplar(), poplarTensor);
    return Tensor::wrap(f.graph.wrap(poplarTensor, /*requiresGrad*/ false));
}

template <class T>
Tensor full(const Tensor::Shape& shape, const T value, poplar::Type type) {
    Frame f("fr::ops::full");
    auto poplarTensor = f.graph.poplar().addConstant(type, shape, value, f.di);
    mapping::Linear().apply(f.graph.poplar(), poplarTensor);
    return Tensor::wrap(f.graph.wrap(poplarTensor, /*requiresGrad*/ false));
}

}  // namespace ops

namespace util {

template <class T>
Seq<T> seq(const T& sequence) {
    return Seq<T>{sequence};
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Seq<T>& seq) {
    bool separator = false;
    out << "{";
    for (auto& element : seq.sequence) {
        if (separator) out << ", ";
        out << element;
        separator = true;
    }
    return out << "}";
}

template <class T>
bool operator==(const Seq<T>& lhs, const Seq<T>& rhs) {
    return std::equal(lhs.sequence.begin(), lhs.sequence.end(), rhs.sequence.begin(),
                      rhs.sequence.end());
}
template <class T>
bool operator!=(const Seq<T>& lhs, const Seq<T>& rhs) {
    return !(lhs == rhs);
}

template <class T>
std::vector<T> arange(T start, T end) {
    std::vector<T> result(static_cast<size_t>(end - start));
    std::iota(result.begin(), result.end(), start);
    return result;
}

template <class T>
std::vector<T> arange(T end) {
    return arange(T(0), end);
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

}  // namespace util

}  // namespace fr

#endif  // FRUCTOSE_HPP
