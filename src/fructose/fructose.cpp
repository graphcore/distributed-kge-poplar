// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "fructose.hpp"

#include <cmath>
#include <iostream>
#include <sstream>

#include <poplin/MatMul.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Loop.hpp>
#include <poputil/TileMapping.hpp>

namespace fr {

///////////////////////////////////////////////////////////////////////////////
// State management

/**
 * A TensorPool is a grow-only stash for fr::Tensor internal data.
 *
 * Storing data here allows fr::Tensor to be copyable (as they are slim 'views'),
 * much like poplar::Tensor. Unlike poplar::Graph, the TensorPool is stored in a
 * global context, so it doesn't need to be passed around.
 */
struct TensorPool {
    struct TensorData {
        struct Metadata : Tensor::Spec {
            bool requiresGrad;
            std::string name;
        };
        Metadata metadata;
        mutable pag::Tensor tensor;
    };

    TensorPool() {
        assert(Tensor::Invalid == 0u);
        m_tensors.push_back({{{{}, poplar::CHAR}, false, "INVALID"}, {}});
    }

    Tensor::ID add(const TensorData& data) {
        m_tensors.push_back(data);
        return m_tensors.size() - 1;
    }

    TensorData& operator[](Tensor::ID id) {
        if (id == Tensor::Invalid) {
            throw std::runtime_error(
                "Trying to lookup data for an invalid (default-constructed) tensor");
        }
        return m_tensors.at(id);
    }

   private:
    std::vector<TensorData> m_tensors;
};

Frame::Frame(const std::string& name, poplar::SourceLocation loc)
    : Frame(Environment::frame().graph,
            Environment::frame().matMulCache,
            Environment::frame().tape,
            Environment::frame().streams,
            Environment::frame().di,
            name,
            loc) {}
Frame::~Frame() {
    assert(Environment::instance().m_stack.back() == this);
    Environment::instance().m_stack.pop_back();
}
Frame::Frame(pag::Graph& graph,
             poplin::PlanningCache& matMulCache,
             pag::Tape& tape,
             std::unordered_map<std::string, Stream>& streams,
             const poplar::DebugInfo& di,
             const std::string& name,
             const poplar::SourceLocation& loc)
    : graph(graph),
      matMulCache(matMulCache),
      tape(tape),
      streams(streams),
      di({di, name, loc}, "fructose") {
    Environment::instance().m_stack.push_back(this);
}
unsigned Frame::replicationFactor() const {
    return graph.poplar().getReplicationFactor();
}

SubProgramFrame::SubProgramFrame(const std::string& name, poplar::SourceLocation loc)
    : Frame(Environment::frame().graph,
            Environment::frame().matMulCache,
            m_tape,
            m_streams,
            Environment::frame().di,
            name,
            loc) {}

RootFrame::RootFrame(const poplar::Target& target, poplar::SourceLocation loc)
    : Frame(m_pagGraph,
            m_matMulCache,
            m_tape,
            m_streams,
            poplar::DebugInfo("", "fructose"),
            "",
            loc),
      pool(std::make_unique<TensorPool>()),
      m_poplarGraph(target, poplar::replication_factor(target.getNumIPUs())),
      m_pagGraph(m_poplarGraph) {
    assert(Environment::instance().m_root == nullptr);
    Environment::instance().m_root = this;
}
RootFrame::~RootFrame() {
    assert(Environment::instance().m_stack.size() == 1);
    assert(Environment::instance().m_root == this);
    Environment::instance().m_root = nullptr;
}

Environment::Environment() : m_root(nullptr) {}
Environment& Environment::instance() {
    static Environment instance;
    return instance;
}

Frame& Environment::frame() {
    return *instance().m_stack.back();
}
RootFrame& Environment::rootFrame() {
    assert(instance().m_root);
    return *instance().m_root;
}

///////////////////////////////////////////////////////////////////////////////
// Graph construction

namespace {
TensorPool& getPool() {
    return *Environment::rootFrame().pool;
}
}  // namespace

Tensor::Tensor() : m_id(Invalid) {}
Tensor::Tensor(ID id) : m_id(id) {}
Tensor Tensor::declare(const Spec& spec, bool requiresGrad, const std::string& name) {
    return Tensor(getPool().add({{spec, requiresGrad, name}, {}}));
}
Tensor Tensor::wrap(const pag::Tensor& pagTensor) {
    auto& f = Environment::frame();
    auto poplarTensor = f.graph.unwrap(pagTensor);
    return Tensor(getPool().add({{{poplarTensor.shape(), poplarTensor.elementType()},
                                  f.graph.requiresGrad(pagTensor),
                                  poplarTensor.getDebugStr()},
                                 pagTensor}));
}

void Tensor::set(const poplar::Tensor& tensor) const {
    auto& data = getPool()[m_id];
    if (data.tensor.valid()) {
        std::ostringstream msg;
        msg << "Trying to set tensor '" << data.metadata.name << "' that has already been set";
        throw std::logic_error(msg.str());
    }
    if (!tensor.valid()) {
        std::ostringstream msg;
        msg << "Trying to set tensor '" << data.metadata.name
            << "' with a tensor that is not valid()";
        throw std::invalid_argument(msg.str());
    }
    if (util::seq(data.metadata.shape) != util::seq(tensor.shape())) {
        std::ostringstream msg;
        msg << "Setting a tensor with incorrect shape, expected: " << util::seq(data.metadata.shape)
            << ", actual: " << util::seq(tensor.shape());
        throw std::invalid_argument(msg.str());
    }
    data.tensor = Environment::frame().graph.wrap(tensor, data.metadata.requiresGrad);
}
void Tensor::hostAccess(bool read, bool write) const {
    auto& f = Environment::frame();
    auto tensor = f.graph.unwrap(pag());
    if (read) {
        f.graph.poplar().createHostRead(name(), tensor);
    }
    if (write) {
        f.graph.poplar().createHostWrite(name(), tensor);
    }
}
void Tensor::backward(const Tensor& rootGrad) const {
    auto& f = Environment::frame();
    auto poplarRootGrad = rootGrad.valid() ? f.graph.unwrap(rootGrad.pag()) : poplar::Tensor();
    f.tape.backward(f.graph, pag(), poplarRootGrad);
}

Tensor Tensor::transpose() const {
    Frame f("fr::Tensor::transpose");
    mapping::setDefault(mapping::Linear(), {*this});
    return wrap(pag::ops::transpose(f.graph, pag(), f.tape));
}
Tensor Tensor::reshape(const Shape& shape) const {
    Frame f("fr::Tensor::reshape");
    mapping::setDefault(mapping::Linear(), {*this});
    return wrap(pag::ops::reshape(f.graph, pag(), shape, f.tape));
}
Tensor Tensor::slice(size_t dim, poplar::Interval region) const {
    Frame f("fr::Tensor::slice");
    mapping::setDefault(mapping::Linear(), {*this});
    return wrap(pag::ops::slice(f.graph, pag(), dim, region, f.tape));
}
std::vector<Tensor> Tensor::split(size_t dim, const std::vector<size_t>& sizes) const {
    Frame f("fr::Tensor::split");
    mapping::setDefault(mapping::Linear(), {*this});
    return util::mapVector(pag::ops::split(f.graph, pag(), dim, sizes, f.tape),
                           [](auto& t) { return wrap(t); });
}

Tensor::ID Tensor::id() const {
    return m_id;
}
const Tensor::Spec& Tensor::spec() const {
    return getPool()[m_id].metadata;
}
const Tensor::Shape& Tensor::shape() const {
    return getPool()[m_id].metadata.shape;
}
size_t Tensor::rank() const {
    return shape().size();
}
size_t Tensor::numElements() const {
    return util::numElements(shape());
}
poplar::Type Tensor::dtype() const {
    return getPool()[m_id].metadata.dtype;
}
const std::string& Tensor::name() const {
    return getPool()[m_id].metadata.name;
}
bool Tensor::valid() const {
    return (m_id != Invalid) && getPool()[m_id].tensor.valid();
}
pag::Tensor Tensor::pag() const {
    if (!valid()) {
        std::ostringstream msg;
        msg << "Trying to get pag::Tensor from fr::Tensor '" << name()
            << "' before it is valid - expected set() to have been called";
        throw std::logic_error(msg.str());
    }
    return getPool()[m_id].tensor;
}
Tensor Tensor::grad() const {
    auto& f = Environment::frame();
    return wrap(f.graph.wrap(f.graph.grad(pag()), /*requiresGrad*/ false));
}

Tensor Tensor::astype(poplar::Type type) const {
    Frame f("fr::Tensor::astype");
    mapping::setDefault(mapping::Linear(), {*this});
    return wrap(pag::ops::cast(f.graph, pag(), type, f.tape, f.di));
}

Tensor Tensor::operator[](const Tensor& index) const {
    Frame f("fr::Tensor::operator[]");
    if (index.shape().size() != 0) {
        std::ostringstream msg;
        msg << "Calling Tensor::operator[] with index of shape " << fr::util::seq(index.shape())
            << ", but only scalar indexing is implemented";
        throw std::invalid_argument(msg.str());
    }
    if (f.graph.requiresGrad(pag())) {
        std::ostringstream msg;
        msg << "Tensor::operator[] called on tensor '" << name()
            << "', which requires gradients, but gradients are not implemented";
        throw std::invalid_argument(msg.str());
    }
    if (!valid()) {
        set(popops::createSliceableTensor(f.graph.poplar(), dtype(), shape(), {0}, {1}, 0ull,
                                          name()));
    }
    mapping::setDefault(mapping::Linear(), {index});

    auto poplarTensor = popops::dynamicSlice(f.graph.poplar(), f.graph.unwrap(pag()),
                                             f.graph.unwrap(index.pag()).reshape({1}), {0}, {1},
                                             f.tape.prog(), f.di);
    return wrap(f.graph.wrap(poplarTensor.squeeze({0}), /*requiresGrad*/ false));
}

bool operator==(const Tensor::Spec& lhs, const Tensor::Spec& rhs) {
    return util::seq(lhs.shape) == util::seq(rhs.shape) && lhs.dtype == rhs.dtype;
}
bool operator!=(const Tensor::Spec& lhs, const Tensor::Spec& rhs) {
    return !(lhs == rhs);
}
std::ostream& operator<<(std::ostream& out, const Tensor::Spec& spec) {
    return out << spec.dtype << util::seq(spec.shape);
}

///////////////////////////////////////////////////////////////////////////////
// Utility

namespace util {

namespace {
void printExpectedShape(std::ostream& out, const Tensor::Shape& shape) {
    bool separator = false;
    out << "{";
    for (auto& element : shape) {
        if (separator) out << ", ";
        if (element == 0) {
            out << "*";
        } else {
            out << element;
        }
        separator = true;
    }
    out << "}";
}
}  // namespace

void checkArgument(const Tensor& tensor,
                   const std::string& message,
                   const Tensor::Shape& shape,
                   const std::vector<poplar::Type>& types) {
    auto tensorShape = tensor.spec().shape;
    std::string error;
    if (tensorShape.size() != shape.size()) {
        error = "rank";
    } else {
        for (auto i = 0u; i < shape.size(); ++i) {
            if (shape[i] != 0 && shape[i] != tensorShape[i]) {
                error = "shape";
            }
        }
    }
    if (!error.empty()) {
        std::ostringstream msg;
        msg << "Bad " << error << " of " << message << ", expected shape: ";
        printExpectedShape(msg, shape);
        msg << ", actual shape: " << util::seq(tensorShape);
        throw std::invalid_argument(msg.str());
    }

    auto tensorDtype = tensor.spec().dtype;
    if (!types.empty() && std::find(types.begin(), types.end(), tensorDtype) == types.end()) {
        std::ostringstream msg;
        msg << "Bad type of " << message << ", expected: " << seq(types)
            << ", actual: " << tensorDtype;
        throw std::invalid_argument(msg.str());
    }
}

unsigned numElements(const fr::Tensor::Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<size_t>());
}

}  // namespace util

///////////////////////////////////////////////////////////////////////////////
// Mapping

namespace mapping {

void Linear::apply(poplar::Graph& graph, const poplar::Tensor& tensor) const {
    poputil::mapTensorLinearly(graph, tensor);
}

OneTile::OneTile() : tile(-1) {}
OneTile::OneTile(int tile) : tile(tile) {}
void OneTile::apply(poplar::Graph& graph, const poplar::Tensor& tensor) const {
    // Note: <int> % <unsigned> doesn't behave as expected
    auto numTiles = static_cast<int>(graph.getTarget().getNumTiles());
    auto offset = tile % numTiles;
    graph.setTileMapping(tensor, offset + (offset < 0) * numTiles);
}

Copy::Copy(const poplar::Tensor& base) : base(base) {}
void Copy::apply(poplar::Graph& graph, const poplar::Tensor& tensor) const {
    graph.setTileMapping(tensor, graph.getTileMapping(base));
}

void setDefault(const Method& method, const std::vector<Tensor>& tensors) {
    auto& poplarGraph = Environment::frame().graph.poplar();
    for (auto& tensor : tensors) {
        if (!tensor.valid()) {
            auto poplarTensor =
                poplarGraph.addVariable(tensor.spec().dtype, tensor.spec().shape, tensor.name());
            method.apply(poplarGraph, poplarTensor);
            tensor.set(poplarTensor);
        }
    }
}

}  // namespace mapping

///////////////////////////////////////////////////////////////////////////////
// Ops library

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    Frame f("operator+");
    mapping::setDefault(mapping::Linear(), {lhs, rhs});
    return Tensor::wrap(pag::ops::add(f.graph, lhs.pag(), rhs.pag(), f.tape, f.di));
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    Frame f("operator-");
    mapping::setDefault(mapping::Linear(), {lhs, rhs});
    return Tensor::wrap(pag::ops::sub(f.graph, lhs.pag(), rhs.pag(), f.tape, f.di));
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    Frame f("operator*");
    mapping::setDefault(mapping::Linear(), {lhs, rhs});
    return Tensor::wrap(pag::ops::mul(f.graph, lhs.pag(), rhs.pag(), f.tape, f.di));
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    Frame f("operator/");
    mapping::setDefault(mapping::Linear(), {lhs, rhs});
    return Tensor::wrap(pag::ops::div(f.graph, lhs.pag(), rhs.pag(), f.tape, f.di));
}

Tensor operator-(const Tensor& tensor) {
    Frame f("operator-");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::neg(f.graph, tensor.pag(), f.tape, f.di));
}

Tensor operator~(const Tensor& tensor) {
    Frame f("operator~");
    // No gradients, so we use poplibs directly
    mapping::setDefault(mapping::Linear(), {tensor});
    auto poplarOutput =
        popops::logicalNot(f.graph.poplar(), f.graph.unwrap(tensor.pag()), f.tape.prog(), f.di);
    return Tensor::wrap(f.graph.wrap(poplarOutput, /*requiresGrad*/ false));
}

Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
    Frame f("operator<");
    mapping::setDefault(mapping::Linear(), {lhs, rhs});
    // No gradients, so we use poplibs directly
    auto poplarOutput = popops::lt(f.graph.poplar(), f.graph.unwrap(lhs.pag()),
                                   f.graph.unwrap(rhs.pag()), f.tape.prog(), f.di);
    return Tensor::wrap(f.graph.wrap(poplarOutput, /*requiresGrad*/ false));
}

namespace ops {

Tensor variable(const std::string& name,
                const Tensor::Spec& spec,
                std::optional<bool> requiresGrad) {
    auto& variables = Environment::rootFrame().variables;
    if (variables.find(name) != variables.end()) {
        std::ostringstream msg;
        msg << "Variable '" << name << "' already exists, existing spec: " << variables[name].spec()
            << ", new spec: " << spec;
        throw std::invalid_argument(msg.str());
    }
    auto tensor = Tensor::declare(spec, requiresGrad.value_or(spec.dtype.isFloatingPoint()), name);
    variables[name] = tensor;
    return tensor;
}

Tensor randomNormal(float mean,
                    float stdDev,
                    const Tensor::Shape& shape,
                    unsigned seed,
                    poplar::Type type) {
    Frame f("fr::ops::randomNormal");
    auto referenceTensor = f.graph.poplar().addVariable(type, shape, f.di);
    auto replicationIndex = f.graph.poplar().addReplicationIndexConstant(f.di).expand({0});
    auto inputSeed = f.graph.poplar().addConstant(poplar::UNSIGNED_INT, {1}, seed, f.di);
    auto seedTensor = poplar::concat(replicationIndex, inputSeed);
    poputil::mapTensorLinearly(f.graph.poplar(), referenceTensor);
    poputil::mapTensorLinearly(f.graph.poplar(), replicationIndex);
    poputil::mapTensorLinearly(f.graph.poplar(), inputSeed);
    referenceTensor = poprand::normal(f.graph.poplar(), &seedTensor, 0, referenceTensor, type, mean,
                                      stdDev, f.tape.prog(), f.di);
    return Tensor::wrap(f.graph.wrap(referenceTensor, /*requiresgrad*/ false));
}

namespace {
Stream& getOrCreateStream(const std::string& handle,
                          const Tensor::Spec& spec,
                          poplar::DataStreamType type) {
    auto& streams = Environment::frame().streams;
    auto it = streams.find(handle);
    if (it != streams.end()) {
        if (spec != it->second.spec()) {
            std::ostringstream msg;
            msg << "Existing stream for '" << handle << "' doesn't match tensor spec " << spec;
            throw std::invalid_argument(msg.str());
        }
    } else {
        it = streams.insert(std::make_pair(handle, Stream(handle, spec, type))).first;
    }
    return it->second;
}
}  // namespace

Tensor input(const std::string& handle, const Tensor::Spec& spec) {
    Frame f("fr::ops::input");
    return getOrCreateStream(handle, spec, poplar::DataStreamType::HostToDeviceFIFO).read();
}

void output(const std::string& handle, const Tensor& tensor) {
    Frame f("fr::ops::output");
    getOrCreateStream(handle, tensor.spec(), poplar::DataStreamType::DeviceToHostFIFO)
        .write(tensor);
}

void print(const std::string& message, const Tensor& tensor) {
    Frame f("fr::ops::print");
    f.tape.prog().add(poplar::program::PrintTensor(message, f.graph.unwrap(tensor.pag()), f.di));
}

Tensor abs(const Tensor& a) {
    Frame f("fr::ops::abs");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::abs(f.graph, a.pag(), f.tape, f.di));
}

Tensor max(const Tensor& a, const Tensor& b) {
    Frame f("fr::ops::max");
    mapping::setDefault(mapping::Linear(), {a, b});
    return Tensor::wrap(pag::ops::max(f.graph, a.pag(), b.pag(), f.tape, f.di));
}

Tensor square(const Tensor& a) {
    Frame f("fr::ops::square");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::square(f.graph, a.pag(), f.tape, f.di));
}

Tensor pow(const Tensor& a, float exponent, bool safeGradZero) {
    Frame f("fr::ops::pow");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::pow(f.graph, a.pag(), exponent, safeGradZero, f.tape, f.di));
}

Tensor sqrt(const Tensor& a) {
    Frame f("fr::ops::sqrt");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::sqrt(f.graph, a.pag(), f.tape, f.di));
}

Tensor cbrt(const Tensor& a) {
    Frame f("fr::ops::cbrt");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::cbrt(f.graph, a.pag(), f.tape, f.di));
}

Tensor sin(const Tensor& a) {
    Frame f("fr::ops::sin");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::sin(f.graph, a.pag(), f.tape, f.di));
}

Tensor cos(const Tensor& a) {
    Frame f("fr::ops::cos");
    mapping::setDefault(mapping::Linear(), {a});
    return Tensor::wrap(pag::ops::cos(f.graph, a.pag(), f.tape, f.di));
}

Tensor l1distance(const Tensor& a, const Tensor& b) {
    Frame f("fr::ops::l1distance");
    mapping::setDefault(mapping::Linear(), {a, b});
    return Tensor::wrap(pag::ops::l1distance(f.graph, a.pag(), b.pag(), f.tape, f.di));
}

Tensor l2distance(const Tensor& a, const Tensor& b) {
    Frame f("fr::ops::l2distance");
    mapping::setDefault(mapping::Linear(), {a, b});
    return Tensor::wrap(pag::ops::l2distance(f.graph, a.pag(), b.pag(), f.tape, f.di));
}

Tensor gather(const Tensor& tensor, const Tensor& indices) {
    Frame f("fr::ops::gather");
    util::checkArgument(tensor, "gather 'tensor'", /*shape*/ {0u, 0u});
    util::checkArgument(indices, "gather 'indices'", /*shape*/ {0u}, {poplar::UNSIGNED_INT});

    auto numIndices = indices.shape()[0];
    poplar::OptionFlags options;
    auto plan = popops::embedding::plan(f.graph.poplar(), tensor.dtype(), tensor.shape()[0],
                                        tensor.shape()[1], {numIndices}, options);

    if (!tensor.valid()) {
        tensor.set(popops::createSliceableTensor(f.graph.poplar(), tensor.dtype(), tensor.shape(),
                                                 {0}, {1}, plan, options));
    }
    if (!indices.valid()) {
        auto rawIndices =
            popops::createIndicesTensor(f.graph.poplar(), {0}, numIndices, plan, options);
        indices.set(rawIndices.squeeze({1}));
    }

    auto out = pag::ops::multiSlice(
        f.graph, tensor.pag(), pag::ops::reshape(f.graph, indices.pag(), {numIndices, 1}, f.tape),
        {0}, {1}, f.tape, plan, options, f.di);

    return Tensor::wrap(pag::ops::reshape(f.graph, out, {numIndices, tensor.shape()[1]}, f.tape));
}

Tensor sum(const Tensor& tensor, const std::vector<size_t>& dims) {
    Frame f("fr::ops::sum");
    mapping::setDefault(mapping::Linear(), {tensor});
    auto reduceDims = dims.empty() ? util::arange<size_t>(tensor.rank()) : dims;
    return Tensor::wrap(
        pag::ops::reduce(f.graph, tensor.pag(), reduceDims, popops::Operation::ADD, f.tape, f.di));
}

Tensor mean(const Tensor& tensor, const std::vector<size_t>& dims) {
    Frame f("fr::ops::mean");
    auto tensorSum = sum(tensor, dims);
    auto scale = static_cast<float>(f.graph.unwrap(tensorSum.pag()).numElements()) /
                 f.graph.unwrap(tensor.pag()).numElements();
    return tensorSum * constant(scale);
}

Tensor matMul(const Tensor& lhs, const Tensor& rhs) {
    Frame f("fr::ops::matMul");
    if (!lhs.valid()) {
        lhs.set(poplin::createMatMulInputLHS(f.graph.poplar(), lhs.dtype(), lhs.shape(),
                                             rhs.shape(), lhs.name(), {}, &f.matMulCache));
    }
    if (!rhs.valid()) {
        rhs.set(poplin::createMatMulInputRHS(f.graph.poplar(), rhs.dtype(), lhs.shape(),
                                             rhs.shape(), rhs.name(), {}, &f.matMulCache));
    }
    return Tensor::wrap(
        pag::ops::matMul(f.graph, lhs.pag(), rhs.pag(), f.tape, f.di, {}, &f.matMulCache));
}

Tensor logSoftmax(const Tensor& tensor) {
    Frame f("fr::ops::logSoftmax");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::logSoftmax(f.graph, tensor.pag(), f.tape, f.di));
}

Tensor sigmoid(const Tensor& tensor) {
    Frame f("fr::ops::sigmoid");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::sigmoid(f.graph, tensor.pag(), f.tape, f.di));
}

Tensor logSigmoid(const Tensor& tensor) {
    Frame f("fr::ops::logSigmoid");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::logSigmoid(f.graph, tensor.pag(), f.tape, f.di));
}

Tensor oneHot(const Tensor& tensor, size_t N, poplar::Type type) {
    Frame f("fr::ops::oneHot");
    mapping::setDefault(mapping::Linear(), {tensor});
    auto poplarTensor = f.graph.unwrap(tensor.pag());
    auto result = f.graph.poplar().addVariable(type, {poplarTensor.numElements(), N},
                                               poplar::VariableMappingMethod::LINEAR);
    popops::encodeOneHot(f.graph.poplar(), poplarTensor.flatten(), result, f.tape.prog(), f.di);
    return Tensor::wrap(
        f.graph.wrap(result.reshapePartial(0, 1, tensor.shape()), /*requiresGrad*/ false));
}

Tensor startGrad(const Tensor& tensor) {
    Frame f("fr::ops::startGrad");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::identity(f.graph, tensor.pag(), /*requiresGrad*/ true, f.tape));
}

Tensor concat(const std::vector<Tensor>& tensors, size_t dim) {
    Frame f("fr::ops::concat");
    mapping::setDefault(mapping::Linear(), tensors);
    return Tensor::wrap(pag::ops::concat(
        f.graph, util::mapVector(tensors, [](auto& t) { return t.pag(); }), dim, f.tape));
}

Tensor copyToLinearTensor(const Tensor& tensor,
                          std::optional<unsigned> minElementsPerTile,
                          std::optional<unsigned> grainSize) {
    Frame f("fr::ops::copyToLinearTensor");
    if (!tensor.valid()) {
        mapping::setDefault(mapping::Linear(), {tensor});
        return tensor;
    }
    poplar::DebugContext di(f.di);
    auto name = tensor.name() + "/copyToLinearTensor";
    auto poplarCopy = f.graph.poplar().addVariable(tensor.dtype(), tensor.shape(),
                                                   poplar::VariableMappingMethod::LINEAR, name);
    auto& target = f.graph.poplar().getTarget();
    poputil::mapTensorLinearly(
        f.graph.poplar(), poplarCopy,
        minElementsPerTile.value_or(128 / target.getTypeSize(tensor.dtype())),
        grainSize.value_or(target.getVectorWidth(tensor.dtype())));
    f.tape.prog().add(
        poplar::program::Copy(f.graph.unwrap(tensor.pag()), poplarCopy, /*dontOutline*/ false, di));

    auto requiresGrad = f.graph.requiresGrad(tensor.pag());
    auto output = fr::Tensor::wrap(f.graph.wrap(poplarCopy, requiresGrad));
    if (requiresGrad) {
        f.tape.addBackwardOp([=](pag::Graph& graph, poplar::program::Sequence& prog) {
            graph.addGrad(tensor.pag(), graph.grad(output.pag()), prog, {di, "grad"});
        });
    }
    return output;
}

Tensor allGather(const Tensor& tensor) {
    Frame f("fr::ops::allGather");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::allGatherCrossReplica(f.graph, tensor.pag(), f.tape, {}, f.di));
}

Tensor allToAll(const Tensor& tensor) {
    Frame f("fr::ops::allToAll");
    mapping::setDefault(mapping::Linear(), {tensor});
    return Tensor::wrap(pag::ops::allToAllCrossReplica(f.graph, tensor.pag(), f.tape, {}, f.di));
}

void forN(unsigned n, const std::function<void(const fr::Tensor&)>& body) {
    fr::Frame f("fr::ops::forN");
    f.tape.prog().add(popops::countedLoop(
        f.graph.poplar(), n,
        [body](const poplar::Tensor& index) {
            fr::SubProgramFrame f("body");
            body(fr::Tensor::wrap(f.graph.wrap(index.reshape({}), /*requiresGrad*/ false)));
            return f.tape.prog();
        },
        f.di));
}

}  // namespace ops

///////////////////////////////////////////////////////////////////////////////
// Streams & Buffers

Stream::Stream(const std::string& handle, const Tensor::Spec& spec, poplar::DataStreamType type)
    : m_shape(spec.shape) {
    Frame f("fr::Stream");
    switch (type) {
        case poplar::DataStreamType::HostToDeviceFIFO:
            m_stream = f.graph.poplar().addHostToDeviceFIFO(handle, spec.dtype,
                                                            util::numElements(spec.shape));
            break;
        case poplar::DataStreamType::DeviceToHostFIFO:
            m_stream = f.graph.poplar().addDeviceToHostFIFO(handle, spec.dtype,
                                                            util::numElements(spec.shape));
            break;
        default: {
            std::ostringstream msg;
            msg << "Unexpected poplar::DataStreamType " << static_cast<int>(type) << " for stream '"
                << handle << "'";
            throw std::invalid_argument(msg.str());
        }
    }
}

std::string Stream::handle() const {
    return m_stream.handle();
}
Tensor::Spec Stream::spec() const {
    return {m_shape, m_stream.elementType()};
}

Tensor Stream::read() const {
    Frame f("fr::Stream::read");
    auto poplarTensor = f.graph.poplar().addVariable(
        m_stream.elementType(), m_shape, poplar::VariableMappingMethod::LINEAR, m_stream.handle());
    auto tensor = Tensor::wrap(f.graph.wrap(poplarTensor, /*requiresGrad*/ false));
    f.tape.prog().add(
        poplar::program::Copy(m_stream, poplarTensor, /*optimiseMemory*/ false, f.di));
    return tensor;
}

void Stream::write(const Tensor& tensor) {
    Frame f("fr::Stream::write");
    util::checkArgument(tensor, "tensor", m_shape, {m_stream.elementType()});

    auto poplarTensor = f.graph.unwrap(tensor.pag());
    f.tape.prog().add(
        poplar::program::Copy(poplarTensor, m_stream, /*optimiseMemory*/ false, f.di));
}

Buffer::Buffer(const std::string& name, const Tensor::Spec& spec) {
    if (spec.shape.size() < 2) {
        std::ostringstream msg;
        msg << "Buffer shape should be >= 2D, actual shape: " << util::seq(spec.shape);
        throw std::invalid_argument(msg.str());
    }
    auto repeats = spec.shape.front();
    m_rowShape = {spec.shape.begin() + 1, spec.shape.end()};
    m_buffer = Environment::frame().graph.poplar().addRemoteBuffer(
        name, spec.dtype, util::numElements(m_rowShape), repeats);
}

Tensor Buffer::read(const Tensor& indices) const {
    Frame f("fr::Buffer::read");
    auto poplarTensor = f.graph.poplar().addVariable(m_buffer.elementType(), rwShape(indices),
                                                     poplar::VariableMappingMethod::LINEAR, f.di);
    f.tape.prog().add(poplar::program::Copy(
        m_buffer, poplarTensor.reshape({indices.shape().front(), m_buffer.numElements()}),
        f.graph.unwrap(indices.pag()), f.di));
    return Tensor::wrap(f.graph.wrap(poplarTensor, /*requiresGrad*/ false));
}

void Buffer::write(const Tensor& data, const Tensor& indices) {
    Frame f("fr::Buffer::write");
    util::checkArgument(data, "data", rwShape(indices), {m_buffer.elementType()});
    f.tape.prog().add(poplar::program::Copy(
        f.graph.unwrap(data.pag()).reshape({indices.shape().front(), m_buffer.numElements()}),
        m_buffer, f.graph.unwrap(indices.pag()), f.di));
}

size_t Buffer::totalBytes(const poplar::Target& target) const {
    auto numElementsPadded = std::pow(2, std::ceil(std::log2(m_buffer.numElements())));
    return numElementsPadded * m_buffer.getRepeats() * target.getTypeSize(m_buffer.elementType());
}

std::vector<size_t> Buffer::rwShape(const Tensor& indices) const {
    util::checkArgument(indices, "indices", {0u}, {poplar::UNSIGNED_INT});
    std::vector<size_t> shape;
    shape.push_back(indices.shape().front());
    std::copy(m_rowShape.begin(), m_rowShape.end(), std::back_inserter(shape));
    return shape;
}

}  // namespace fr
