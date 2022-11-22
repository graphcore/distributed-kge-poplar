// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "pag.hpp"

#include <sstream>

#include <gcl/Collectives.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "poplar_extensions/distance.hpp"

namespace pag {

Tensor::Tensor() : m_id(Invalid) {}
Tensor::Tensor(ID id) : m_id(id) {}
Tensor::ID Tensor::id() const {
    return m_id;
}
bool Tensor::valid() const {
    return m_id != Invalid;
}

struct GraphImpl {
    struct TensorData {
        poplar::Tensor tensor;
        bool requiresGrad;
        poplar::Tensor grad;
    };
    poplar::Graph& graph;
    std::vector<TensorData> tensors;

    GraphImpl(poplar::Graph& graph) : graph(graph), tensors{{{}, false, {}}} {}
};

// Add a dummy TensorData, corresponding to Tensor::Invalid
Graph::Graph(poplar::Graph& graph) : m_impl(std::make_unique<GraphImpl>(graph)) {}

Graph::~Graph() = default;

const poplar::Graph& Graph::poplar() const {
    return m_impl->graph;
}
poplar::Graph& Graph::poplar() {
    return m_impl->graph;
}
poplar::Tensor Graph::unwrap(const Tensor& tensor) const {
    return m_impl->tensors[tensor.id()].tensor;
}
poplar::Tensor Graph::grad(const Tensor& tensor, bool checkValid) const {
    auto grad = m_impl->tensors[tensor.id()].grad;
    if (checkValid && !grad.valid()) {
        std::ostringstream msg;
        msg << "Requesting graph.grad() for a tensor '" << unwrap(tensor).getDebugStr()
            << "' that has no grad set";
        throw std::invalid_argument(msg.str());
    }
    return grad;
}
bool Graph::requiresGrad(const Tensor& tensor) const {
    return m_impl->tensors[tensor.id()].requiresGrad;
}

Tensor Graph::wrap(const poplar::Tensor& tensor, bool requiresGrad) {
    auto id = m_impl->tensors.size();
    m_impl->tensors.push_back({tensor, requiresGrad, poplar::Tensor{}});
    return Tensor(id);
}

void Graph::addGrad(const Tensor& tensor,
                    const poplar::Tensor& grad,
                    poplar::program::Sequence& prog,
                    const poplar::DebugContext& debugContext) {
    auto& data = m_impl->tensors[tensor.id()];

    if (grad.elementType() != data.tensor.elementType()) {
        std::ostringstream msg;
        msg << "Gradient type for tensor '" << data.tensor.getDebugStr()
            << "' does not match tensor type (tensor: " << data.tensor.elementType().toString()
            << ", gradient: " << grad.elementType().toString() << ")";
        throw std::invalid_argument(msg.str());
    }
    auto tensorShape = data.tensor.shape();
    auto gradShape = grad.shape();
    if (!std::equal(tensorShape.begin(), tensorShape.end(), gradShape.begin(), gradShape.end())) {
        std::ostringstream msg;
        msg << "Gradient shape for tensor '" << data.tensor.getDebugStr()
            << "' does not match tensor shape (tensor: " << data.tensor.shapeToString()
            << ", gradient: " << grad.shapeToString() << ")";
        throw std::invalid_argument(msg.str());
    }

    if (data.grad.valid()) {
        // If the gradient already exists, the tensor must have been used twice in the
        // forward pass, so we need to sum up the gradient in the backwards pass
        data.grad = popops::add(m_impl->graph, data.grad, grad, prog, debugContext);

    } else {
        // First gradient for this tensor
        data.grad = grad;
    }
}

poplar::program::Sequence& Tape::prog() {
    return m_prog;
}

void Tape::addBackwardOp(const BackwardOp& op) {
    m_backwardOps.push_back(op);
}

void Tape::backward(Graph& graph, const Tensor& root, const poplar::Tensor& rootGrad) {
    if (root.valid()) {
        poplar::Tensor rootGradTensor = rootGrad;
        if (!rootGradTensor.valid()) {
            rootGradTensor = graph.poplar().clone(graph.unwrap(root));
            popops::fill(graph.poplar(), rootGradTensor, prog(), 1.0f);
        }
        graph.addGrad(root, rootGradTensor, prog(), "grad");
    }
    for (auto it = m_backwardOps.rbegin(); it != m_backwardOps.rend(); ++it) {
        (*it)(graph, prog());
    }
}

namespace util {

poplar::Tensor broadcastGrad(Graph& graph,
                             const poplar::Tensor& grad,
                             const Tensor& tensor,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& debugContext) {
    auto poplarTensor = graph.unwrap(tensor);
    std::vector<size_t> dimsToReduce;
    for (auto gradDim = 0u; gradDim < grad.rank(); ++gradDim) {
        // Note - this can underflow unsigned subtraction, but then the following
        // test will do the right thing
        auto tensorDim = (gradDim + poplarTensor.rank()) - grad.rank();
        if (tensorDim >= poplarTensor.rank() || poplarTensor.dim(tensorDim) != grad.dim(gradDim)) {
            dimsToReduce.push_back(gradDim);
        }
    }
    return popops::reduce(graph.poplar(), grad, dimsToReduce, popops::Operation::ADD, prog,
                          debugContext)
        .reshape(poplarTensor.shape());
}

}  // namespace util

///////////////////////////////////////////////////////////////////////////////
// Ops library

namespace ops {

Tensor identity(Graph& graph, const Tensor& tensor, bool requiresGrad, Tape& tape) {
    auto poplarTensor = graph.unwrap(tensor);
    poplar::DebugContext di(poplarTensor.getDebugStr());
    auto output = graph.wrap(poplarTensor, requiresGrad);
    if (graph.requiresGrad(tensor)) {
        if (!requiresGrad) {
            throw std::invalid_argument(
                "Cannot use pag::ops::identity() as a stop grad (input requiresGrad is true, "
                "output requiresGrad is false)");
        }
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            graph.addGrad(tensor, graph.grad(output), prog, di);
        });
    }
    return output;
}

Tensor transpose(Graph& graph, const Tensor& tensor, Tape& tape) {
    auto poplarTensor = graph.unwrap(tensor);
    poplar::DebugContext di(poplarTensor.getDebugStr());
    auto requiresGrad = graph.requiresGrad(tensor);
    auto output = graph.wrap(poplarTensor.transpose(), requiresGrad);
    if (requiresGrad) {
        auto tensorShape = poplarTensor.shape();
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            graph.addGrad(tensor, graph.grad(output).transpose(), prog, di);
        });
    }
    return output;
}

Tensor reshape(Graph& graph, const Tensor& tensor, const std::vector<size_t>& shape, Tape& tape) {
    auto poplarTensor = graph.unwrap(tensor);
    poplar::DebugContext di(poplarTensor.getDebugStr());
    auto requiresGrad = graph.requiresGrad(tensor);
    auto output = graph.wrap(poplarTensor.reshape(shape), requiresGrad);
    if (requiresGrad) {
        auto tensorShape = poplarTensor.shape();
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            graph.addGrad(tensor, graph.grad(output).reshape(tensorShape), prog, di);
        });
    }
    return output;
}

Tensor slice(Graph& graph, const Tensor& tensor, size_t dim, poplar::Interval region, Tape& tape) {
    auto poplarTensor = graph.unwrap(tensor);
    poplar::DebugContext di(poplarTensor.getDebugStr());
    auto requiresGrad = graph.requiresGrad(tensor);
    auto output = graph.wrap(poplarTensor.slice(region, dim), requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            // Note: padding may be somewhat inefficient, rather than just updating a slice
            std::vector<std::ptrdiff_t> paddingLower(poplarTensor.rank());
            std::vector<std::ptrdiff_t> paddingUpper(poplarTensor.rank());
            paddingLower[dim] = region.begin();
            paddingUpper[dim] = poplarTensor.dim(dim) - region.upper();
            graph.addGrad(
                tensor, popops::pad(graph.poplar(), graph.grad(output), paddingLower, paddingUpper),
                prog, di);
        });
    }
    return output;
}

Tensor concat(Graph& graph, const std::vector<Tensor>& tensors, size_t dim, Tape& tape) {
    auto requiresGrad = std::accumulate(tensors.begin(), tensors.end(), false,
                                        [&graph](bool requiresGrad, const Tensor& tensor) {
                                            return requiresGrad || graph.requiresGrad(tensor);
                                        });
    std::vector<poplar::Tensor> poplarTensors;
    for (auto& tensor : tensors) {
        poplarTensors.push_back(graph.unwrap(tensor));
    }
    auto output = graph.wrap(poplar::concat(poplarTensors, dim), requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            size_t offset = 0;
            auto gradOutput = graph.grad(output);
            for (const auto& tensor : tensors) {
                auto poplarTensor = graph.unwrap(tensor);
                auto size = poplarTensor.dim(dim);
                graph.addGrad(tensor, gradOutput.slice({offset, offset + size}, dim), prog,
                              {poplar::DebugContext(poplarTensor.getDebugStr()), "grad"});
                offset += size;
            }
        });
    }
    return output;
}

std::vector<Tensor> split(Graph& graph,
                          const Tensor& tensor,
                          size_t dim,
                          const std::vector<size_t>& sizes,
                          Tape& tape) {
    auto poplarTensor = graph.unwrap(tensor);
    poplar::DebugContext di(poplarTensor.getDebugStr());
    auto requiresGrad = graph.requiresGrad(tensor);

    // Forward pass slicing
    auto totalSize = std::accumulate(sizes.begin(), sizes.end(), 0u);
    if (poplarTensor.dim(dim) != totalSize) {
        std::ostringstream msg;
        msg << "pag::ops::split of dim " << dim << ", size = " << poplarTensor.dim(dim)
            << ", but sum(sizes) = " << totalSize;
        throw std::invalid_argument(msg.str());
    }
    size_t offset = 0u;
    std::vector<Tensor> slices;
    for (auto size : sizes) {
        slices.push_back(
            graph.wrap(poplarTensor.slice({offset, offset + size}, dim), requiresGrad));
        offset += size;
    }

    // Backward pass concatenation
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            std::vector<poplar::Tensor> grads;
            for (const auto& slice : slices) {
                grads.push_back(graph.grad(slice));
            }
            graph.addGrad(tensor, poplar::concat(grads, dim), prog, di);
        });
    }
    return slices;
}

Tensor add(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(popops::add(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                         tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                auto gradA =
                    util::broadcastGrad(graph, gradOutput, A, prog, {debugContext, "grad_A"});
                graph.addGrad(A, gradA, prog, {debugContext, "grad_A"});
            }
            if (graph.requiresGrad(B)) {
                auto gradB =
                    util::broadcastGrad(graph, gradOutput, B, prog, {debugContext, "grad_B"});
                graph.addGrad(B, gradB, prog, {debugContext, "grad_B"});
            }
        });
    }
    return output;
}

Tensor sub(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(popops::sub(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                         tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto gradA = util::broadcastGrad(graph, gradOutput, A, prog, di);
                graph.addGrad(A, gradA, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                auto gradB = util::broadcastGrad(
                    graph, popops::neg(graph.poplar(), gradOutput, prog, di), B, prog, di);
                graph.addGrad(B, gradB, prog, di);
            }
        });
    }
    return output;
}

Tensor mul(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(popops::mul(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                         tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto gradA = util::broadcastGrad(
                    graph,
                    popops::mul(graph.poplar(), graph.unwrap(B), gradOutput, prog, di, options), A,
                    prog, di);
                graph.addGrad(A, gradA, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                auto gradB = util::broadcastGrad(
                    graph,
                    popops::mul(graph.poplar(), graph.unwrap(A), gradOutput, prog, di, options), B,
                    prog, di);
                graph.addGrad(B, gradB, prog, di);
            }
        });
    }
    return output;
}

Tensor div(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(popops::div(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                         tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto gradA = util::broadcastGrad(
                    graph,
                    popops::div(graph.poplar(), gradOutput, graph.unwrap(B), prog, di, options), A,
                    prog, di);
                graph.addGrad(A, gradA, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                namespace pe = popops::expr;
                auto gradB = util::broadcastGrad(
                    graph,
                    popops::map(graph.poplar(), -pe::_1 * pe::_2 / (pe::_3 * pe::_3),
                                {gradOutput, graph.unwrap(A), graph.unwrap(B)}, prog, di),
                    B, prog, di);
                graph.addGrad(B, gradB, prog, di);
            }
        });
    }
    return output;
}

Tensor neg(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(popops::neg(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad_A");
            auto gradA = popops::neg(graph.poplar(), graph.grad(output), prog, di, options);
            graph.addGrad(A, gradA, prog, di);
        });
    }
    return output;
}

Tensor abs(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(popops::abs(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad_A");
            namespace pe = popops::expr;
            auto gradA = popops::map(graph.poplar(), pe::Signum(pe::_2) * pe::_1,
                                     {graph.grad(output), graph.unwrap(A)}, prog, di);
            graph.addGrad(A, gradA, prog, di);
        });
    }
    return output;
}

Tensor square(Graph& graph,
              const Tensor& A,
              Tape& tape,
              const poplar::DebugContext& debugContext,
              const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output = graph.wrap(
        popops::square(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
        requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * 2.0f * pe::_2;
            auto grad =
                popops::map(graph.poplar(), expr, {graph.grad(output), graph.unwrap(A)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor pow(Graph& graph,
           const Tensor& A,
           float exponent,
           bool safeGradZero,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output = graph.wrap(
        popops::pow(graph.poplar(), graph.unwrap(A), exponent, tape.prog(), debugContext, options),
        requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = (pe::_1 * exponent * pe::Pow(pe::_2, pe::Const(exponent - 1.0f))).clone();
            if (safeGradZero) {
                expr = (pe::Select(*expr, pe::Const(0.0f), pe::_2 != 0.0f)).clone();
            }
            auto grad =
                popops::map(graph.poplar(), *expr, {graph.grad(output), graph.unwrap(A)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor sqrt(Graph& graph,
            const Tensor& A,
            Tape& tape,
            const poplar::DebugContext& debugContext,
            const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output = graph.wrap(
        popops::sqrt(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
        requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * 0.5f * pe::Inv(pe::_2);
            auto grad = popops::map(graph.poplar(), expr,
                                    {graph.grad(output), graph.unwrap(output)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor cbrt(Graph& graph,
            const Tensor& A,
            Tape& tape,
            const poplar::DebugContext& debugContext,
            const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output = graph.wrap(
        popops::cbrt(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
        requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * pe::Square(pe::Inv(pe::_2)) / 3.0f;
            auto grad = popops::map(graph.poplar(), expr,
                                    {graph.grad(output), graph.unwrap(output)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor sin(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(popops::sin(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * pe::Cos(pe::_2);
            auto grad =
                popops::map(graph.poplar(), expr, {graph.grad(output), graph.unwrap(A)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor cos(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(popops::cos(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = -pe::_1 * pe::Sin(pe::_2);
            auto grad =
                popops::map(graph.poplar(), expr, {graph.grad(output), graph.unwrap(A)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor dropout(Graph& graph,
               const Tensor& A,
               float p,
               Tape& tape,
               const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(poprand::dropout(graph.poplar(), nullptr, 0, graph.unwrap(A), graph.unwrap(A),
                                    1. - p, 1. / (1. - p), tape.prog(), debugContext),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            namespace pe = popops::expr;
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * pe::Cast(pe::_2 != 0.0f, poplar::FLOAT) / (1.0f - p);
            auto grad = popops::map(graph.poplar(), expr,
                                    {graph.grad(output), graph.unwrap(output)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    };
    return output;
};

Tensor cast(Graph& graph,
            const Tensor& A,
            poplar::Type type,
            Tape& tape,
            const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output =
        graph.wrap(popops::cast(graph.poplar(), graph.unwrap(A), type, tape.prog(), debugContext),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad_A");
            auto gradA = popops::cast(graph.poplar(), graph.grad(output),
                                      graph.unwrap(A).elementType(), prog, debugContext);
            graph.addGrad(A, gradA, prog, di);
        });
    }
    return output;
}

Tensor max(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext,
           const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(popops::max(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                         tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            auto gradFn = [&](const Tensor& tensor,
                              const poplar::DebugContext& di) -> poplar::Tensor {
                auto grad = popops::eq(graph.poplar(), graph.unwrap(tensor), graph.unwrap(output),
                                       prog, di);
                grad = popops::cast(graph.poplar(), grad, gradOutput.elementType(), prog, di);
                popops::mulInPlace(graph.poplar(), grad, gradOutput, prog, di);
                return util::broadcastGrad(graph, grad, tensor, prog, di);
            };
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                graph.addGrad(A, gradFn(A, di), prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                graph.addGrad(B, gradFn(B, di), prog, di);
            }
        });
    }
    return output;
}

Tensor matMul(Graph& graph,
              const Tensor& A,
              const Tensor& B,
              Tape& tape,
              const poplar::DebugContext& debugContext,
              const poplar::OptionFlags& options,
              poplin::PlanningCache* cache) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto output = graph.wrap(poplin::matMul(graph.poplar(), graph.unwrap(A), graph.unwrap(B),
                                            tape.prog(), debugContext, options, cache),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            // Note: use 'options' for fwd & bwd matmuls, as they're likely compatible (except for
            // `fullyConnectedPass`).
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto grad = poplin::matMul(graph.poplar(), gradOutput, graph.unwrap(B).transpose(),
                                           prog, di, options, cache);
                graph.addGrad(A, grad, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                auto grad = poplin::matMul(graph.poplar(), graph.unwrap(A).transpose(), gradOutput,
                                           prog, di, options, cache);
                graph.addGrad(B, grad, prog, di);
            }
        });
    }
    return output;
}

Tensor multiSlice(Graph& graph,
                  const Tensor& t,
                  const Tensor& offsets,
                  const std::vector<size_t>& dims,
                  const std::vector<size_t>& sizes,
                  Tape& tape,
                  const popops::SlicePlan& plan,
                  const poplar::OptionFlags& options,
                  const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(t);
    auto output =
        graph.wrap(popops::multiSlice(graph.poplar(), graph.unwrap(t), graph.unwrap(offsets), dims,
                                      sizes, tape.prog(), plan, options, debugContext),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad_t");
            auto gradOutput = graph.grad(output);

            popops::SlicePlan bwdPlan;
            poplar::OptionFlags bwdOptions;
            if (dims.size() == 1 && dims.front() == 0 && sizes.size() == 1 && sizes.front() == 1) {
                bwdPlan = popops::embedding::plan(graph.poplar(), gradOutput.elementType(),
                                                  graph.unwrap(t).dim(0), graph.unwrap(t).dim(1),
                                                  {graph.unwrap(offsets).dim(0)}, bwdOptions);
            }

            auto scale = graph.poplar().addConstant(poplar::FLOAT, {}, 1.0f, {di, "scale"});
            graph.poplar().setTileMapping(scale, 17965 % graph.poplar().getTarget().getNumTiles());

            auto gradT = popops::createSliceableTensor(graph.poplar(), gradOutput.elementType(),
                                                       graph.unwrap(t).shape(), dims, sizes,
                                                       /*minGrainSize*/ 0, di);
            popops::zero(graph.poplar(), gradT, prog, di);
            popops::multiUpdateAdd(graph.poplar(), gradT, gradOutput, graph.unwrap(offsets), scale,
                                   dims, sizes, prog, bwdPlan, bwdOptions, di);
            graph.addGrad(t, gradT, prog, di);
        });
    }
    return output;
}

Tensor reduce(Graph& graph,
              const Tensor& in,
              const std::vector<size_t>& dims,
              popops::ReduceParams params,
              Tape& tape,
              const poplar::DebugContext& debugContext,
              const poplar::OptionFlags& options) {
    if (params.useScale || params.op != popops::Operation::ADD) {
        throw std::invalid_argument(
            "pag::ops::reduce() is only implemented for Operation::ADD, with no scaling");
    }
    auto requiresGrad = graph.requiresGrad(in);
    auto output = graph.wrap(popops::reduce(graph.poplar(), graph.unwrap(in), dims, params,
                                            tape.prog(), debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto grad = graph.grad(output);
            auto sortedDims = dims;
            std::sort(sortedDims.begin(), sortedDims.end());
            for (auto dim : sortedDims) {
                grad = grad.expand({dim}).broadcast(graph.unwrap(in).dim(dim), dim);
            }
            graph.addGrad(in, grad, prog, {debugContext, "grad_in"});
        });
    }
    return output;
}

Tensor l1distance(Graph& graph,
                  const Tensor& A,
                  const Tensor& B,
                  Tape& tape,
                  const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto poplarOutput = poplar_extensions::l1distance(graph.poplar(), graph.unwrap(A),
                                                      graph.unwrap(B), tape.prog(), debugContext);
    auto output = graph.wrap(poplarOutput, requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto gradA = poplar_extensions::l1distancegrad(
                    graph.poplar(), graph.unwrap(A), graph.unwrap(B), gradOutput, prog, di);
                graph.addGrad(A, gradA, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                auto gradB = poplar_extensions::l1distancegrad(graph.poplar(), graph.unwrap(B),
                                                               graph.unwrap(A),
                                                               gradOutput.transpose(), prog, di);
                graph.addGrad(B, gradB, prog, di);
            }
        });
    }
    return output;
}

Tensor l2distance(Graph& graph,
                  const Tensor& A,
                  const Tensor& B,
                  Tape& tape,
                  const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(A) || graph.requiresGrad(B);
    auto poplarOutput = poplar_extensions::l2distance(graph.poplar(), graph.unwrap(A),
                                                      graph.unwrap(B), tape.prog(), debugContext);
    auto output = graph.wrap(poplarOutput, requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            auto gradOutput = graph.grad(output);
            if (graph.requiresGrad(A)) {
                poplar::DebugContext di(debugContext, "grad_A");
                auto gradA = poplar_extensions::l2distancegrad(graph.poplar(), graph.unwrap(A),
                                                               graph.unwrap(B), poplarOutput,
                                                               gradOutput, prog, di);
                graph.addGrad(A, gradA, prog, di);
            }
            if (graph.requiresGrad(B)) {
                poplar::DebugContext di(debugContext, "grad_B");
                auto gradB = poplar_extensions::l2distancegrad(
                    graph.poplar(), graph.unwrap(B), graph.unwrap(A), poplarOutput.transpose(),
                    gradOutput.transpose(), prog, di);
                graph.addGrad(B, gradB, prog, di);
            }
        });
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////
// Neural Networks

Tensor logSoftmax(Graph& graph,
                  const Tensor& t,
                  Tape& tape,
                  const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(t);
    auto output =
        graph.wrap(popnn::logSoftmax(graph.poplar(), graph.unwrap(t), tape.prog(), debugContext),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            auto gradOutput = graph.grad(output);
            auto finalDim = gradOutput.rank() - 1;
            auto sumGradOutput = popops::reduce(graph.poplar(), gradOutput, {finalDim},
                                                popops::Operation::ADD, prog, di)
                                     .expand({finalDim});
            namespace pe = popops::expr;
            auto grad = popops::map(graph.poplar(), pe::_1 - pe::_2 * pe::Exp(pe::_3),
                                    {gradOutput, sumGradOutput, graph.unwrap(output)}, prog, di);
            graph.addGrad(t, grad, prog, di);
        });
    }
    return output;
}

Tensor sigmoid(Graph& graph,
               const Tensor& A,
               Tape& tape,
               const poplar::DebugContext& debugContext,
               const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(A);
    auto output = graph.wrap(
        popops::sigmoid(graph.poplar(), graph.unwrap(A), tape.prog(), debugContext, options),
        requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            namespace pe = popops::expr;
            auto expr = pe::_1 * pe::_2 * (1 - pe::_2);
            auto grad = popops::map(graph.poplar(), expr,
                                    {graph.grad(output), graph.unwrap(output)}, prog, di);
            graph.addGrad(A, grad, prog, di);
        });
    }
    return output;
}

Tensor logSigmoid(Graph& graph,
                  const Tensor& t,
                  Tape& tape,
                  const poplar::DebugContext& debugContext) {
    auto requiresGrad = graph.requiresGrad(t);
    namespace pe = popops::expr;
    auto expr =
        pe::TernaryOp(pe::TernaryOpType::SELECT,
                      -pe::Log1p(pe::Exp(-pe::Max(pe::_1, pe::Const(-11)))), pe::_1, pe::_1 > -11);
    auto output =
        graph.wrap(popops::map(graph.poplar(), expr, {graph.unwrap(t)}, tape.prog(), debugContext),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            auto expr = pe::_1 * pe::Inv(1.0f + pe::Exp(pe::_2));
            auto grad =
                popops::map(graph.poplar(), expr, {graph.grad(output), graph.unwrap(t)}, prog, di);
            graph.addGrad(t, grad, prog, di);
        });
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////
// Collectives

Tensor allToAllCrossReplica(Graph& graph,
                            const Tensor& data,
                            Tape& tape,
                            const gcl::CommGroup& group,
                            const poplar::DebugContext& debugContext,
                            const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(data);
    auto output = graph.wrap(gcl::allToAllCrossReplica(graph.poplar(), graph.unwrap(data),
                                                       tape.prog(), group, debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            auto gradOutput = graph.grad(output);
            auto gradData =
                gcl::allToAllCrossReplica(graph.poplar(), gradOutput, prog, group, di, options);
            graph.addGrad(data, gradData, prog, di);
        });
    }
    return output;
}

Tensor reduceScatterCrossReplica(Graph& graph,
                                 const Tensor& data,
                                 gcl::CollectiveOperator op,
                                 Tape& tape,
                                 const gcl::CommGroup& group,
                                 const poplar::DebugContext& debugContext,
                                 const poplar::OptionFlags& options) {
    if (op != gcl::CollectiveOperator::ADD) {
        std::ostringstream msg;
        msg << "pag::ops::reduceScatterCrossReplica is only implemented for "
               "gcl::CollectiveOperator::ADD, but was called with "
            << op;
        throw std::invalid_argument(msg.str());
    }
    auto requiresGrad = graph.requiresGrad(data);
    auto output =
        graph.wrap(gcl::reduceScatterCrossReplica(graph.poplar(), graph.unwrap(data), op,
                                                  tape.prog(), group, debugContext, options),
                   requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            auto gradOutput = graph.grad(output);
            auto gradData =
                gcl::allGatherCrossReplica(graph.poplar(), gradOutput, prog, group, di, options)
                    .flatten();
            graph.addGrad(data, gradData, prog, di);
        });
    }
    return output;
}

Tensor allGatherCrossReplica(Graph& graph,
                             const Tensor& data,
                             Tape& tape,
                             const gcl::CommGroup& group,
                             const poplar::DebugContext& debugContext,
                             const poplar::OptionFlags& options) {
    auto requiresGrad = graph.requiresGrad(data);
    auto output = graph.wrap(gcl::allGatherCrossReplica(graph.poplar(), graph.unwrap(data),
                                                        tape.prog(), group, debugContext, options),
                             requiresGrad);
    if (requiresGrad) {
        tape.addBackwardOp([=](Graph& graph, poplar::program::Sequence& prog) {
            poplar::DebugContext di(debugContext, "grad");
            auto gradOutput = graph.grad(output);
            if (!gradOutput.isParallelWriteable()) {
                // It's unclear why gcl::reduceScatterCrossReplica includes the assertion
                // isParallelWritable() on the input tensor.
                gradOutput = poputil::duplicate(graph.poplar(), gradOutput, prog);
            }
            auto gradData = gcl::reduceScatterCrossReplica(graph.poplar(), gradOutput.flatten(),
                                                           gcl::CollectiveOperator::ADD, prog,
                                                           group, di, options)
                                .reshape(graph.unwrap(data).shape());
            graph.addGrad(data, gradData, prog, di);
        });
    }
    return output;
}

}  // namespace ops
}  // namespace pag
