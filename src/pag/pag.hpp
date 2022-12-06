// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef PAG_HPP
#define PAG_HPP

#include <memory>
#include <vector>

#include <gcl/Collectives.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplin/Convolution.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Reduce.hpp>

/**
 * PAG = Poplar AutoGrad
 *
 * A rough-and-ready implementation of autograd for plain Poplar.
 */
namespace pag {

struct GraphImpl;

/**
 * A forward pass activation or weight, that can have a backward pass (gradient) associated
 * with it (see Graph).
 */
struct Tensor {
    using ID = unsigned;
    static constexpr ID Invalid = 0u;

    Tensor();
    explicit Tensor(ID id);
    ID id() const;
    bool valid() const;

   private:
    ID m_id;
};

/**
 * A stash of forward pass (activation) and backward pass (gradient) `poplar::Tensor`s.
 */
struct Graph {
    explicit Graph(poplar::Graph&);
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph();

    const poplar::Graph& poplar() const;
    poplar::Graph& poplar();
    poplar::Tensor unwrap(const Tensor& tensor) const;
    poplar::Tensor grad(const Tensor& tensor, bool checkValid = true) const;
    bool requiresGrad(const Tensor& tensor) const;

    /**
     * Add a `poplar::Tensor` to the graph.
     */
    Tensor wrap(const poplar::Tensor& tensor, bool requiresGrad);

    /**
     * Sets or accumulates a gradient tensor.
     *
     * If there is already a gradient tensor set (for example when the tensor was used by
     * multiple forward pass operations), accumulate the gradient.
     */
    void addGrad(const Tensor& tensor,
                 const poplar::Tensor& grad,
                 poplar::program::Sequence& prog,
                 const poplar::DebugContext& debugContext);

   private:
    std::unique_ptr<GraphImpl> m_impl;
};

/**
 * A sequential program that records forward pass operations, allowing generation of a
 * backward pass.
 */
struct Tape {
    using BackwardOp = std::function<void(Graph&, poplar::program::Sequence&)>;

    poplar::program::Sequence& prog();

    void addBackwardOp(const BackwardOp& op);
    void backward(Graph& graph, const Tensor& root = {}, const poplar::Tensor& rootGrad = {});

   private:
    poplar::program::Sequence m_prog;
    std::vector<BackwardOp> m_backwardOps;
};

namespace util {

poplar::Tensor broadcastGrad(Graph& graph,
                             const poplar::Tensor& grad,
                             const Tensor& tensor,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& debugContext = {});

}  // namespace util

/**
 * Extensible library of differentiable ops
 */
namespace ops {

/**
 * Can be used as a "start grad".
 */
Tensor identity(Graph& graph, const Tensor& tensor, bool requiresGrad, Tape& tape);

Tensor transpose(Graph& graph, const Tensor& tensor, Tape& tape);

Tensor reshape(Graph& graph, const Tensor& tensor, const std::vector<size_t>& shape, Tape& tape);

Tensor slice(Graph& graph, const Tensor& tensor, size_t dim, poplar::Interval region, Tape& tape);

Tensor concat(Graph& graph, const std::vector<Tensor>& tensors, size_t dim, Tape& tape);

/**
 * Splits a tensor in dimension `dim`, where each part has specified size.
 *
 * Requires: sum(sizes) == tensor.dim(dim)
 */
std::vector<Tensor> split(Graph& graph,
                          const Tensor& tensor,
                          size_t dim,
                          const std::vector<size_t>& sizes,
                          Tape& tape);

Tensor add(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor sub(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor mul(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor div(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor neg(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor abs(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor square(Graph& graph,
              const Tensor& A,
              Tape& tape,
              const poplar::DebugContext& debugContext = {},
              const poplar::OptionFlags& options = {});

Tensor pow(Graph& graph,
           const Tensor& A,
           float exponent,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor sqrt(Graph& graph,
            const Tensor& A,
            Tape& tape,
            const poplar::DebugContext& debugContext = {},
            const poplar::OptionFlags& options = {});

Tensor cbrt(Graph& graph,
            const Tensor& A,
            Tape& tape,
            const poplar::DebugContext& debugContext = {},
            const poplar::OptionFlags& options = {});

Tensor sin(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor cos(Graph& graph,
           const Tensor& A,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor dropout(Graph& graph,
               const Tensor& A,
               float p,
               Tape& tape,
               const poplar::DebugContext& debugContext = {});

Tensor cast(Graph& graph,
            const Tensor& A,
            poplar::Type type,
            Tape& tape,
            const poplar::DebugContext& debugContext = {});

/**
 * Note: if A == B, both tensors will receive gradient in the bwd pass.
 */
Tensor max(Graph& graph,
           const Tensor& A,
           const Tensor& B,
           Tape& tape,
           const poplar::DebugContext& debugContext = {},
           const poplar::OptionFlags& options = {});

Tensor matMul(Graph& graph,
              const Tensor& A,
              const Tensor& B,
              Tape& tape,
              const poplar::DebugContext& debugContext = {},
              const poplar::OptionFlags& options = {},
              poplin::PlanningCache* cache = nullptr);

/**
 * WARNING - tested only for very limited cases (2D-table embedding lookups).
 */
Tensor multiSlice(Graph& graph,
                  const Tensor& t,
                  const Tensor& offsets,
                  const std::vector<size_t>& dims,
                  const std::vector<size_t>& sizes,
                  Tape& tape,
                  const popops::SlicePlan& plan,
                  const poplar::OptionFlags& options,
                  const poplar::DebugContext& debugContext = {});

Tensor reduce(Graph& graph,
              const Tensor& in,
              const std::vector<size_t>& dims,
              popops::ReduceParams params,
              Tape& tape,
              const poplar::DebugContext& debugContext = {},
              const poplar::OptionFlags& options = {});

Tensor l1distance(Graph& graph,
                  const Tensor& A,
                  const Tensor& B,
                  Tape& tape,
                  const poplar::DebugContext& debugContext = {});

Tensor l2distance(Graph& graph,
                  const Tensor& A,
                  const Tensor& B,
                  Tape& tape,
                  const poplar::DebugContext& debugContext = {});

///////////////////////////////////////////////////////////////////////////////
// Neural Networks

Tensor logSoftmax(Graph& graph,
                  const Tensor& t,
                  Tape& tape,
                  const poplar::DebugContext& debugContext = {});

Tensor sigmoid(Graph& graph,
               const Tensor& A,
               Tape& tape,
               const poplar::DebugContext& debugContext = {},
               const poplar::OptionFlags& options = {});

Tensor logSigmoid(Graph& graph,
                  const Tensor& t,
                  Tape& tape,
                  const poplar::DebugContext& debugContext = {});

///////////////////////////////////////////////////////////////////////////////
// Collectives

Tensor allToAllCrossReplica(Graph& graph,
                            const Tensor& data,
                            Tape& tape,
                            const gcl::CommGroup& group,
                            const poplar::DebugContext& debugContext = {},
                            const poplar::OptionFlags& options = {});

Tensor reduceScatterCrossReplica(Graph& graph,
                                 const Tensor& data,
                                 gcl::CollectiveOperator op,
                                 Tape& tape,
                                 const gcl::CommGroup& group,
                                 const poplar::DebugContext& debugContext = {},
                                 const poplar::OptionFlags& options = {});

Tensor allGatherCrossReplica(Graph& graph,
                             const Tensor& data,
                             Tape& tape,
                             const gcl::CommGroup& group,
                             const poplar::DebugContext& debugContext = {},
                             const poplar::OptionFlags& options = {});

}  // namespace ops

}  // namespace pag

#endif  // PAG_HPP
