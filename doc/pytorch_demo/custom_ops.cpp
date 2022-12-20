// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <vector>

#include <gcl/Collectives.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#pragma GCC diagnostic pop

// AllToAll

namespace Onnx::CustomOperators {
const popart::OperatorIdentifier AllToAll = {"ai.graphcore", "AllToAll", 1};
}  // namespace Onnx::CustomOperators

namespace {
namespace all_to_all {
struct CustomOp : popart::Op {
    CustomOp(const popart::OperatorIdentifier& _opid, const popart::Op::Settings& settings_)
        : popart::Op(_opid, settings_) {}
    std::unique_ptr<Op> clone() const final { return std::make_unique<CustomOp>(*this); }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    void setup() { outInfo(0) = inInfo(0); }  // shape inference
    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<popart::Op>> result;
        result.emplace_back(new CustomOp(*this));  // grad(allToAll) == allToAll
        return result;
    }
    const std::vector<popart::GradInOutMapper>& gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> inInfo = {
            {0, 0, popart::GradOpInType::GradOut}};
        return inInfo;
    }
    const std::map<int, int>& gradOutToNonGradIn() const {
        static const std::map<int, int> outInfo = {{0, 0}};
        return outInfo;
    }
};

struct CustomOpx : popart::popx::Opx {
    CustomOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<CustomOp>(op, Onnx::CustomOperators::AllToAll);
    }
    void grow(poplar::program::Sequence& prog) const final {
        auto input = get(inId(0));
        auto output = gcl::allToAllCrossReplica(graph(), input, prog, {}, debugContext("allToAll"));
        insert(outId(0), output);
    }
};

popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16, popart::DataType::FLOAT};
popart::OpCreator<CustomOp> opCreator(
    {{Onnx::CustomOperators::AllToAll,
      {popart::OpDefinition::Inputs({{"input", T}}), popart::OpDefinition::Outputs({{"output", T}}),
       popart::OpDefinition::Attributes({})}}},
    [](const popart::OpCreatorInfo& info) {
        return std::make_unique<CustomOp>(info.opid, info.settings);
    },
    true);
popart::popx::OpxCreator<CustomOpx> opxCreator(Onnx::CustomOperators::AllToAll);
}  // namespace all_to_all
}  // namespace

// RemoveAllReducePattern

namespace {
struct RemoveAllReducePattern : popart::PreAliasPattern {
    bool matches(popart::Op* op) const override {
        return op->isConvertibleTo<popart::ReplicatedAllReduceOp>();
    }

    std::vector<const popart::Tensor*> touches(popart::Op*) const override { return {}; }

    bool apply(popart::Op* op) const override {
        auto rar_op = static_cast<popart::ReplicatedAllReduceOp*>(op);
        if (rar_op->getReplicaGrouping().getGroupSize() == 1) {
            popart::Tensor* in_rar = rar_op->inTensor(popart::ReplicatedAllReduceOp::getInIndex());
            popart::Tensor* out_rar =
                rar_op->outTensor(popart::ReplicatedAllReduceOp::getOutIndex());
            // std::cerr << "Removing ReplicatedAllReduceOp with groupSize=1: " << in_rar->id
            //           << std::endl;
            for (auto cons : out_rar->consumers.getOps()) {
                for (auto in_index : cons->input->indices(out_rar)) {
                    cons->disconnectInTensor(out_rar);
                    cons->connectInTensor(in_index, in_rar->id);
                }
            }
            op->disconnectAllInputs();
            op->disconnectAllOutputs();
            op->getGraph().eraseOp(rar_op->id);
            return true;
        }
        return false;
    }
};

static popart::PatternCreator<RemoveAllReducePattern> RemoveAllReducePatternCreator(
    "RemoveAllReducePattern",
    false);
}  // namespace
