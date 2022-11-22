// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "poplar_kge.hpp"
#include "fructose/frnn.hpp"
#include "fructose/fructose.hpp"

#include <cmath>
#include <iostream>
#include <sstream>

#include <poplar/CSRFunctions.hpp>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Loop.hpp>
#include <popops/TopK.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace poplar {
template <>
struct equivalent_device_type<poplar_kge::float16> {
    const Type& value = HALF;
};
}  // namespace poplar

namespace poplar_kge {

namespace {
template <class T>
const T& get(const std::unordered_map<std::string, T>& data, const std::string& key) {
    auto it = data.find(key);
    if (it == data.end()) {
        std::ostringstream msg;
        msg << "Key not found '" << key << "'";
        throw std::invalid_argument(msg.str());
    }
    return it->second;
}
template <class T>
T& get(std::unordered_map<std::string, T>& data, const std::string& key) {
    return const_cast<T&>(get(static_cast<const std::unordered_map<std::string, T>&>(data), key));
}
template <class T>
const T& extract(const Batch& data, const std::string& key) {
    auto& value = get(data, key);
    if (!std::holds_alternative<T>(value)) {
        std::ostringstream msg;
        msg << "Key '" << key << "' holds value of unexpected type. Expected: '" << typeid(T).name()
            << "', actual: '" << std::visit([](auto& v) { return typeid(v).name(); }, value)
            << "'.";
        throw std::invalid_argument(msg.str());
    }
    return std::get<T>(value);
}
template <class T>
T& extract(Batch& data, const std::string& key) {
    return const_cast<T&>(extract<T>(static_cast<const Batch&>(data), key));
}

template <class... Args>
struct GetData {};
template <class Head, class... Tail>
struct GetData<Head, Tail...> {
    static std::tuple<void*, void*> value(const std::string& name,
                                          const fr::Tensor::Spec& spec,
                                          Batch::mapped_type& value) {
        if (std::holds_alternative<ArrayView<Head>>(value)) {
            auto& array = std::get<ArrayView<Head>>(value);
            auto headType = poplar::equivalent_device_type<Head>().value;
            if (headType != spec.dtype) {
                std::ostringstream msg;
                msg << "'" << name << "' expected array of type " << spec.dtype
                    << ", actual: " << headType;
                throw std::invalid_argument(msg.str());
            }
            if (fr::util::seq(spec.shape) != fr::util::seq(array.shape())) {
                std::ostringstream msg;
                msg << "'" << name << "' expected array of shape " << fr::util::seq(spec.shape)
                    << ", actual: " << fr::util::seq(array.shape());
                throw std::invalid_argument(msg.str());
            }
            return {array.data(), array.data() + fr::util::numElements(array.shape())};
        }
        return GetData<Tail...>::value(name, spec, value);
    }
};
template <>
struct GetData<> {
    static std::tuple<void*, void*> value(const std::string& name,
                                          const fr::Tensor::Spec& spec,
                                          Batch::mapped_type& value) {
        std::ostringstream msg;
        msg << "'" << name << "' expected type " << spec.dtype << ", actual: <not found> (index "
            << value.index() << ")";
        throw std::invalid_argument(msg.str());
    }
};

poplar::Device attach(const Batch& settings) {
    auto nShard = extract<unsigned>(settings, "model.n_shard");
    auto type = extract<std::string>(settings, "execution.device");
    if (type == "cpu") {
        return poplar::Device::createCPUDevice(nShard);

    } else if (type == "ipu") {
        auto manager = poplar::DeviceManager::createDeviceManager();
        for (auto&& device : manager.getDevices(poplar::TargetType::IPU, nShard)) {
            if (device.attach()) {
                return std::move(device);
            }
        }
        std::ostringstream msg;
        msg << "Could not attach to an " << nShard << " IPU device";
        throw std::runtime_error(msg.str());

    } else if (type == "ipu_model") {
        poplar::IPUModel model;
        model.numIPUs = nShard;
        return model.createDevice();

    } else {
        std::ostringstream msg;
        msg << "Unexpected setting \"device\": '" << nShard << "', expected \"cpu\" or \"ipu\"";
        throw std::invalid_argument(msg.str());
    }
}

struct Model {
    struct FeatureNetwork {
        fr::Tensor featureProjection;
        fr::Tensor mlpUpProjection;
        fr::Tensor mlpDownProjection;
    };

    struct NormRegularisation {
        float power;
        float weight;

        NormRegularisation(const Batch& settings, const std::string& prefix)
            : power(extract<float>(settings, prefix + ".power")),
              weight(extract<float>(settings, prefix + ".weight")) {}
    };

    // Settings
    Batch settings;
    unsigned seed;
    std::string scoreFn;
    std::string distanceFn;
    unsigned nShard;
    unsigned nEntity;
    unsigned nRelationType;
    unsigned entityEmbeddingSize;
    unsigned relationEmbeddingSize;
    unsigned entityFeatureSize;
    unsigned featureMlpSize;
    float featureDropout;
    bool shareFeatureNetworks;
    float gamma;
    float initScale;
    fr::nn::AdamParams adamParams;
    std::unordered_map<std::string, float> learningRateModifiers;
    unsigned batchSize;
    unsigned a2aSize;
    float negativeAdversarialScale;
    std::string lossFn;
    NormRegularisation embeddingRegularisation;
    NormRegularisation featureRegularisation;
    NormRegularisation hiddenRegularisation;
    float softmaxLossCorrectionWeight;
    float lossScale;

    poplar::Type dtype;
    unsigned trainStepsPerProgramRun;
    unsigned rwBatchSize;
    unsigned predictHrBatchSize;
    unsigned predictTailBatchSize;
    unsigned predictNBest;

    // Variables/buffers
    fr::Tensor relationEmbedding;
    fr::Tensor relationNormal;
    FeatureNetwork headFeatureNetwork;
    FeatureNetwork tailFeatureNetwork;
    fr::Buffer entityData;

    // Parameters, a trainable subset of variables
    std::unordered_set<std::string> parameters;

    Model(const Batch& settings)
        : settings(settings),
          seed(extract<unsigned>(settings, "model.seed")),
          scoreFn(extract<std::string>(settings, "model.score_fn")),
          distanceFn(extract<std::string>(settings, "model.distance_fn")),
          nShard(extract<unsigned>(settings, "model.n_shard")),
          nEntity(extract<unsigned>(settings, "model.n_entity")),
          nRelationType(extract<unsigned>(settings, "model.n_relation_type")),
          entityEmbeddingSize(extract<unsigned>(settings, "model.embedding_size")),
          entityFeatureSize(extract<unsigned>(settings, "model.entity_feature_size")),
          featureMlpSize(extract<unsigned>(settings, "model.feature_mlp_size")),
          featureDropout(extract<float>(settings, "model.feature_dropout")),
          shareFeatureNetworks(extract<bool>(settings, "model.share_feature_networks")),
          gamma(extract<float>(settings, "model.gamma")),
          initScale(extract<float>(settings, "model.init_scale")),
          adamParams{/*betaM*/ extract<float>(settings, "training.adam_beta_m"),
                     /*betaV*/ extract<float>(settings, "training.adam_beta_v"),
                     /*epsilon*/ extract<float>(settings, "training.adam_epsilon"),
                     /*weightDecay*/ extract<float>(settings, "training.weight_decay")},
          learningRateModifiers{
              extract<std::unordered_map<std::string, float>>(settings,
                                                              "training.learning_rate_modifiers")},
          batchSize(extract<unsigned>(settings, "data.batch_size")),
          a2aSize(extract<unsigned>(settings, "data.a2a_size")),
          negativeAdversarialScale(extract<float>(settings, "model.negative_adversarial_scale")),
          lossFn(extract<std::string>(settings, "training.loss.type")),
          embeddingRegularisation(settings, "training.embedding_regularisation"),
          featureRegularisation(settings, "training.feature_regularisation"),
          hiddenRegularisation(settings, "training.hidden_regularisation"),
          softmaxLossCorrectionWeight(
              lossFn == "softmax" ? extract<float>(settings, "training.loss.correction_weight")
                                  : 0),
          lossScale(extract<float>(settings, "training.loss_scale")),
          trainStepsPerProgramRun(
              extract<unsigned>(settings, "execution.train_steps_per_program_run")),
          rwBatchSize(extract<unsigned>(settings, "execution.rw_batch_size")),
          predictHrBatchSize(extract<unsigned>(settings, "execution.predict_hr_batch_size")),
          predictTailBatchSize(extract<unsigned>(settings, "execution.predict_tail_batch_size")),
          predictNBest(extract<unsigned>(settings, "execution.predict_n_best")) {
        if (extract<std::string>(settings, "execution.dtype") == "float32") {
            dtype = poplar::FLOAT;
        } else if (extract<std::string>(settings, "execution.dtype") == "float16") {
            dtype = poplar::HALF;
        } else {
            std::ostringstream msg;
            msg << "'execution.dtype' must either be 'float16' or 'float32', not '"
                << extract<std::string>(settings, "execution.dtype") << "'";
            throw std::invalid_argument(msg.str());
        }
        // Register variables/parameters/buffers
        relationEmbeddingSize =
            (scoreFn == "RotatE") ? entityEmbeddingSize / 2 : entityEmbeddingSize;
        relationEmbedding =
            addParameter("relation_embedding", {nRelationType, relationEmbeddingSize});
        if (scoreFn == "TransH") {
            relationNormal =
                addParameter("relation_normal", {nRelationType, relationEmbeddingSize});
        }
        auto createFeatureNetwork = [this](const std::string& prefix) {
            FeatureNetwork network;
            network.featureProjection = addParameter(prefix + "feature_projection",
                                                     {entityFeatureSize, entityEmbeddingSize});
            if (featureMlpSize > 0) {
                network.mlpUpProjection = addParameter(prefix + "mlp_up_projection",
                                                       {2 * entityEmbeddingSize, featureMlpSize});
                network.mlpDownProjection = addParameter(prefix + "mlp_down_projection",
                                                         {featureMlpSize, entityEmbeddingSize});
            }
            return network;
        };
        if (shareFeatureNetworks) {
            headFeatureNetwork = tailFeatureNetwork = createFeatureNetwork("");
        } else {
            headFeatureNetwork = createFeatureNetwork("head_");
            tailFeatureNetwork = createFeatureNetwork("tail_");
        }
        entityData = fr::Buffer("entity_data",
                                {{nEntity, 3 * entityEmbeddingSize + entityFeatureSize}, dtype});

        // Checks
        for (auto& item : learningRateModifiers) {
            if (!parameters.count(item.first) && item.first != "entity_embedding") {
                std::ostringstream msg;
                msg << "Bad training.learning_rate_modifier: unknown parameter '" << item.first
                    << "'";
                throw std::invalid_argument(msg.str());
            }
        }
    }

    // Utilities

    fr::Tensor addParameter(const std::string& name, const fr::Tensor::Shape& shape) {
        assert(parameters.find(name) == parameters.end() && "duplicate parameter");
        parameters.insert(name);

        if (shape[0] % nShard) {
            std::ostringstream msg;
            msg << "Parameter '" << name << "' shape[0] (" << shape[0]
                << ") is not divisible by n_shard (" << nShard
                << "). Note shape: " << fr::util::seq(shape);
            throw std::invalid_argument(msg.str());
        }
        auto shardShape = shape;
        shardShape[0] /= nShard;
        return fr::ops::variable(name, {shardShape, poplar::FLOAT});
    }

    std::unordered_map<std::string, fr::Tensor::Spec> finaliseVariables() const {
        std::unordered_map<std::string, fr::Tensor::Spec> result;
        for (auto& entry : fr::Environment::rootFrame().variables) {
            entry.second.hostAccess();
            auto shape = entry.second.shape();
            if (shape.empty()) {
                shape.push_back(1);
            }
            shape.front() *= nShard;
            result.insert({entry.first, {shape, entry.second.dtype()}});
        }
        return result;
    }

    // Helpers

    fr::Tensor modifiedStepSize(const fr::Tensor& globalStepSize, const std::string& parameter) {
        if (learningRateModifiers.count(parameter)) {
            return globalStepSize * fr::ops::constant(learningRateModifiers[parameter]);
        }
        return globalStepSize;
    }

    void updateParameters(const fr::Tensor& stepSize) {
        for (auto& name : parameters) {
            auto& variable = get(fr::Environment::rootFrame().variables, name);
            auto adamM = fr::ops::variable(variable.name() + "/adam_m", variable.spec(),
                                           /*requiresGrad*/ false);
            auto adamV = fr::ops::variable(variable.name() + "/adam_v", variable.spec(),
                                           /*requiresGrad*/ false);
            fr::nn::adam(variable, adamM, adamV, modifiedStepSize(stepSize, variable.name()),
                         adamParams);
        }
    }

    fr::Tensor gatherShards(const fr::Tensor& shard) {
        if (2 <= nShard) {
            auto shape = shard.shape();
            shape.front() *= nShard;
            return fr::ops::allGather(shard).reshape(shape);
        }
        return shard;
    }

    /**
     * Implements a dynamic slice with a switch statement, which gives more predictable memory
     * usage than popops::dynamicSlice.
     */
    static fr::Tensor switchedSlice(const fr::Tensor& tensor, const fr::Tensor& index) {
        fr::Frame f("switchedSlice");
        if (f.graph.requiresGrad(tensor.pag())) {
            throw std::logic_error("switchedSlice does not support gradients");
        }
        auto poplarTensor = f.graph.unwrap(tensor.pag());
        auto output = fr::Tensor::declare(
            {{tensor.shape().begin() + 1, tensor.shape().end()}, tensor.dtype()},
            /*requiresGrad*/ false, tensor.name() + "/switchedSlice");
        fr::mapping::setDefault(fr::mapping::Copy(poplarTensor[0]), {output});

        poplar::program::Switch switch_(f.graph.unwrap(index.pag()), f.di);
        for (auto i = 0u; i < tensor.shape()[0]; ++i) {
            std::ostringstream name;
            name << "case_" << i;
            switch_.add(i, poplar::program::Copy(poplarTensor[i], f.graph.unwrap(output.pag()),
                                                 /*dontOutline*/ false, {f.di, name.str()}));
        }
        f.tape.prog().add(switch_);
        return output;
    }

    // Distance functions
    // Note: these direct implementations show high memory usage

    fr::Tensor l1distance(const fr::Tensor& a, const fr::Tensor& b) {
        return fr::ops::sum(fr::ops::abs(a.reshape({a.shape().at(0), 1, a.shape()[1]}) -
                                         b.reshape({1, b.shape().at(0), b.shape()[1]})),
                            {2});
    }

    fr::Tensor l2norm(const fr::Tensor& a) {
        return fr::ops::sqrt(fr::ops::sum(fr::ops::square(a), {a.rank() - 1}));
    }

    fr::Tensor lpnorm(const fr::Tensor& a, float p) {
        return fr::ops::pow(fr::ops::sum(fr::ops::abs(fr::ops::pow(a, p)), {a.rank() - 1}), 1. / p,
                            /*safeGradZero=*/true);
    }

    fr::Tensor l2distance(const fr::Tensor& a, const fr::Tensor& b) {
        return l2norm(a.reshape({a.shape().at(0), 1, a.shape()[1]}) -
                      b.reshape({1, b.shape().at(0), b.shape()[1]}));
    }

    struct EntityEmbedding {
        fr::Tensor value;
        fr::Tensor adamM;
        fr::Tensor adamSqrtV;
        fr::Tensor feature;
    };

    EntityEmbedding getEntityData(const fr::Tensor& indices) {
        auto data = entityData.read(indices).astype(poplar::FLOAT);
        auto parts = data.split(
            1u, {entityEmbeddingSize, entityEmbeddingSize, entityEmbeddingSize, entityFeatureSize});
        return {parts[0], parts[1], parts[2], parts[3]};
    }

    void setEntityData(const EntityEmbedding& entities, const fr::Tensor& indices) {
        auto data = fr::ops::concat(
                        {entities.value, entities.adamM, entities.adamSqrtV, entities.feature}, 1u)
                        .astype(dtype);
        entityData.write(data, indices);
    }

    fr::Tensor entityHiddenPredict(const EntityEmbedding& entities, const std::string& part) {
        fr::Tensor dummy;
        return entityHidden(entities, part, dummy, /*training=*/false);
    }

    fr::Tensor entityHiddenTrain(const EntityEmbedding& entities,
                                 const std::string& part,
                                 fr::Tensor& regularisationLoss) {
        return entityHidden(entities, part, regularisationLoss, /*training=*/true);
    }

    void addNormRegularisation(const NormRegularisation& regularisation,
                               const fr::Tensor& tensor,
                               fr::Tensor& loss) {
        if (loss.valid() && regularisation.weight > 0) {
            loss = loss + fr::ops::constant(regularisation.weight) *
                              fr::ops::sum(lpnorm(tensor, regularisation.power));
        }
    }

    fr::Tensor entityHidden(const EntityEmbedding& entities,
                            const std::string& part,
                            fr::Tensor& regularisationLoss,
                            bool training) {
        FeatureNetwork network;
        if (part == "head") {
            network = headFeatureNetwork;
        } else if (part == "tail") {
            network = tailFeatureNetwork;
        } else {
            assert(false && "unexpected part - expected 'head' or 'tail'");
        }
        auto featureHidden =
            fr::ops::matMul(entities.feature, gatherShards(network.featureProjection));
        if (featureMlpSize > 0) {
            auto entityConcat = fr::ops::concat({entities.value, featureHidden}, 1u);
            auto entityBoom =
                fr::nn::relu(fr::ops::matMul(entityConcat, gatherShards(network.mlpUpProjection)));
            featureHidden = fr::ops::matMul(entityBoom, gatherShards(network.mlpDownProjection));
        }
        if (training && featureDropout > 0) {
            featureHidden = fr::nn::dropout(featureHidden, featureDropout);
        }
        auto hidden = entities.value + featureHidden;
        addNormRegularisation(embeddingRegularisation, entities.value, regularisationLoss);
        addNormRegularisation(featureRegularisation, featureHidden, regularisationLoss);
        addNormRegularisation(hiddenRegularisation, hidden, regularisationLoss);
        return hidden;
    }

    fr::Tensor distance(const fr::Tensor& a, const fr::Tensor& b) {
        if (distanceFn == "L1") {
            return fr::ops::l1distance(a, b);
        } else if (distanceFn == "L1_old") {
            return l1distance(a, b);
        } else if (distanceFn == "L2") {
            return fr::ops::l2distance(a, b);
        } else if (distanceFn == "L2_old") {
            return l2distance(a, b);
        } else if (distanceFn == "MatMul") {
            return fr::ops::matMul(a, b.transpose());
        } else {
            std::ostringstream msg;
            msg << "'model.distance_fn' must be one of 'L1', 'L1_old', 'L2' or 'L2_old', not "
                << extract<std::string>(settings, "model.distance_fn") << "'";
            throw std::invalid_argument(msg.str());
        }
    }

    fr::Tensor transEPredict(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        return heads + fr::ops::gather(gatherShards(relationEmbedding), relationIndices);
    }

    fr::Tensor transEScore(const fr::Tensor& predictedTails, const fr::Tensor& tails) {
        // Note: copyToLinearTensor is a temporary workaround improve the exchange compilation
        // time into l1distance
        return fr::ops::constant(gamma) -
               distance(
                   fr::ops::copyToLinearTensor(predictedTails, std::nullopt, entityEmbeddingSize),
                   fr::ops::copyToLinearTensor(tails, std::nullopt, entityEmbeddingSize));
    }

    fr::Tensor complExPredict(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        auto dim = heads.rank() - 1;
        auto relations = fr::ops::gather(gatherShards(relationEmbedding), relationIndices);
        auto complexHeads = heads.split(dim, {entityEmbeddingSize / 2, entityEmbeddingSize / 2});
        auto complexRels =
            relations.split(dim, {relationEmbeddingSize / 2, relationEmbeddingSize / 2});
        auto rePredictedTails = complexHeads[0] * complexRels[0] - complexHeads[1] * complexRels[1];
        auto imPredictedTails = complexHeads[0] * complexRels[1] + complexHeads[1] * complexRels[0];
        return fr::ops::concat({rePredictedTails, imPredictedTails}, dim);
    }

    fr::Tensor complExScore(const fr::Tensor& predictedTails, const fr::Tensor& tails) {
        return distance(predictedTails, tails);
    }

    fr::Tensor rotatEPredict(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        auto dim = heads.rank() - 1;
        auto parts = heads.split(dim, {relationEmbeddingSize, relationEmbeddingSize});
        auto reHead = parts[0];
        auto imHead = parts[1];

        auto relations =
            fr::ops::gather(gatherShards(relationEmbedding), relationIndices).astype(poplar::HALF);
        auto sinRelation = fr::ops::sin(relations).astype(poplar::FLOAT);
        auto cosRelation = fr::ops::cos(relations).astype(poplar::FLOAT);

        auto reHeadRel = reHead * cosRelation - imHead * sinRelation;
        auto imHeadRel = imHead * cosRelation + reHead * sinRelation;
        return fr::ops::concat({reHeadRel, imHeadRel}, dim);
    }

    fr::Tensor rotatEScore(const fr::Tensor& predictedTails, const fr::Tensor& tails) {
        return fr::ops::constant(gamma) -
               distance(
                   fr::ops::copyToLinearTensor(predictedTails, std::nullopt, entityEmbeddingSize),
                   fr::ops::copyToLinearTensor(tails, std::nullopt, entityEmbeddingSize));
    }

    fr::Tensor transHPredict(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        auto normalVecs = fr::ops::gather(gatherShards(relationNormal), relationIndices);
        normalVecs = normalVecs / l2norm(normalVecs).reshape({normalVecs.shape().at(0), 1});
        // Project heads to hyperplane
        auto headsNorm = fr::ops::sum(normalVecs * heads, {heads.rank() - 1});
        auto projectedHeads = heads - normalVecs * headsNorm.reshape({headsNorm.shape().at(0), 1});
        // Predict tails
        auto predictedTails =
            projectedHeads + fr::ops::gather(gatherShards(relationEmbedding), relationIndices);
        return fr::ops::concat({normalVecs, predictedTails}, normalVecs.rank() - 1);
    }

    fr::Tensor transHScore(const fr::Tensor& normAndTails, const fr::Tensor& tails) {
        auto rank = normAndTails.rank();
        auto parts = normAndTails.split(rank - 1, {relationEmbeddingSize, relationEmbeddingSize});
        auto normalVecs = parts[0];
        auto predictedTails =
            parts[1].reshape({normAndTails.shape().at(0), 1, entityEmbeddingSize});
        // Project tails to hyperplane
        auto tailsNorm = fr::ops::matMul(normalVecs, tails.transpose());
        auto projectedTails =
            tails.reshape({1, tails.shape().at(0), entityEmbeddingSize}) -
            normalVecs.reshape({normalVecs.shape().at(0), 1, relationEmbeddingSize}) *
                tailsNorm.reshape({tailsNorm.shape().at(0), tailsNorm.shape().at(1), 1});
        return fr::ops::constant(gamma) -
               fr::ops::sum(fr::ops::abs(predictedTails - projectedTails), {2});
    }

    fr::Tensor distMultPredict(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        return heads * fr::ops::gather(gatherShards(relationEmbedding), relationIndices);
    }

    fr::Tensor distMultScore(const fr::Tensor& headTimesRel, const fr::Tensor& tails) {
        return fr::ops::matMul(headTimesRel, tails.transpose());
    }

    fr::Tensor predictTail(const fr::Tensor& heads, const fr::Tensor& relationIndices) {
        if (scoreFn == "TransE") {
            return transEPredict(heads, relationIndices);
        } else if (scoreFn == "ComplEx") {
            return complExPredict(heads, relationIndices);
        } else if (scoreFn == "RotatE") {
            return rotatEPredict(heads, relationIndices);
        } else if (scoreFn == "TransH") {
            return transHPredict(heads, relationIndices);
        } else if (scoreFn == "DistMult") {
            return distMultPredict(heads, relationIndices);
        } else {
            std::ostringstream msg;
            msg << "'model.score_fn' must be one of 'TransE', 'ComplEx', 'RotatE', 'TransH', or "
                   "'DistMult', not '"
                << extract<std::string>(settings, "model.score_fn") << "'";
            throw std::invalid_argument(msg.str());
        }
    }

    fr::Tensor score(const fr::Tensor& predictedTails, const fr::Tensor& tails) {
        if (scoreFn == "TransE") {
            return transEScore(predictedTails, tails);
        } else if (scoreFn == "ComplEx") {
            return complExScore(predictedTails, tails);
        } else if (scoreFn == "RotatE") {
            return rotatEScore(predictedTails, tails);
        } else if (scoreFn == "TransH") {
            return transHScore(predictedTails, tails);
        } else if (scoreFn == "DistMult") {
            return distMultScore(predictedTails, tails);
        } else {
            std::ostringstream msg;
            msg << "'model.score_fn' must be one of 'TransE', 'ComplEx', 'RotatE', 'TransH', or "
                   "'DistMult', not '"
                << extract<std::string>(settings, "model.score_fn") << "'";
            throw std::invalid_argument(msg.str());
        }
    }

    fr::Tensor getLoss(const fr::Tensor& scores, const fr::Tensor& tailIndices) {
        if (lossFn == "logsigmoid") {
            auto nTails = nShard * a2aSize;
            auto oneHotTails = fr::ops::oneHot(tailIndices, nTails, poplar::FLOAT);
            auto negativeWeight = (0.5f * nTails) / (nTails - 1);
            auto positiveWeight = 0.5f * nTails;
            auto weight = fr::ops::constant(positiveWeight - negativeWeight) * oneHotTails +
                          fr::ops::constant(negativeWeight);
            if (negativeAdversarialScale > 0) {
                fr::Frame f;
                auto negScoreSoftmax =
                    detachedSoftmax((fr::ops::constant(negativeAdversarialScale, poplar::FLOAT) *
                                     fr::Tensor::wrap(f.graph.wrap(f.graph.unwrap(scores.pag()),
                                                                   /*requiresGrad*/ false))) +
                                    (fr::ops::constant(-10000.0f, poplar::FLOAT) * oneHotTails));
                weight =
                    weight * (oneHotTails + fr::ops::constant(nShard * a2aSize - 1, poplar::FLOAT) *
                                                negScoreSoftmax);
            }
            auto tailMask = fr::ops::constant(2.0f) * oneHotTails - fr::ops::constant(1.0f);
            return -fr::ops::sum(weight * fr::ops::logSigmoid(tailMask * scores)) /
                   fr::ops::constant(nShard * batchSize * a2aSize * nShard, poplar::FLOAT);

        } else if (lossFn == "softmax") {
            auto correctedScores = scores;
            if (softmaxLossCorrectionWeight) {
                // The correction for noise scores includes:
                // + log(nClasses)   -- increase by size of vocabulary
                // - log(nSamples)   -- decrease according to sampling probability (flat sampling)
                float correction = std::log(nEntity * nShard - 1) - std::log(nShard * a2aSize - 1);
                correctedScores =
                    correctedScores +
                    fr::ops::constant(softmaxLossCorrectionWeight * correction) *
                        (fr::ops::constant(1.0f) -
                         fr::ops::oneHot(tailIndices, nShard * a2aSize, poplar::FLOAT));
            }
            return fr::ops::sum(fr::nn::softmaxCrossEntropy(correctedScores, tailIndices)) /
                   fr::ops::constant(nShard * batchSize, poplar::FLOAT);

        } else {
            std::ostringstream msg;
            msg << "'training.loss.type' must be one of 'logsigmoid' or 'softmax', not '" << lossFn
                << "'";
            throw std::invalid_argument(msg.str());
        }
    }

    // Programs

    fr::Tensor trainStep(const fr::Tensor& learningRate,
                         const fr::Tensor& remoteIndices,
                         const fr::Tensor& a2aIndices,
                         const fr::Tensor& headIndices,
                         const fr::Tensor& relationIndices,
                         const fr::Tensor& tailIndices) {
        // 1. Gather entity embeddings and combine features
        auto entities = getEntityData(remoteIndices);
        entities.value = fr::ops::startGrad(entities.value);
        auto regularisationLoss = fr::ops::constant(0.0f);

        // 2. Redistribute negative samples
        auto heads =
            fr::ops::gather(entityHiddenTrain(entities, "head", regularisationLoss), headIndices);
        auto tails = fr::ops::gather(entityHiddenTrain(entities, "tail", regularisationLoss),
                                     a2aIndices.reshape({nShard * a2aSize}));
        tails = fr::ops::allToAll(tails.reshape({nShard, a2aSize, entityEmbeddingSize}))
                    .reshape({nShard * a2aSize, entityEmbeddingSize});

        // 3. Compute scores
        auto predictedTails = predictTail(heads, relationIndices);
        auto scores = score(predictedTails, tails);

        // 4. Compute loss
        auto loss = getLoss(scores, tailIndices) + regularisationLoss;

        // 5. Compute update
        loss.backward(fr::ops::constant(lossScale));
        auto stepSize = fr::nn::adamStepSizeAutoIncrement(
            fr::ops::variable("step", {{}, poplar::UNSIGNED_INT}), learningRate, adamParams);
        updateParameters(stepSize);
        {
            fr::Frame f;
            popops::squareInPlace(f.graph.poplar(), f.graph.unwrap(entities.adamSqrtV.pag()),
                                  f.tape.prog(), f.di);
            fr::nn::adam(entities.value, entities.adamM, entities.adamSqrtV,
                         modifiedStepSize(stepSize, "entity_embedding"), adamParams);
            popops::sqrtInPlace(f.graph.poplar(), f.graph.unwrap(entities.adamSqrtV.pag()),
                                f.tape.prog(), f.di);
        }
        setEntityData(entities, remoteIndices);

        return loss;
    }

    void trainStepLoop() {
        auto nStep = trainStepsPerProgramRun;

        // Input streams
        auto learningRate = fr::ops::input("learning_rate", {{}, poplar::FLOAT});
        auto remoteIndices =
            fr::ops::input("remote", {{nStep, batchSize + nShard * a2aSize}, poplar::UNSIGNED_INT});
        auto a2aIndices = fr::ops::input("a2a", {{nStep, nShard, a2aSize}, poplar::UNSIGNED_INT});
        auto headIndices = fr::ops::input("head", {{nStep, batchSize}, poplar::UNSIGNED_INT});
        auto relationIndices =
            fr::ops::input("relation", {{nStep, batchSize}, poplar::UNSIGNED_INT});
        auto tailIndices = fr::ops::input("tail", {{nStep, batchSize}, poplar::UNSIGNED_INT});

        // Training step loop
        auto totalLoss = fr::Tensor::declare({{}, poplar::FLOAT}, false, "total_loss");
        fr::mapping::setDefault(fr::mapping::OneTile(), {totalLoss});
        {
            fr::Frame f("total_loss");
            popops::zero(f.graph.poplar(), f.graph.unwrap(totalLoss.pag()), f.tape.prog(), f.di);
        }
        fr::ops::forN(nStep, [&](const fr::Tensor& index) {
            fr::Frame f;
            auto stepLoss =
                trainStep(learningRate, switchedSlice(remoteIndices, index),
                          switchedSlice(a2aIndices, index), switchedSlice(headIndices, index),
                          switchedSlice(relationIndices, index), switchedSlice(tailIndices, index));
            popops::addInPlace(f.graph.poplar(), f.graph.unwrap(totalLoss.pag()),
                               f.graph.unwrap(stepLoss.pag()), f.tape.prog(), f.di);
        });

        // Output streams
        fr::ops::output("loss", totalLoss / fr::ops::constant<float>(nStep));
    }

    struct TopKCollector {
        static constexpr float BadScore = -1e6f;

        unsigned hrBatchSize;
        unsigned tailBatchSize;
        unsigned nBest;

        TopKCollector(unsigned hrBatchSize, unsigned tailBatchSize, unsigned nBest)
            : hrBatchSize(hrBatchSize), tailBatchSize(tailBatchSize), nBest(nBest) {
            fr::Frame f("TopKCollector");
            m_mergedScores = f.graph.poplar().addVariable(
                poplar::FLOAT, {hrBatchSize, nBest + tailBatchSize},
                poplar::VariableMappingMethod::LINEAR, {f.di, "mergedScores"});
            popops::fill(f.graph.poplar(), m_mergedScores, f.tape.prog(), BadScore, f.di);

            m_bestIndices = f.graph.poplar().addVariable(poplar::UNSIGNED_INT, {hrBatchSize, nBest},
                                                         poplar::VariableMappingMethod::LINEAR,
                                                         {f.di, "bestIndices"});
            popops::zero(f.graph.poplar(), m_bestIndices, f.tape.prog(), f.di);
        }

        void add(const fr::Tensor& scores, const fr::Tensor& indices) {
            fr::Frame f("TopKCollector::add");

            f.tape.prog().add(
                poplar::program::Copy(f.graph.unwrap(scores.pag()),
                                      m_mergedScores.slice({nBest, nBest + tailBatchSize}, 1u),
                                      /*dontOutline*/ false, f.di));

            auto ki = popops::topKWithPermutation(
                f.graph.poplar(), f.tape.prog(), m_mergedScores,
                {nBest, /*largest*/ true, popops::SortOrder::NONE}, f.di);

            f.tape.prog().add(poplar::program::Copy(ki.first, m_mergedScores.slice({0, nBest}, 1u),
                                                    /*dontOutline*/ false, f.di));

            auto mergedIndices = poplar::concat(
                {m_bestIndices,
                 f.graph.unwrap(indices.pag()).expand({0u}).broadcast(hrBatchSize, 0u)},
                1u);
            auto plan = popops::embedding::plan(f.graph.poplar(), poplar::UNSIGNED_INT,
                                                /*groupSize*/ hrBatchSize,
                                                /*numEntries*/ nBest + tailBatchSize,
                                                /*outputSize*/ 1, /*numLookups*/ {nBest}, {});
            auto newBestIndices =
                popops::groupedMultiSlice(f.graph.poplar(), mergedIndices.expand({2u}),
                                          ki.second.expand({2u}), {0u}, {1u}, f.tape.prog(), plan,
                                          {}, f.di)
                    .squeeze({2u, 3u});
            f.tape.prog().add(poplar::program::Copy(newBestIndices, m_bestIndices,
                                                    /*dontOutline*/ false, f.di));
        }

        fr::Tensor bestScores() const {
            auto& f = fr::Environment::frame();
            return fr::Tensor::wrap(
                f.graph.wrap(m_mergedScores.slice({0, nBest}, 1u), /*requiresGrad*/ false));
        }

        fr::Tensor bestIndices() const {
            auto& f = fr::Environment::frame();
            return fr::Tensor::wrap(f.graph.wrap(m_bestIndices, /*requiresGrad*/ false));
        }

       private:
        poplar::Tensor m_mergedScores;
        poplar::Tensor m_bestIndices;
    };

    void predict() {
        // 0. Input streams
        auto headIndices =
            fr::ops::input("predict_head", {{predictHrBatchSize}, poplar::UNSIGNED_INT});
        auto relationIndices =
            fr::ops::input("predict_relation", {{predictHrBatchSize}, poplar::UNSIGNED_INT});
        // Note: entityCount should be the number of "real" entities (not including padding=#0)
        auto entityCount = fr::ops::input("predict_entity_count", {{}, poplar::UNSIGNED_INT});

        // 1. Head computation
        const auto allHrBatchSize = nShard * predictHrBatchSize;
        const auto predictedTailSize =
            (scoreFn == "TransH") ? entityEmbeddingSize * 2 : entityEmbeddingSize;
        auto predictedTails =
            fr::ops::allGather(predictTail(entityHiddenPredict(getEntityData(headIndices), "head"),
                                           relationIndices))
                .reshape({allHrBatchSize, predictedTailSize});

        // 2. Tail scoring loop
        TopKCollector collector(allHrBatchSize, predictTailBatchSize, predictNBest);
        auto baseIndexRange =
            fr::ops::constant(fr::util::arange<unsigned>(1, 1 + predictTailBatchSize));
        auto nLoop = (nEntity - 1 + predictTailBatchSize - 1) / predictTailBatchSize;
        fr::ops::forN(nLoop, [&](const fr::Tensor& index) {
            auto tailIndices = baseIndexRange + fr::ops::constant(predictTailBatchSize) * index;
            auto tailMask = tailIndices < (entityCount + fr::ops::constant(1u));
            auto tails = entityHiddenPredict(
                getEntityData(tailIndices * tailMask.astype(poplar::UNSIGNED_INT)), "tail");
            auto scores =
                score(predictedTails, tails) +
                (fr::ops::constant(TopKCollector::BadScore) * (~tailMask).astype(poplar::FLOAT));
            collector.add(scores, tailIndices);
        });

        // 3. Swap results back to the original shard
        auto bestScores = fr::ops::allToAll(
            collector.bestScores().reshape({nShard, predictHrBatchSize, predictNBest}));
        auto bestIndices = fr::ops::allToAll(
            collector.bestIndices().reshape({nShard, predictHrBatchSize, predictNBest}));

        // 3. Output streams
        fr::ops::output("predict_scores", bestScores);
        fr::ops::output("predict_indices", bestIndices);
    }

    void readEntity() {
        auto indices = fr::ops::input("read_indices", {{rwBatchSize}, poplar::UNSIGNED_INT});
        auto data = entityData.read(indices);
        fr::ops::output("read_data", data);
    }

    /* N.B. also zeros the adam moments */
    void writeEntity() {
        auto embeddings = fr::ops::randomNormal(0.0f, initScale / entityEmbeddingSize,
                                                {rwBatchSize, entityEmbeddingSize}, seed, dtype);
        auto adamMoments = fr::ops::full({{rwBatchSize, 2 * entityEmbeddingSize}}, 0.0f, dtype);
        auto features = fr::ops::input("write_features", {{rwBatchSize, entityFeatureSize}, dtype});
        auto indices = fr::ops::input("write_indices", {{rwBatchSize}, poplar::UNSIGNED_INT});
        entityData.write(fr::ops::concat({embeddings, adamMoments, features}, 1u), indices);
    }
};

std::vector<size_t> extendReplicaShape(std::vector<size_t> shape) {
    shape.insert(shape.begin(), fr::Environment::frame().replicationFactor());
    return shape;
}

struct Program {
    std::string name;
    poplar::program::Sequence prog;
    // includes any replicas as a leading axis
    std::unordered_map<std::string, fr::Tensor::Spec> streams;

    static Program build(const std::string& name, std::function<void()> func) {
        fr::SubProgramFrame frame(name);
        // At time of writing there are still some (non-breaking) exceptions generated
        // so we are disabling these checks temporarily
        // poplar::setFloatingPointBehaviour(
        //     frame.graph.poplar(), frame.tape.prog(),
        //     {/*inv*/ true, /*div*/ true, /*oflo*/ true, /*esr*/ false, /*nanoo*/ false},
        //     frame.di);
        func();
        std::unordered_map<std::string, fr::Tensor::Spec> streams;
        for (auto& item : frame.streams) {
            streams[item.first] = {extendReplicaShape(item.second.spec().shape),
                                   item.second.spec().dtype};
        }
        return {name, frame.tape.prog(), streams};
    }
};

}  // namespace

fr::Tensor detachedSoftmax(const fr::Tensor& tensor) {
    fr::Frame f("poplar_kge::detachedSoftmax");
    fr::mapping::setDefault(fr::mapping::Linear(), {tensor});
    auto poplarTensor = f.graph.unwrap(tensor.pag());
    auto result = popnn::nonLinearity(f.graph.poplar(), popnn::NonLinearityType::SOFTMAX_STABLE,
                                      poplarTensor, f.tape.prog(), f.di);
    return fr::Tensor::wrap(f.graph.wrap(result, /*requiresGrad*/ false));
}

struct EngineImpl {
    poplar::Engine engine;
    std::unordered_map<std::string, fr::Tensor::Spec> variables;
    std::vector<Program> programs;

    static std::unique_ptr<EngineImpl> create(const Batch& settings, const std::string& gpFolder) {
        auto device = attach(settings);
        fr::RootFrame frame(device.getTarget());
        popops::addCodelets(frame.graph.poplar());
        poplin::addCodelets(frame.graph.poplar());
        poprand::addCodelets(frame.graph.poplar());
        frame.graph.poplar().addCodelets(gpFolder + "/poplar_extensions.gp");

        Model model(settings);
        std::vector<Program> programs(
            // Order is important here - training first, as the backward() call may pollute things
            {Program::build("train_step_loop", std::bind(&Model::trainStepLoop, model)),
             Program::build("predict", std::bind(&Model::predict, model)),
             Program::build("read_entity", std::bind(&Model::readEntity, model)),
             Program::build("write_entity", std::bind(&Model::writeEntity, model))});
        auto variables = model.finaliseVariables();
        auto poplarPrograms =
            fr::util::mapVector(programs, [](auto& p) { return poplar::program::Program(p.prog); });

        // Only set target.extendedMemory when we have to (buffer >= 16 GiB), to
        // keep support for legacy systems
        poplar::OptionFlags engineOptions;
        if (model.entityData.totalBytes(device.getTarget()) >= 16 * (size_t(1) << 30)) {
            engineOptions.set("target.extendedMemory", "true");
        }

        poplar::Engine engine(frame.graph.poplar(), poplarPrograms, engineOptions);
        engine.load(device);
        return std::unique_ptr<EngineImpl>(new EngineImpl{std::move(engine), variables, programs});
    }

    Batch run(const std::string& command, Batch& data) {
        for (auto i = 0u; i < programs.size(); ++i) {
            if (command == programs[i].name) {
                return runProgram(i, data);
            }
        }
        if (command == "read" || command == "write") {
            return runReadWrite(command, data);
        } else if (command == "variables") {
            return runVariables();
        } else {
            std::ostringstream msg;
            msg << "Unknown command '" << command << "'";
            throw std::invalid_argument(msg.str());
        }
    }

    Batch runProgram(unsigned index, Batch& data) {
        for (const auto& stream : programs[index].streams) {
            auto ptr = GetData<float, uint32_t, float16>::value(stream.first, stream.second,
                                                                get(data, stream.first));
            engine.connectStream(stream.first, std::get<0>(ptr));
        }
        engine.run(index);
        return {};
    }

    Batch runReadWrite(const std::string& command, Batch& data) {
        engine.disableExecutionProfiling();
        auto name = extract<std::string>(data, "name");
        auto spec = get(variables, name);
        auto ptr = GetData<float, uint32_t, float16>::value(name, spec, get(data, "value"));
        if (command == "read") {
            engine.readTensor(name, std::get<0>(ptr), std::get<1>(ptr));
        } else if (command == "write") {
            engine.writeTensor(name, std::get<0>(ptr), std::get<1>(ptr));
        } else {
            assert(false && "bad command");
        }
        engine.enableExecutionProfiling();
        return {};
    }

    Batch runVariables() {
        return {{"variables", fr::util::mapVector(variables, [](auto& entry) {
                     return std::tuple<std::string, std::vector<unsigned>>{
                         entry.first, {entry.second.shape.begin(), entry.second.shape.end()}};
                 })}};
    }
};

Engine::Engine(const Batch& batch, const std::string& gpFolder)
    : m_impl(EngineImpl::create(batch, gpFolder)) {}
Engine::~Engine() = default;
Batch Engine::run(const std::string& command, Batch& data) {
    return m_impl->run(command, data);
}

}  // namespace poplar_kge
