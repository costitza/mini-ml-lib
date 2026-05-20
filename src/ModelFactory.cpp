#include "ModelFactory.h"
#include "LinearRModel.h"
#include "LogicRModel.h"
#include "KNNModel.h"

MLModel* ModelFactory::createModel(int type, const std::string& name, const Hyperparameters& hp) {
    if (type == 1) return new LinearRModel(name, hp, 0.0);
    if (type == 2) return new LogicRModel(name, hp, 2, 0.5);
    if (type == 3) return new KNNModel(name, hp, 3, true);
    return nullptr;
}

MLModel* ModelFactory::createModel(const std::string& modelType, const std::string& name, const Hyperparameters& hp) {
    if (modelType == "LinearRModel") return new LinearRModel(name, hp, 0.0);
    if (modelType == "LogicRModel")  return new LogicRModel(name, hp, 2, 0.5);
    if (modelType == "KNNModel")     return new KNNModel(name, hp, 3, true);
    return nullptr;
}