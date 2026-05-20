#include "ModelManager.h"

ModelManager::~ModelManager() {
    clear();
}

void ModelManager::addModel(MLModel* model) {
    if (model) {
        models.push_back(model);
    }
}

MLModel* ModelManager::getModelByID(const std::string& targetID) const {
    for (MLModel* model : models) {
        if (model->getModelID() == targetID) {
            return model;
        }
    }
    return nullptr;
}

const std::vector<MLModel*>& ModelManager::getModels() const {
    return models;
}

bool ModelManager::isEmpty() const {
    return models.empty();
}

void ModelManager::clear() {
    for (MLModel* model : models) {
        delete model;
    }
    models.clear();
}
