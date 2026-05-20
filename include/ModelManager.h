#pragma once
#include <vector>
#include <string>
#include <memory>
#include "MLModel.h"

class ModelManager {
private:
    std::vector<MLModel*> models;

public:
    ModelManager() = default;
    ~ModelManager();

    // delete copy and assignment
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    void addModel(MLModel* model);
    MLModel* getModelByID(const std::string& targetID) const;
    const std::vector<MLModel*>& getModels() const;
    bool isEmpty() const;
    void clear();
};
