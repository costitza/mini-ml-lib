#pragma once
#include "MLModel.h"
#include "Hyperparameters.h"
#include <string>

class ModelFactory {
public:

    // fresh made
    static MLModel* createModel(int type, const std::string& name, const Hyperparameters& hp);
    
    // loaded from json
    static MLModel* createModel(const std::string& modelType, const std::string& name, const Hyperparameters& hp);
};