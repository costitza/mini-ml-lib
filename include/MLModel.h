#pragma once
#include <string>
#include "Hyperparameters.h"
#include "Dataset.h"
#include <iostream>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann :: json;


class MLModel{
protected:
    std :: string name;
    bool isTrained;
    Hyperparameters params;

    // static variables + id
    static int totalModels;
    std :: string modelID;

public:
    MLModel() {}
    MLModel(std::string modelName, const Hyperparameters& hp);

    virtual ~MLModel();

    // methods
    virtual void train(const Dataset& data) = 0;
    virtual double predict(const Eigen :: VectorXd& input) const = 0;

    // save / load to json
    virtual json serialize() const = 0;
    virtual void deserialize(const json& j) = 0;

    virtual void print(std :: ostream& os) const;
    friend std :: ostream& operator<<(std :: ostream& os, const MLModel& model);

    // getters / setters
    std :: string getName() const;

    std :: string getModelID() const;

    bool getIsTrained() const;

    Hyperparameters getHyperparameters() const;
    void setIsTrained(const bool a);

    void setHyperparameters(const Hyperparameters& hp) { params = hp; }

    // static methods
    static int getTotalModels();
    static std :: string generateRandomID();
};