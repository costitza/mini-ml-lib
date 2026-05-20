#pragma once

#include "Classifier.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// model for logistic regression derived from Classifier
// the predict method will return the sigmoid of the dot product of the input and the weights 
// (very similar to linear regression but with a sigmoid activation function, i.e. the output will be between 0 and 1)

class LogicRModel : public Classifier{
protected:
    Eigen :: VectorXd weights;
    double bias;

    // static method for squashing prediction between 0 and 1
    static double sigmoid(double z){
        return 1.0 / (1.0 + std :: exp(-z));
    }

public:
    LogicRModel() : Classifier(), bias(0.0) {}
    LogicRModel(std::string modelName, const Hyperparameters& hp, int classes = 2, double threshold = 0.5, double b = 0.0);

    virtual ~LogicRModel() override = default;

    // methods
    void train(const Dataset& data) override;
    double predict(const Eigen :: VectorXd& input) const override;

    void print(std :: ostream& os) const override;

    // save / load to json
    json serialize() const override;
    void deserialize(const json& j) override;

    // getters / setters
    Eigen :: VectorXd getWeights() const;

    double getBias() const;
    void setDecisionThreshold(double t) { decisionThreshold = t; } 

};
