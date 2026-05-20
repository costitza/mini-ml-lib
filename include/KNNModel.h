#pragma once

#include "Classifier.h"
#include "Regressor.h"
#include "Dataset.h"
#include <iostream>

// model for KNN derived from both Classifier and Regressor
// the predict method will return the majority class for classification and the mean of the k nearest neighbors for regression


// the train method will simply save the training data and the value of k 
// and whether it is a classification or regression model, since KNN is a lazy learner and does not have a training phase 
// in the traditional sense


// the confusion matrix will be calculated based on the predictions and the true labels of the training data for classification
// the MSE will be calculated based on the predictions and the true labels of the training data for regression


class KNNModel : public Classifier, public Regressor{
    int kNeighbors;
    bool isClassification;

    Dataset* savedData;

public:
    KNNModel() : Classifier(), Regressor(), kNeighbors(3), isClassification(true), savedData(nullptr) {}
    KNNModel(std::string modelName, const Hyperparameters& hp, int k = 3, bool isClass = true);

    // copy constructor + deep copy
    KNNModel(const KNNModel& other);
    KNNModel& operator=(const KNNModel& other);

    ~KNNModel() override;

    // override methods
    void train(const Dataset& data) override;
    double predict(const Eigen :: VectorXd& input) const override;

    void print(std :: ostream& os) const override;

    // json serialization
    json serialize() const override;
    void deserialize(const json& j) override;

    // getters / setters
    int getKNeighbors() const;
    bool getIsClassification() const;
    void setK(int newk);
};