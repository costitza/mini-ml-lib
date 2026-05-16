#include "LinearRModel.h"
#include <vector>
#include <iostream>
#include "Exceptions.h"

LinearRModel :: LinearRModel(std::string modelName, const Hyperparameters& hp, double l2, double b)
    : MLModel(modelName, hp), Regressor(modelName, hp, l2), bias(b) {
    weights = Eigen :: VectorXd :: Zero(hp.getInputFeatures());
}

// Normal Equation
void LinearRModel :: train(const Dataset& data){
    int n = data.getRows();
    int f = data.getCols();

    // f + 1 to calculate weigths + bias at the same time
    Eigen :: MatrixXd X(n, f + 1);
    Eigen :: VectorXd Y(n);

    for (int i = 0;i < n;i++){
        // get row as eigen vector
        Eigen :: VectorXd row = data.getRowsAsEigen(i);

        for (int j = 0; j < f; j++){
            X(i, j) = row(j);
        }
        // bias column
        X(i, f) = 1.0;

        Y(i) = data.getLabel(i);
    }

    // regularization matrix (identity * l2penalty)
    Eigen :: MatrixXd I = Eigen :: MatrixXd :: Identity(f + 1, f + 1) * this->getL2Penalty();
    I(f, f) = 0.0;


    // w = (Xt * X + I) ^ (-1) * Xt * Y
    Eigen :: MatrixXd XT = X.transpose();
    Eigen :: MatrixXd parantheses = XT * X + I;

    Eigen :: VectorXd weights_bias = parantheses.colPivHouseholderQr().solve(XT * Y);

    weights = weights_bias.head(f); // first f elements
    bias = weights_bias(f);

    Hyperparameters currentHp = this->getHyperparameters();
    if (currentHp.getInputFeatures() != f) {
        currentHp.setInputFeatures(f);
        this->setHyperparameters(currentHp);
    }
    
    this->setIsTrained(true);
}


double LinearRModel :: predict(const Eigen :: VectorXd& input) const{
    if (input.size() != weights.size()) {
        throw DimensionMismatchException(weights.size(), input.size());
    }

    // dot product y = xw + b
    return input.dot(weights) + bias;
}


void LinearRModel :: print(std :: ostream& os) const{
    MLModel::print(os); // print base info
    os << " | Type: Linear Regression | Bias: " << bias;
}


json LinearRModel :: serialize() const {
    json j;
    j["model_type"] = "LinearRModel";
    j["name"] = this -> getName();
    j["bias"] = bias;
    j["l2Penalty"] = l2Penalty;

    j["hyperparameters"] = this->getHyperparameters().serialize();

    // convert to normal vector the weights
    std :: vector<double> w_vec(weights.data(), weights.data() + weights.size());
    j["weights"] = w_vec;

    return j;
}


void LinearRModel :: deserialize(const json& j){
    name = j.value("name", "modelLoaded");
    bias = j.value("bias", 0.0);
    l2Penalty = j.value("l2Penalty", 0.0);

    if (j.contains("weights") && !j["weights"].is_null()) {
        std::vector<double> w_vec = j["weights"];
        weights = Eigen::Map<Eigen::VectorXd>(w_vec.data(), w_vec.size());
        this->setIsTrained(true);
    } else {
        std::cout << "  -> [Notice] Loaded an untrained model.\n";
        this->setIsTrained(false);
    }
}


Eigen :: VectorXd LinearRModel :: getWeights() const{
    return weights;
}

double LinearRModel :: getBias() const{
    return bias;
}

void LinearRModel :: setL2Penalty(double p){
    l2Penalty = p;
}
