#include "LogicRModel.h"
#include "Exceptions.h"
#include <iostream>

LogicRModel :: LogicRModel(std::string modelName, const Hyperparameters& hp, int classes, double threshold, double b)
    : MLModel(modelName, hp), Classifier(modelName, hp, classes, threshold), bias(b) {

    weights = Eigen :: VectorXd :: Zero(hp.getInputFeatures());
}

double LogicRModel :: getBias() const {
    return bias;
}

Eigen :: VectorXd LogicRModel :: getWeights() const{
    return weights;
}


// different from linear regression, this method uses gradient descent
void LogicRModel :: train(const Dataset& data) {
    int n = data.getRows();
    int f = data.getCols();

    this -> notifyObservers("Started training on dataset with " + std::to_string(data.getRows()) + " rows.");

    if (weights.size() != f) {
        std::cout << "  -> [Notice] Auto-resizing model from " 
                  << weights.size() << " to " << f << " features to match data.\n";
        weights = Eigen::VectorXd::Zero(f);
        
        // Update the hyperparameters to reflect reality
        Hyperparameters currentHp = this->getHyperparameters();
        currentHp.setInputFeatures(f);
        this->setHyperparameters(currentHp);
        
        bias = 0.0; 
        this->setIsTrained(false);
    }

    // extract data to eigen :: matrix
    Eigen :: MatrixXd X(n, f);
    Eigen :: VectorXd Y(n);

    // save labels and values of matrix
    for (int i = 0; i < n; i++){
        X.row(i) = data.getRowsAsEigen(i);
        Y(i) = data.getLabel(i);
    }

    int epochs = this -> getHyperparameters().getEpochs();
    double lr = this -> getHyperparameters().getLearningRate();

    for (int epoch = 0; epoch < epochs; epoch ++){
        // Z = X*w + b;
        Eigen :: VectorXd Z = (X * weights).array() + bias;


        // unary expression from eigen => function we define to every elemnt of vector
        Eigen :: VectorXd A = Z.unaryExpr([](double z){
            return LogicRModel :: sigmoid(z); // used static methods defined in header
        });

        // error
        Eigen :: VectorXd dZ = A - Y;

        Eigen :: VectorXd dW = (1.0 / n) * X.transpose() * dZ;
        double db = dZ.sum() / n;

        // update weights and bias 
        weights -= lr * dW;
        bias -= lr * db;

        // gotta love c++
        // all this training could be just:
        // forward + calc loss + back propagation in pytorch :(
    }

    this -> notifyObservers("Finished training! Final Bias: " + std::to_string(bias));

    this -> setIsTrained(true);
}


double LogicRModel :: predict(const Eigen :: VectorXd& input) const{

    if (input.size() != weights.size()) {
        throw DimensionMismatchException(weights.size(), input.size());
    }

    // same as linear regression
    // calculate z and then squash value between 0 and 1

    double z = input.dot(weights) + bias;

    // sigmoid function
    double probability = LogicRModel :: sigmoid(z);
    return (probability >= this -> getDecisionThreshold()) ? 1.0 : 0.0;
}


void LogicRModel::print(std::ostream& os) const {
    MLModel::print(os);
    os << " | Type: Logistic Regression | Threshold: " << this->getDecisionThreshold() 
       << " | Bias: " << bias;
}


json LogicRModel::serialize() const {
    json j;
    j["model_type"] = "LogicRModel";
    j["name"] = this->getName();
    j["bias"] = bias;
    j["decisionThreshold"] = this->getDecisionThreshold();

    j["hyperparameters"] = this->getHyperparameters().serialize();

    // convert Eigen::VectorXd to std::vector
    std::vector<double> w_vec(weights.data(), weights.data() + weights.size());
    j["weights"] = w_vec;
    
    return j;
}

void LogicRModel::deserialize(const json& j) {
    name = j.value("name", "modelLoaded");
    bias = j.value("bias", 0.0);
    
    if (j.contains("weights")) {
        std :: vector<double> w_vec = j["weights"];
        weights = Eigen :: Map<Eigen :: VectorXd>(w_vec.data(), w_vec.size());
        this -> setIsTrained(true);
    } else {
        std :: cout << "[Warning] No weights found in this JSON file!\n";
        this -> setIsTrained(false);
    }
}
