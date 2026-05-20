#include "KNNModel.h"
#include "Exceptions.h"

// constr
KNNModel::KNNModel(std::string modelName, const Hyperparameters& hp, int k, bool isClass)
    : MLModel(modelName, hp), 
      Classifier(modelName, hp, 2, 0.5), 
      Regressor(modelName, hp, 0.0), 
      kNeighbors(k), 
      isClassification(isClass), 
      savedData(nullptr) {}

// copy constructor
KNNModel::KNNModel(const KNNModel& other) 
    : MLModel(other), Classifier(other), Regressor(other),
      kNeighbors(other.kNeighbors), isClassification(other.isClassification) {
    
    if (other.savedData != nullptr) {
        // dataset copy constructor
        this->savedData = new Dataset(*(other.savedData));
    } else {
        this->savedData = nullptr;
    }
}


KNNModel& KNNModel::operator=(const KNNModel& other) {
    if (this == &other) return *this;

    // clean up old memory
    delete savedData;

    this->kNeighbors = other.kNeighbors;
    this->isClassification = other.isClassification;

    // deep copy the dataset
    if (other.savedData != nullptr) {
        this->savedData = new Dataset(*(other.savedData));
    } else {
        this->savedData = nullptr;
    }

    return *this;
}

KNNModel::~KNNModel() {
    // for memory leaks
    delete savedData; 
}


// train + predict methods
// for training we just save the dataset => lazy learner
void KNNModel :: train(const Dataset& data) {
    this -> notifyObservers("Started training (lazy loading) with " + std::to_string(data.getRows()) + " points.");
    delete savedData;

    // using copy constructor for data
    savedData = new Dataset(data);

    this -> setIsTrained(true);
    this -> notifyObservers("Finished training! Model is now ready for prediction.");
}


double KNNModel::predict(const Eigen::VectorXd& input) const {
    if (!this -> getIsTrained() || savedData == nullptr) {
        throw NotTrainedException(this -> getName());
    }

    int n = savedData -> getRows();
    
    // store pairs of <Distance, Label>
    std::vector<std::pair<double, double>> distances;
    distances.reserve(n);

    // calculate the euclidean distance between the input and every row in the saved data
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd row = savedData->getRowsAsEigen(i);

        // eigen norm() calculates euclidian distance
        double dist = (row - input).norm(); 
        distances.push_back({dist, savedData->getLabel(i)});
    }

    // sort the distances from closest to furthest
    std::sort(distances.begin(), distances.end());

    // make sure dont ask for more neighbors then data points
    int k = std::min(kNeighbors, n);

    // make the Prediction based on the mode
    if (isClassification) {
        // CLASSIFICATION: Majority Vote
        std::map<int, int> classVotes;
        int bestClass = -1;
        int maxVotes = -1;

        for (int i = 0; i < k; ++i) {
            int label = static_cast<int>(distances[i].second);


            classVotes[label]++;
            
            if (classVotes[label] > maxVotes) {
                maxVotes = classVotes[label];
                bestClass = label;
            }
        }
        return static_cast<double>(bestClass);
        
    } else {
        // REGRESSION: Mean Average
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            sum += distances[i].second;
        }
        return sum / k;
    }
}


void KNNModel::print(std::ostream& os) const {
    MLModel::print(os);
    os << " | Type: K-Nearest Neighbors | K: " << kNeighbors 
       << " | Mode: " << (isClassification ? "Classification" : "Regression");
}


json KNNModel::serialize() const {
    json j;
    j["model_type"] = "KNNModel";
    j["name"] = this->getName();
    j["kNeighbors"] = kNeighbors;
    j["isClassification"] = isClassification;

    j["hyperparameters"] = this->getHyperparameters().serialize();
    
    // serializing an entire raw dataset into JSON can create massive files
    // so this is a nono
    
    return j;
}

void KNNModel::deserialize(const json& j) {
    
    name = j.value("name", "modelLoaded");
    kNeighbors = j.value("kNeighbors", 0);
    isClassification = j.value("isClassification", false);


    // a real deserialization would require loading the dataset back in
}

int KNNModel::getKNeighbors() const {
    return kNeighbors; 
}


bool KNNModel::getIsClassification() const {
    return isClassification; 
}

void KNNModel :: setK(int newk){
    kNeighbors = newk;
}