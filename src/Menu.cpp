#include "Menu.h"
#include "LinearRModel.h"
#include "LogicRModel.h"
#include "KNNModel.h"
#include "Dataset.h"
#include "Hyperparameters.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>
#include "DataLoader.h"

namespace fs = std :: filesystem;

// private constr
Menu :: Menu() : isRunning(true) {}

Menu :: ~Menu() {
    for (MLModel* model : models){
        delete model;
    }
    models.clear();
}

Menu& Menu :: getInstance(){
    static Menu instance;
    return instance;
}


void Menu::printHeader() const {
    std::cout << "\n=========================================\n";
    std::cout << "      Machine Learning Library C++       \n";
    std::cout << "          Total Models: " << MLModel::getTotalModels() << "\n";
    std::cout << "=========================================\n";
    std::cout << "1. Create a New Model\n";
    std::cout << "2. Train a Model (Dummy Data)\n";
    std::cout << "3. Modify Model Parameters\n";
    std::cout << "4. List Active Models\n";
    std::cout << "5. Save a Model to File\n";
    std::cout << "6. Load a Model from File\n";
    std::cout << "7. Exit\n";
    std::cout << "Choose an option: ";
}


void Menu::pause() const {
    std::cout << "\nPress Enter to return to the menu...";
    // throw away any leftover characters (like the '\n' from previous inputs)
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 
}


void Menu::run() {
    int choice;
    while (isRunning) {
        printHeader();
        
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "\n[Error] Invalid input. Please enter a number.\n";
            continue;
        }

        switch (choice) {
            case 1: createModel(); break;
            case 2: trainModel(); break;
            case 3: modifyModel(); break;
            case 4: listModels(); break;
            case 5: saveModel(); break;
            case 6: loadModel(); break; 
            case 7: 
                std::cout << "\nExiting program. Cleaning up memory...\n";
                isRunning = false; 
                break;
            default:
                std::cout << "\n[Error] Invalid choice.\n";
                break;
        }

        if(isRunning){
            pause();
        }
    }
}

MLModel* Menu :: getModelByID(std :: string targetID){
    MLModel* modelToSave = nullptr;
    for (MLModel* model : models) {
        if (model->getModelID() == targetID) {
            modelToSave = model;
            break;
        }
    }
    return modelToSave;
}


void Menu::createModel() {
    std::cout << "\n--- Create Model ---\n";
    std::cout << "1. Linear Regression\n";
    std::cout << "2. Logistic Regression (Classifier)\n";
    std::cout << "3. K-Nearest Neighbors\n";
    std::cout << "Select type: ";
    
    int type;
    std::cin >> type;

    std::string name;
    std::cout << "Enter a name for the model: ";
    std::cin >> name;

    // Create default hyperparameters for now (5 features, lr=0.01, 100 epochs)
    Hyperparameters hp(5, 0.01, 100);

    MLModel* newModel = nullptr;

    if (type == 1) {
        newModel = new LinearRModel(name, hp, 0.0);
    } else if (type == 2) {
        newModel = new LogicRModel(name, hp, 2, 0.5);
    } else if (type == 3) {
        newModel = new KNNModel(name, hp, 3, true);
    } else {
        std::cout << "[Error] Unknown model type.\n";
        return;
    }

    models.push_back(newModel);
    std::cout << "[Success] Created " << newModel->getName() 
              << " with ID: " << newModel->getModelID() << "\n";
}


void Menu::trainModel() {
    if (models.empty()) {
        std::cout << "\n[Error] No models available to train.\n";
        return;
    }

    listModels(); // Show the user the available IDs
    std::cout << "\nEnter the exact ID of the model to train (e.g., MOD-A3F1): ";
    
    std::string targetID;
    std::cin >> targetID;

    MLModel* modelToTrain = getModelByID(targetID);

    if (modelToTrain == nullptr) {
        std::cout << "\n[Error] Could not find a model with ID: " << targetID << "\n";
        return;
    }

    std::cout << "\n--- Select Data Source ---\n";
    std::cout << "1. Use auto-generated Dummy Data\n";
    std::cout << "2. Load from CSV file\n";
    std::cout << "Choice: ";
    
    int dataChoice;
    std::cin >> dataChoice;

    // create an empty dataset object locally. 
    Dataset myData(0, 0); 

    if (dataChoice == 1) {
        // generate dummy data matching the EXACT features the model expects
        int expectedFeatures = modelToTrain->getHyperparameters().getInputFeatures();
        myData = Dataset(100, expectedFeatures);
        myData.populateDummyData(); 
        std::cout << "[Info] Dummy data generated (" << expectedFeatures << " features).\n";
    } 
    else if (dataChoice == 2) {
        // load from csv
        std::string filepath;
        int labelCol;
        
        std::cout << "Enter CSV filepath (e.g., ../data/logic_test.csv): ";
        std::cin >> filepath;
        std::cout << "Enter label column index (-1 for last column): ";
        std::cin >> labelCol;
        
        try {
            myData = DataLoader::loadFromCSV(filepath, labelCol);
            
            int modelFeatures = modelToTrain->getHyperparameters().getInputFeatures();
            if (myData.getCols() != modelFeatures) {
                std::cout << "\n[Warning] CSV features (" << myData.getCols() 
                          << ") do not match model's expected features (" << modelFeatures << ")!\n"
                          << "Training might crash or fail.\n";
            }
        } 
        catch (const std::exception& e) {
            std::cout << "[Error] " << e.what() << "\n";
            return;
        }
    } 
    else {
        std::cout << "[Error] Invalid choice. Aborting training.\n";
        return;
    }


    std::cout << "\nTraining model " << targetID << "...\n";
    modelToTrain->train(myData); 
    
    std::cout << "[Success] Training complete!\n";
}


void Menu::listModels() const {
    std::cout << "\n--- Active Models ---\n";
    if (models.empty()) {
        std::cout << "No models instantiated yet.\n";
        return;
    }

    for(size_t i = 0; i < models.size(); ++i){
        std :: cout << i << ". " << *models[i] << '\n';
    }
}


void Menu::saveModel() {
    if (models.empty()) {
        std::cout << "\n[Error] No models available to save.\n";
        return;
    }

    listModels();
    std::cout << "\nEnter the exact ID of the model to save: ";
    std::string targetID;
    std::cin >> targetID;

    MLModel* modelToSave = getModelByID(targetID);

    if (modelToSave != nullptr) {
        std::cout << "Enter filename to save to (e.g., model.json): ";
        std::string filename;
        std::cin >> filename;

        std::string dataDir = "../data/";
        fs::create_directories(dataDir);
        
        std::string fullPath = dataDir + filename;

        std::ofstream file(fullPath);
        if (file.is_open()) {
            json j = modelToSave->serialize(); 
            file << j.dump(4); 
            file.close();
            std::cout << "[Success] Model " << targetID << " saved to " << fullPath << "\n";
        } else {
            std::cout << "[Error] Could not open file for writing at " << fullPath << "\n";
        }
    } else {
        std::cout << "\n[Error] Could not find a model with ID: " << targetID << "\n";
    }
}


void Menu::loadModel() {
    std::cout << "\n--- Load Model ---\n";
    std::cout << "Enter filename to load from (e.g., model.json): ";
    std::string filename;
    std::cin >> filename;

    std::string fullPath = "../data/" + filename;

    std::ifstream file(fullPath);

    if (!file.is_open()) {
        std::cout << "[Error] Could not open file " << filename << ". Does it exist?\n";
        return;
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cout << "[Error] Invalid JSON format: " << e.what() << "\n";
        return;
    }
    file.close();

    std::string modelType = j.value("model_type", "Unknown");
    std::string name = j.value("name", "LoadedModel");
    
    // create default hyperparameters (they will be overwritten if theyre serialized)
    Hyperparameters hp(5, 0.01, 100);
    if (j.contains("hyperparameters")) {
        hp.deserialize(j["hyperparameters"]);
    }

    MLModel* newModel = nullptr;

    if (modelType == "LinearRModel") {
        newModel = new LinearRModel(name, hp, 0.0);
    } else if (modelType == "LogicRModel") {
        newModel = new LogicRModel(name, hp, 2, 0.5);
    } else if (modelType == "KNNModel") {
        newModel = new KNNModel(name, hp, 3, true);
    } else {
        std::cout << "[Error] Unknown or missing model_type in JSON: " << modelType << "\n";
        return;
    }

    // push model into blank
    try {
        newModel->deserialize(j);
        models.push_back(newModel);
        std::cout << "[Success] Model loaded and created with new ID: " << newModel->getModelID() << "\n";
    } catch (const std::exception& e) {
        std::cout << "[Error] Failed to deserialize. Did you pick the wrong model type?\n";
        std::cout << "Details: " << e.what() << "\n";
        // delete if error for memory leaks
        delete newModel;
    }
}



void Menu::modifyModel() {
    if (models.empty()) {
        std::cout << "\n[Error] No models available to modify.\n";
        return;
    }

    listModels();
    std::cout << "\nEnter the exact ID of the model to modify: ";
    std::string targetID;
    std::cin >> targetID;

    MLModel* modelToModify = getModelByID(targetID);

    if (modelToModify == nullptr) {
        std::cout << "\n[Error] Could not find a model with ID: " << targetID << "\n";
        return;
    }

    // get a copy of the current hyperparameters
    Hyperparameters hp = modelToModify->getHyperparameters();

    std::cout << "\n--- Modifying Model: " << modelToModify->getName() << " ---\n";
    std::cout << "1. Modify Learning Rate (Current: " << hp.getLearningRate() << ")\n";
    std::cout << "2. Modify Epochs (Current: " << hp.getEpochs() << ")\n";

    // use dynamic_cast to check what kind of child class
    LogicRModel* logicModel = dynamic_cast<LogicRModel*>(modelToModify);
    LinearRModel* linearModel = dynamic_cast<LinearRModel*>(modelToModify);
    KNNModel* knnModel = dynamic_cast<KNNModel*>(modelToModify);

    // show specific options based on the model type
    if (logicModel != nullptr) {
        std::cout << "3. Modify Decision Threshold\n";
    } else if (linearModel != nullptr) {
        std::cout << "3. Modify L2 Regularization Penalty\n";
    } else if (knnModel != nullptr) {
        std::cout << "3. Modify K (Neighbors)\n";
    }
    
    std::cout << "0. Cancel\n";
    std::cout << "Select a parameter to change: ";
    
    int choice;
    std::cin >> choice;

    switch (choice) {
        case 1: {
            double newLr;
            std::cout << "Enter new learning rate (e.g., 0.05): ";
            std::cin >> newLr;
            hp.setLearningRate(newLr);
            modelToModify->setHyperparameters(hp); // save it back to the model
            std::cout << "[Success] Learning rate updated!\n";
            break;
        }
        case 2: {
            int newEpochs;
            std::cout << "Enter new epochs (e.g., 500): ";
            std::cin >> newEpochs;
            hp.setEpochs(newEpochs);
            modelToModify->setHyperparameters(hp);
            std::cout << "[Success] Epochs updated!\n";
            break;
        }
        case 3: {
            // handle the specific model parameter
            if (logicModel != nullptr) {
                double newThreshold;
                std::cout << "Enter new decision threshold (0.0 - 1.0): ";
                std::cin >> newThreshold;
                // Make sure you added this setter to LogicRModel!
                logicModel->setDecisionThreshold(newThreshold); 
                std::cout << "[Success] Threshold updated!\n";
            } 
            else if (linearModel != nullptr) {
                double newL2;
                std::cout << "Enter new L2 Penalty: ";
                std::cin >> newL2;
                linearModel->setL2Penalty(newL2); // Add to LinearRModel/Regressor
                std::cout << "[Success] L2 Penalty updated!\n";
            }
            else if (knnModel != nullptr) {
                int newK;
                std::cout << "Enter new K value: ";
                std::cin >> newK;
                knnModel->setK(newK); // Add to KNNModel
                std::cout << "[Success] K updated!\n";
            }
            break;
        }
        case 0:
            std::cout << "Modification cancelled.\n";
            break;
        default:
            std::cout << "[Error] Invalid choice.\n";
            break;
    }
}
