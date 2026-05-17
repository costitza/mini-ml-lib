#include "Menu.h"
#include "LinearRModel.h"
#include "LogicRModel.h"
#include "KNNModel.h"
#include "Dataset.h"
#include "Hyperparameters.h"
#include "DataLoader.h"
#include "ModelFactory.h"
#include "UIUtils.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>

namespace fs = std :: filesystem;

// private constr
Menu :: Menu() : isRunning(true) {}

Menu& Menu :: getInstance(){
    static Menu instance;
    return instance;
}

void Menu::run() {
    while (isRunning) {
        UIUtils::printHeader(MLModel::getTotalModels());
        
        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            UIUtils::printError("Invalid input. Please enter a number.");
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
                UIUtils::printError("Invalid choice.");
                break;
        }

        if(isRunning){
            UIUtils::pause();
        }
    }
}

void Menu::createModel() {
    std::cout << "\n--- Create Model ---\n";
    std::cout << "1. Linear Regression\n";
    std::cout << "2. Logistic Regression (Classifier)\n";
    std::cout << "3. K-Nearest Neighbors\n";
    int type = UIUtils::getIntInput("Select type: ");

    std::string name = UIUtils::getStringInput("Enter a name for the model: ");

    // Create default hyperparameters for now (5 features, lr=0.01, 100 epochs)
    Hyperparameters hp(5, 0.01, 100);

    MLModel* newModel = ModelFactory :: createModel(type, name, hp);

    if(newModel){
        newModel -> addObserver(&auditLogger);
        modelManager.addModel(newModel);
        UIUtils::printSuccess("Created " + newModel->getName() + " with ID: " + newModel->getModelID());
    }
    else{
        UIUtils::printError("Unknown model type.");
    }
}

MLModel* selectModel(const ModelManager& manager, const std::string& actionName) {
    if (manager.isEmpty()) {
        UIUtils::printError("No models available to " + actionName + ".");
        return nullptr;
    }

    // List models
    std::cout << "\n--- Active Models ---\n";
    for (const auto* model : manager.getModels()) {
        std::cout << *model << "\n";
    }

    std::string targetID = UIUtils::getStringInput("\nEnter the exact ID of the model to " + actionName + ": ");
    MLModel* model = manager.getModelByID(targetID);

    if (model == nullptr) {
        UIUtils::printError("Could not find a model with ID: " + targetID);
    }
    return model;
}

void Menu::trainModel() {
    MLModel* modelToTrain = selectModel(modelManager, "train");
    if (!modelToTrain) return;

    std::cout << "\n--- Select Data Source ---\n";
    std::cout << "1. Use auto-generated Dummy Data\n";
    std::cout << "2. Load from CSV file\n";
    int dataChoice = UIUtils::getIntInput("Choice: ");

    Dataset myData(0, 0); 

    if (dataChoice == 1) {
        int expectedFeatures = modelToTrain->getHyperparameters().getInputFeatures();
        myData = Dataset(100, expectedFeatures);
        myData.populateDummyData(); 
        UIUtils::printInfo("Dummy data generated (" + std::to_string(expectedFeatures) + " features).");
    } 
    else if (dataChoice == 2) {
        std::string filepath = UIUtils::getStringInput("Enter CSV filepath (e.g., ../data/logic_test.csv): ");
        int labelCol = UIUtils::getIntInput("Enter label column index (-1 for last column): ");
        
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
            UIUtils::printError(e.what());
            return;
        }
    } 
    else {
        UIUtils::printError("Invalid choice. Aborting training.");
        return;
    }

    std::cout << "\nTraining model " << modelToTrain->getModelID() << "...\n";
    modelToTrain->train(myData); 
    UIUtils::printSuccess("Training complete!");
}

void Menu::listModels() const {
    if (modelManager.isEmpty()) {
        std::cout << "\nNo models instantiated yet.\n";
        return;
    }

    std::cout << "\n--- Active Models ---\n";
    const auto& models = modelManager.getModels();
    for(size_t i = 0; i < models.size(); ++i){
        std :: cout << i << ". " << *models[i] << '\n';
    }
}

void Menu::saveModel() {
    MLModel* modelToSave = selectModel(modelManager, "save");
    if (!modelToSave) return;

    std::string filename = UIUtils::getStringInput("Enter filename to save to (e.g., model.json): ");

    std::string dataDir = "../data/";
    fs::create_directories(dataDir);
    std::string fullPath = dataDir + filename;

    std::ofstream file(fullPath);
    if (file.is_open()) {
        json j = modelToSave->serialize(); 
        file << j.dump(4); 
        file.close();
        UIUtils::printSuccess("Model " + modelToSave->getModelID() + " saved to " + fullPath);
    } else {
        UIUtils::printError("Could not open file for writing at " + fullPath);
    }
}

void Menu::loadModel() {
    std::cout << "\n--- Load Model ---\n";
    std::string filename = UIUtils::getStringInput("Enter filename to load from (e.g., model.json): ");
    std::string fullPath = "../data/" + filename;

    std::ifstream file(fullPath);
    if (!file.is_open()) {
        UIUtils::printError("Could not open file " + filename + ". Does it exist?");
        return;
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        UIUtils::printError(std::string("Invalid JSON format: ") + e.what());
        return;
    }
    file.close();

    std::string modelType = j.value("model_type", "Unknown");
    std::string name = j.value("name", "LoadedModel");
    
    Hyperparameters hp(5, 0.01, 100);
    if (j.contains("hyperparameters")) {
        hp.deserialize(j["hyperparameters"]);
    }

    MLModel* newModel = ModelFactory :: createModel(modelType, name, hp);

    try {
        newModel->deserialize(j);
        newModel -> addObserver(&auditLogger);
        modelManager.addModel(newModel);
        UIUtils::printSuccess("Model loaded and created with new ID: " + newModel->getModelID());
    } catch (const std::exception& e) {
        UIUtils::printError(std::string("Failed to deserialize: ") + e.what());
        delete newModel;
    }
}

void Menu::modifyModel() {
    MLModel* modelToModify = selectModel(modelManager, "modify");
    if (!modelToModify) return;

    LogicRModel* logicModel = dynamic_cast<LogicRModel*>(modelToModify);
    LinearRModel* linearModel = dynamic_cast<LinearRModel*>(modelToModify);
    KNNModel* knnModel = dynamic_cast<KNNModel*>(modelToModify);

    bool keepModifying = true;
    while (keepModifying) {
        Hyperparameters hp = modelToModify->getHyperparameters();

        std::cout << "\n--- Modifying Model: " << modelToModify->getName() << " ---\n";
        std::cout << "1. Modify Learning Rate (Current: " << hp.getLearningRate() << ")\n";
        std::cout << "2. Modify Epochs        (Current: " << hp.getEpochs() << ")\n";

        if (logicModel != nullptr) {
            std::cout << "3. Modify Decision Threshold\n";
        } else if (linearModel != nullptr) {
            std::cout << "3. Modify L2 Regularization Penalty\n";
        } else if (knnModel != nullptr) {
            std::cout << "3. Modify K (Neighbors)\n";
        }
        
        std::cout << "0. Done / Return to Main Menu\n";
        int choice = UIUtils::getIntInput("Select a parameter to change: ");

        switch (choice) {
            case 1: {
                double newLr = UIUtils::getDoubleInput("Enter new learning rate (e.g., 0.01): ");
                hp.setLearningRate(newLr);
                modelToModify->setHyperparameters(hp); 
                UIUtils::printSuccess("Learning rate updated!");
                break;
            }
            case 2: {
                int newEpochs = UIUtils::getIntInput("Enter new epochs (e.g., 500): ");
                hp.setEpochs(newEpochs);
                modelToModify->setHyperparameters(hp);
                UIUtils::printSuccess("Epochs updated!");
                break;
            }
            case 3: {
                if (logicModel != nullptr) {
                    double newThreshold = UIUtils::getDoubleInput("Enter new decision threshold (0.0 - 1.0): ");
                    logicModel->setDecisionThreshold(newThreshold); 
                    UIUtils::printSuccess("Threshold updated!");
                } 
                else if (linearModel != nullptr) {
                    double newL2 = UIUtils::getDoubleInput("Enter new L2 Penalty: ");
                    linearModel->setL2Penalty(newL2); 
                    UIUtils::printSuccess("L2 Penalty updated!");
                }
                else if (knnModel != nullptr) {
                    int newK = UIUtils::getIntInput("Enter new K value: ");
                    knnModel->setK(newK); 
                    UIUtils::printSuccess("K updated!");
                } else {
                    UIUtils::printError("Invalid choice for this model type.");
                }
                break;
            }
            case 0:
                UIUtils::printInfo("Finished modifying " + modelToModify->getName() + ". Returning to main menu...");
                keepModifying = false;
                break;
                
            default:
                UIUtils::printError("Invalid choice.");
                break;
        }
    }
}
