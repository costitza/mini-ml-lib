#pragma once 
#include <vector>
#include <string>
#include "MLModel.h"
#include "AuditLogger.h"
#include "ModelManager.h"

class Menu{
private:
    ModelManager modelManager;
    bool isRunning;
    AuditLogger auditLogger;

    // constr + destructor private
    Menu();
    ~Menu() = default;

    // delete copy constr and = operator
    Menu(const Menu&) = delete;
    Menu& operator=(const Menu&) = delete;

    // internal ui helpers
    void createModel();
    void trainModel();
    void listModels() const;
    void saveModel();
    void loadModel();
    void modifyModel();

public:
    static Menu& getInstance();

    void run();
};