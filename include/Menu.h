#pragma once 
#include <vector>
#include <string>
#include "MLModel.h"


class Menu{
private:
    std :: vector<MLModel*> models;
    bool isRunning;

    // constr + destructor private
    Menu();
    ~Menu();

    // delete copy constr and = operator
    Menu(const Menu&) = delete;
    Menu& operator=(const Menu&) = delete;

    // internal ui helpers
    void printHeader() const;
    void createModel();
    void trainModel();
    void listModels() const;
    void saveModel();
    void loadModel();
    void modifyModel();
    MLModel* getModelByID(std :: string targetID);

    void pause() const;

public:
    static Menu& getInstance();

    void run();
};