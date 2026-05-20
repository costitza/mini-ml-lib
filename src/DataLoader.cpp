#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdexcept>


Dataset DataLoader::loadFromCSV(const std::string& filepath, int labelColumnIndex) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("DataLoader Error: Could not open file " + filepath);
    }

    std::string line;
    std::vector<std::vector<double>> raw_data;
    
    // Read the file line by line
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue; 

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        // parse CSV values separated by comma
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                // skip this entire row and move to the next line.
                row.clear();
                break; 
            }
        }
        
        if (!row.empty()) {
            raw_data.push_back(row);
        }
    }

    if (raw_data.empty()) {
        throw std::runtime_error("DataLoader Error: File is empty or invalid (maybe missing numbers?).");
    }

    int numRows = raw_data.size();
    // total columns minus 1 (because one column is the label, the rest are features)
    int numCols = raw_data[0].size() - 1; 

    // initialize the Dataset
    Dataset dataset(numRows, numCols);

    // populate the dataset
    for (int i = 0; i < numRows; i++) {
        std::vector<double> features;
        double label = 0.0;

        for (size_t j = 0; j < raw_data[i].size(); j++) {
            // check if this column is the target variable (label)
            if (static_cast<int>(j) == labelColumnIndex || (labelColumnIndex == -1 && j == raw_data[i].size() - 1)) {
                label = raw_data[i][j];
            } else {
                features.push_back(raw_data[i][j]);
            }
        }
        dataset.setRow(i, features, label);
    }

              
    return dataset;
}