#pragma once
#include <Eigen/Dense>
#include <vector>

class Dataset{
    int rows;
    int cols;
    double *labels;
    double **features;

public:
    Dataset(int r, int c);
    // copy constructor
    Dataset(const Dataset& other);

    // deep copy
    Dataset& operator=(const Dataset& other);

    ~Dataset();

    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    
    // get the target label for a specific row
    double getLabel(int index) const;
    // convert c++ array row into eigen vector 
    Eigen :: VectorXd getRowsAsEigen(int index) const;
    void setRow(int index, const std :: vector<double>& rowFeatures, double label);

    void populateDummyData();
};
