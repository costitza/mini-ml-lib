#pragma once
#include "Dataset.h"
#include <string>

class DataLoader {
public:
    static Dataset loadFromCSV(const std :: string& filepath, int labelColumnIndex = -1);
};