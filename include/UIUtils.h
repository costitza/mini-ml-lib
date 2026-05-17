#pragma once
#include <string>
#include <iostream>
#include <limits>

class UIUtils {
public:
    static void pause();
    static void printHeader(int totalModels);
    static int getIntInput(const std::string& prompt);
    static double getDoubleInput(const std::string& prompt);
    static std::string getStringInput(const std::string& prompt);
    static void printError(const std::string& message);
    static void printSuccess(const std::string& message);
    static void printInfo(const std::string& message);
};
