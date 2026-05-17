#include "UIUtils.h"

void UIUtils::pause() {
    std::cout << "\nPress Enter to return to the menu...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}

void UIUtils::printHeader(int totalModels) {
    std::cout << "\n=========================================\n";
    std::cout << "      Machine Learning Library C++       \n";
    std::cout << "          Total Models: " << totalModels << "\n";
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

int UIUtils::getIntInput(const std::string& prompt) {
    int value;
    while (true) {
        std::cout << prompt;
        if (std::cin >> value) {
            return value;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        printError("Invalid input. Please enter a number.");
    }
}

double UIUtils::getDoubleInput(const std::string& prompt) {
    double value;
    while (true) {
        std::cout << prompt;
        if (std::cin >> value) {
            return value;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        printError("Invalid input. Please enter a decimal number.");
    }
}

std::string UIUtils::getStringInput(const std::string& prompt) {
    std::string value;
    std::cout << prompt;
    std::cin >> value;
    return value;
}

void UIUtils::printError(const std::string& message) {
    std::cout << "\n[Error] " << message << "\n";
}

void UIUtils::printSuccess(const std::string& message) {
    std::cout << "\n[Success] " << message << "\n";
}

void UIUtils::printInfo(const std::string& message) {
    std::cout << "\n[Info] " << message << "\n";
}
