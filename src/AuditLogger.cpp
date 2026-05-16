#include "AuditLogger.h"
#include <chrono>
#include <ctime>

AuditLogger::AuditLogger(const std::string& path) : logFilePath(path) {}

void AuditLogger::update(const std::string& modelName, const std::string& message) {
    std::ofstream file(logFilePath, std::ios::app);
    if (file.is_open()) {

        // get current timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::string timeStr = std::ctime(&now_time);
        timeStr.pop_back();

        file << "[" << timeStr << "] MODEL [" << modelName << "] -> " << message << "\n";
    }
}