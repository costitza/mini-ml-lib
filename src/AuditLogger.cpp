#include "AuditLogger.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

AuditLogger::AuditLogger(const std::string& path) : logFilePath(path) {}

void AuditLogger::update(const std::string& modelName, const std::string& message) {
    try {
        fs::path p(logFilePath);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        std::ofstream file(logFilePath, std::ios::app);
        if (file.is_open()) {
            // get current timestamp
            auto now = std::chrono::system_clock::now();
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
            std::string timeStr = std::ctime(&now_time);
            if (!timeStr.empty() && timeStr.back() == '\n') {
                timeStr.pop_back();
            }

            file << "[" << timeStr << "] MODEL [" << modelName << "] -> " << message << "\n";
        } else {
            std::cerr << "[AuditLogger Error] Could not open log file: " << logFilePath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[AuditLogger Error] Exception during logging: " << e.what() << std::endl;
    }
}