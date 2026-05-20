#pragma once
#include "IObserver.h"
#include <fstream>
#include <string>

class AuditLogger : public IObserver {
private:
    std::string logFilePath;
public:
    AuditLogger(const std::string& path = "../data/audit.log");
    void update(const std::string& modelName, const std::string& message) override;
};