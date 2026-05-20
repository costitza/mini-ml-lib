#pragma once
#include <string>

// interface
class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void update(const std::string& modelName, const std::string& message) = 0;
};