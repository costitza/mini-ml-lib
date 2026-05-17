# ML mini library for predictive models

A lightweight, from-scratch Machine Learning library built with modern C++ (C++20). This project is designed for educational purposes to demonstrate the internal mechanics of fundamental ML algorithms without relying on high-level frameworks like scikit-learn or PyTorch.

## Core Models

The library currently supports the following models:
* **Linear Regression:** Predicts continuous numerical values using the Normal Equation and matrix operations.
* **Logistic Regression:** A binary classifier implemented with a sigmoid activation function and gradient-based updates.
* **K-Nearest Neighbors (KNN):** A versatile lazy learner supporting both classification (via majority vote) and regression (via mean averaging).

## Key Features

### Interactive CLI & Menu
A terminal-based interface provides full control over the model lifecycle. The menu system is built for resilience, featuring robust input validation and clear feedback.

### Modular OOP Architecture
The codebase follows strict Object-Oriented principles, utilizing polymorphism, virtual inheritance, and specialized manager classes:
* **ModelManager:** Handles model storage, lifecycle, and ID-based searching.
* **UIUtils:** Centralizes terminal interaction, input parsing, and consistent formatting.
* **ModelFactory:** Manages the instantiation of different model types from a unified interface.

### Persistent Storage (Save/Load)
Models can be serialized into JSON format, preserving their learned weights, hyperparameters, and specific configurations.
* **Serialization:** Saves the complete state to the `data/` directory.
* **Auto-Reconstruction:** When loading, the library automatically detects the model type and restores it to its exact trained state.

### Batch Prediction on CSV
A dedicated feature for evaluating model performance on external data:
* **Input:** Load any compatible CSV file from the `data/` subdirectory.
* **Output:** Generates a result CSV with true labels versus model predictions.
* **Metrics:** Automatically calculates and displays the **Accuracy Score** for classification and **Mean Squared Error (MSE)** for regression.

### Expanded Audit Logging
The library implements the Observer pattern to maintain a detailed security and operation log in `data/audit.log`. The system automatically records:
* Model creation and source data identification.
* Training start/finish events and specific file sources.
* Real-time hyperparameter modifications.
* Serialization and deserialization paths.
* Final performance metrics from batch predictions.

## Technical Stack

* **[Eigen](https://eigen.tuxfamily.org/):** Used for high-performance linear algebra, matrix operations, and vector math.
* **[nlohmann/json](https://github.com/nlohmann/json):** Facilitates seamless JSON parsing and object serialization.
* **Standard Filesystem:** Utilizes `std::filesystem` for automatic directory management and cross-platform path handling.

## Installation

### Prerequisites
* CMake 3.15 or higher.
* A C++20 compatible compiler (GCC 10+, Clang 10+, or MSVC 2019+).

### Build Instructions
1. Clone the repository.
2. Create a build directory: `mkdir build && cd build`.
3. Configure the project: `cmake ..`.
4. Compile: `cmake --build .`.

## How to Use

Run the executable from the `build` directory. All data operations (CSV files and JSON models) are relative to the `data/` folder.

1. **Create a Model (Option 1):** Instantiate a model and assign it a name. The system generates a unique Hex ID (e.g., `MOD-A3F1`).
2. **Train a Model (Option 2):** Select a model by ID and choose between synthetic dummy data or an external CSV file.
3. **Modify Parameters (Option 3):** Tune learning rates, epochs, decision thresholds, or K-neighbors in real-time.
4. **List Models (Option 4):** View all active models and their current status.
5. **Save/Load (Options 5 & 6):** Manage persistent storage of your models.
6. **Predict on CSV (Option 7):** Provide a filename from the `data/` folder to run batch predictions and view performance metrics.

## Error Handling

The library utilizes a custom exception hierarchy to ensure system stability. It validates matrix dimensions, feature counts, and training states before execution, providing descriptive error messages instead of terminating unexpectedly.
