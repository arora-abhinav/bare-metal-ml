#include "DataUtils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>

// Load WDBC CSV file
// Format: ID, Label (M/B), feature1, feature2, ..., feature30
Dataset load_csv(const std::string& filename) {
    Dataset dataset;
    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ','))
            tokens.push_back(token);

        // tokens[0] = ID (skip)
        // tokens[1] = label (M or B)
        // tokens[2..31] = features
        if (tokens.size() < 3) continue;

        dataset.y.push_back(tokens[1]);

        Vector features;
        for (int i = 2; i < (int)tokens.size(); i++)
            features.push_back(std::stod(tokens[i]));

        dataset.X.push_back(features);
    }

    return dataset;
}

// Shuffle and split into 80/20 train/test
TrainTestSplit train_test_split(const Dataset& dataset, double test_ratio, int seed) {
    int n = dataset.X.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle with fixed seed for reproducibility
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    int test_size = (int)(n * test_ratio);
    int train_size = n - test_size;

    TrainTestSplit split;
    for (int i = 0; i < train_size; i++) {
        split.X_train.push_back(dataset.X[indices[i]]);
        split.y_train.push_back(dataset.y[indices[i]]);
    }
    for (int i = train_size; i < n; i++) {
        split.X_test.push_back(dataset.X[indices[i]]);
        split.y_test.push_back(dataset.y[indices[i]]);
    }

    return split;
}

// Z-score standardization
// For each feature j: x = (x - mean_j) / std_j
// mean and std computed from X_train only, applied to both
void standardize(Matrix& X_train, Matrix& X_test) {
    int n_train = X_train.size();
    int n_test  = X_test.size();
    int n_features = X_train[0].size();

    for (int j = 0; j < n_features; j++) {
        // Compute mean of feature j over training set
        double mean = 0.0;
        for (int i = 0; i < n_train; i++)
            mean += X_train[i][j];
        mean /= n_train;

        // Compute std using E[X^2] - E[X]^2
        double mean_sq = 0.0;
        for (int i = 0; i < n_train; i++)
            mean_sq += X_train[i][j] * X_train[i][j];
        mean_sq /= n_train;

        double std = std::sqrt(mean_sq - mean * mean);
        if (std < 1e-10) std = 1e-10; // prevent division by zero

        // Apply to training set
        for (int i = 0; i < n_train; i++)
            X_train[i][j] = (X_train[i][j] - mean) / std;

        // Apply same mean/std to test set
        for (int i = 0; i < n_test; i++)
            X_test[i][j] = (X_test[i][j] - mean) / std;
    }
}
