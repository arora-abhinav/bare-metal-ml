#pragma once
#include <vector>
#include <string>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct Dataset {
    Matrix X;
    std::vector<std::string> y;
};

struct TrainTestSplit {
    Matrix X_train;
    Matrix X_test;
    std::vector<std::string> y_train;
    std::vector<std::string> y_test;
};

// Load WDBC CSV — skips ID column, separates label and features
Dataset load_csv(const std::string& filename);

// Shuffle and split data into train/test sets
// test_ratio: fraction of data to use for testing (default 0.2)
// seed: random seed for reproducibility
TrainTestSplit train_test_split(const Dataset& dataset, double test_ratio = 0.2, int seed = 42);

// Z-score standardization per feature column
// Computes mean and std from X_train, applies to both X_train and X_test
void standardize(Matrix& X_train, Matrix& X_test);
