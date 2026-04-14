#pragma once
#include <vector>
#include <string>
#include <map>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class GDA {
private:
    // Label mapping: 0 = class 0 label, 1 = class 1 label
    std::map<int, std::string> label_map;
    std::string positive_label; // e.g. "M" for malignant

    // GDA parameters
    double phi;           // P(y = 1)
    Vector mu_zero;       // mean feature vector for class 0 (benign)
    Vector mu_one;        // mean feature vector for class 1 (malignant)
    Matrix covariance;    // shared covariance matrix

    int n_features;
    int train_size;
    bool is_trained;

    // Compute log probability under multivariate Gaussian
    // log P(x | mu, Sigma) — uses log space to avoid underflow
    double log_multivariate_gaussian(const Vector& mean, const Matrix& cov, const Vector& x) const;

public:
    // label_map: e.g. {{0, "Benign"}, {1, "Malignant"}}
    // positive_label: the label in raw data that corresponds to class 1, e.g. "M"
    GDA(std::map<int, std::string> label_map, std::string positive_label);

    // Train model: compute phi, mu_zero, mu_one, covariance matrix
    void fit(const Matrix& X_train, const std::vector<std::string>& y_train);

    // Predict class labels for test set
    std::vector<std::string> predict(const Matrix& X_test) const;

    // Return log probabilities for both classes for each test example
    // Returns vector of pairs: {log P(y=0|x), log P(y=1|x)}
    std::vector<std::pair<double,double>> predict_proba(const Matrix& X_test) const;

    // Compute accuracy as percentage
    double score(const Matrix& X_test, const std::vector<std::string>& y_test) const;

    // Print model summary
    void describe() const;
};
