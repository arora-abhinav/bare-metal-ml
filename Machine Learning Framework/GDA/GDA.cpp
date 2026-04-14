#include "GDA.h"
#include "LinearAlgebra.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

GDA::GDA(std::map<int, std::string> label_map, std::string positive_label)
    : label_map(label_map), positive_label(positive_label),
      phi(0.0), n_features(0), train_size(0), is_trained(false) {}

void GDA::fit(const Matrix& X_train, const std::vector<std::string>& y_train) {
    train_size = X_train.size();
    n_features = X_train[0].size();

    // --- Compute phi = P(y = 1) ---
    // Count how many training examples belong to the positive class
    int count_one = 0;
    for (int i = 0; i < train_size; i++)
        if (y_train[i] == positive_label)
            count_one++;
    phi = (double)count_one / train_size;

    // --- Compute mu_zero: mean feature vector for class 0 (benign) ---
    mu_zero.assign(n_features, 0.0);
    int count_zero = 0;
    for (int i = 0; i < train_size; i++) {
        if (y_train[i] != positive_label) {
            for (int j = 0; j < n_features; j++)
                mu_zero[j] += X_train[i][j];
            count_zero++;
        }
    }
    for (int j = 0; j < n_features; j++)
        mu_zero[j] /= count_zero;

    // --- Compute mu_one: mean feature vector for class 1 (malignant) ---
    mu_one.assign(n_features, 0.0);
    for (int i = 0; i < train_size; i++) {
        if (y_train[i] == positive_label) {
            for (int j = 0; j < n_features; j++)
                mu_one[j] += X_train[i][j];
        }
    }
    for (int j = 0; j < n_features; j++)
        mu_one[j] /= count_one;

    // --- Compute shared covariance matrix ---
    // Sigma = (1/m) * sum over all i of (x(i) - mu_y(i)) * (x(i) - mu_y(i))^T
    covariance.assign(n_features, Vector(n_features, 0.0));
    for (int i = 0; i < train_size; i++) {
        // Pick the right mean based on class label
        const Vector& mu = (y_train[i] == positive_label) ? mu_one : mu_zero;

        // Compute (x(i) - mu)
        Vector diff = subtract_vectors(Vector(X_train[i].begin(), X_train[i].end()), mu);

        // Compute outer product (x(i) - mu)(x(i) - mu)^T and accumulate
        Matrix mat = outer_product(diff, n_features);
        for (int j = 0; j < n_features; j++)
            for (int k = 0; k < n_features; k++)
                covariance[j][k] += mat[j][k] / train_size;
    }

    is_trained = true;
}

// Log multivariate Gaussian:
// log P(x | mu, Sigma) = -0.5 * (x-mu)^T Sigma^-1 (x-mu)
//                      - 0.5 * log|Sigma|
//                      - (n/2) * log(2*pi)
double GDA::log_multivariate_gaussian(const Vector& mean, const Matrix& cov, const Vector& x) const {
    // Make a copy and regularize to prevent singular matrix
    Matrix cov_copy = regularize(cov, n_features);

    // LU decomposition
    auto [lower, upper] = lu_decomposition(cov_copy, n_features);

    // Compute inverse
    Matrix inv = matrix_inverse(lower, upper, n_features);

    // Compute (x - mu)
    Vector diff = subtract_vectors(x, mean);

    // Compute Sigma^-1 * (x - mu)
    Vector inv_diff = matrix_vector_product(inv, diff, n_features);

    // Compute (x - mu)^T * Sigma^-1 * (x - mu) — scalar
    double quadratic = dot_product(diff, inv_diff, n_features);

    // Compute log|Sigma| = log of product of U diagonal
    double det = calculate_determinant(upper, n_features);
    double log_det = std::log(std::abs(det));

    double log_prob = -0.5 * quadratic
                    - 0.5 * log_det
                    - (n_features / 2.0) * std::log(2.0 * M_PI);
    return log_prob;
}

std::vector<std::string> GDA::predict(const Matrix& X_test) const {
    if (!is_trained)
        throw std::runtime_error("Model not trained. Call fit() first.");

    std::vector<std::string> predictions;
    for (const auto& x : X_test) {
        // log P(y=0|x) proportional to log P(x|y=0) + log P(y=0)
        double log_zero = log_multivariate_gaussian(mu_zero, covariance, x) + std::log(1.0 - phi);

        // log P(y=1|x) proportional to log P(x|y=1) + log P(y=1)
        double log_one  = log_multivariate_gaussian(mu_one,  covariance, x) + std::log(phi);

        // Predict whichever class has higher log probability
        int pred = (log_one > log_zero) ? 1 : 0;
        predictions.push_back(label_map.at(pred));
    }
    return predictions;
}

std::vector<std::pair<double,double>> GDA::predict_proba(const Matrix& X_test) const {
    if (!is_trained)
        throw std::runtime_error("Model not trained. Call fit() first.");

    std::vector<std::pair<double,double>> probas;
    for (const auto& x : X_test) {
        double log_zero = log_multivariate_gaussian(mu_zero, covariance, x) + std::log(1.0 - phi);
        double log_one  = log_multivariate_gaussian(mu_one,  covariance, x) + std::log(phi);
        probas.push_back({log_zero, log_one});
    }
    return probas;
}

double GDA::score(const Matrix& X_test, const std::vector<std::string>& y_test) const {
    std::vector<std::string> predictions = predict(X_test);
    int correct = 0;
    for (int i = 0; i < (int)predictions.size(); i++)
        if (predictions[i] == label_map.at(y_test[i] == positive_label ? 1 : 0))
            correct++;
    return (double)correct / y_test.size() * 100.0;
}

void GDA::describe() const {
    std::cout << "================================\n";
    std::cout << "Model: Gaussian Discriminant Analysis (GDA)\n";
    std::cout << "Classes: {" << label_map.at(0) << ", " << label_map.at(1) << "}\n";
    std::cout << "Positive class: " << positive_label << "\n";
    std::cout << "Features: " << n_features << "\n";
    std::cout << "Training examples: " << train_size << "\n";
    std::cout << "phi (P(y=1)): " << phi << "\n";
    std::cout << "Trained: " << (is_trained ? "Yes" : "No") << "\n";
    std::cout << "================================\n";
}
