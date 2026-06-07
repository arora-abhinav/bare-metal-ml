#pragma once
#include <vector>

using namespace std;

class LinearRegression {
public:
    double theta0;
    double theta1;

    LinearRegression() : theta0(0.0), theta1(0.0) {}

    double hypothesis(double t0, double t1, double x) {
        return t0 + t1 * x;
    }

    void fit(const vector<double>& x_train, const vector<double>& y_train,
             double learning_rate = 0.01, int iterations = 100) {
        double t0 = 0.0;
        double t1 = 0.0;
        int n = x_train.size();

        for (int iter = 0; iter < iterations; iter++) {
            double sum_error_theta0 = 0.0;
            double sum_error_theta1 = 0.0;

            for (int i = 0; i < n; i++) {
                double h = hypothesis(t0, t1, x_train[i]);
                double error = h - y_train[i];
                sum_error_theta0 += error;
                sum_error_theta1 += error * x_train[i];
            }

            t0 -= (learning_rate / n) * sum_error_theta0;
            t1 -= (learning_rate / n) * sum_error_theta1;
        }

        theta0 = t0;
        theta1 = t1;
    }

    vector<double> predict(const vector<double>& x_test) {
        vector<double> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(hypothesis(theta0, theta1, x_test[i]));
        return predictions;
    }

    double mse(const vector<double>& x_test, const vector<double>& y_test) {
        vector<double> predictions = predict(x_test);
        double total = 0.0;
        for (int i = 0; i < (int)predictions.size(); i++)
            total += (predictions[i] - y_test[i]) * (predictions[i] - y_test[i]);
        return total / y_test.size();
    }
};
