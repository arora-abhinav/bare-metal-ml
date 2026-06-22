#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include "linalg.hpp"

class GDA {
public:
    string positive_class;
    string negative_class;
    double phi;
    Vec mu_zero;
    Vec mu_one;
    Mat covariance;
    int dimension;

    GDA(string positive_class = "") : positive_class(positive_class), phi(0.0), dimension(0) {}

    void fit(const Mat& x_train, const vector<string>& y_train) {
        // Infer positive and negative class from data if not provided
        map<string, int> class_counts;
        for (int i = 0; i < (int)y_train.size(); i++)
            class_counts[y_train[i]]++;

        if (positive_class.empty())
            positive_class = class_counts.begin()->first;

        for (auto& pair : class_counts)
            if (pair.first != positive_class)
                negative_class = pair.first;

        int n = x_train.size();
        dimension = x_train[0].size();

        phi = compute_phi(y_train, n);
        mu_zero = compute_mu(x_train, y_train, n, false);
        mu_one = compute_mu(x_train, y_train, n, true);
        covariance = compute_covariance(x_train, y_train, n);
    }

    string predict_one(const Vec& x) {
        double log_p_one = log(max(multivariate_gaussian(mu_one, covariance, x), 1e-300)) + log(phi);
        double log_p_zero = log(max(multivariate_gaussian(mu_zero, covariance, x), 1e-300)) + log(1.0 - phi);
        return log_p_one >= log_p_zero ? positive_class : negative_class;
    }

    vector<string> predict(const Mat& x_test) {
        vector<string> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(predict_one(x_test[i]));
        return predictions;
    }

    double accuracy(const Mat& x_test, const vector<string>& y_test) {
        vector<string> predictions = predict(x_test);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++)
            if (predictions[i] == y_test[i])
                correct++;
        return (double)correct / y_test.size() * 100.0;
    }

    void save(const string& path = "gda.json") const {
        ofstream f(path);
        f << setprecision(17);
        f << "{\"positive_class\":\"" << positive_class << "\",";
        f << "\"negative_class\":\"" << negative_class << "\",";
        f << "\"dimension\":" << dimension << ",";
        f << "\"phi\":" << phi << ",";
        f << "\"mu_zero\":[";
        for (int i = 0; i < (int)mu_zero.size(); i++) { if (i) f << ","; f << mu_zero[i]; }
        f << "],\"mu_one\":[";
        for (int i = 0; i < (int)mu_one.size(); i++) { if (i) f << ","; f << mu_one[i]; }
        f << "],\"covariance\":[";
        for (int i = 0; i < (int)covariance.size(); i++) {
            if (i) f << ",";
            f << "[";
            for (int j = 0; j < (int)covariance[i].size(); j++) { if (j) f << ","; f << covariance[i][j]; }
            f << "]";
        }
        f << "]}";
    }

    void load(const string& path = "gda.json") {
        ifstream f(path);
        string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        size_t pos;
        pos = s.find("\"positive_class\":\"") + 18;
        positive_class = s.substr(pos, s.find('"', pos) - pos);
        pos = s.find("\"negative_class\":\"") + 18;
        negative_class = s.substr(pos, s.find('"', pos) - pos);
        pos = s.find("\"dimension\":") + 12;
        dimension = stoi(s.substr(pos));
        pos = s.find("\"phi\":") + 6;
        phi = stod(s.substr(pos));
        pos = s.find("\"mu_zero\":[") + 11;
        mu_zero = _parse_vec(s, pos);
        pos = s.find("\"mu_one\":[") + 10;
        mu_one = _parse_vec(s, pos);
        pos = s.find("\"covariance\":[") + 14;
        covariance = _parse_mat(s, pos);
    }

private:
    static Vec _parse_vec(const string& s, size_t& pos) {
        Vec result;
        pos++; // skip '['
        while (s[pos] != ']') {
            if (s[pos] == ',' || s[pos] == ' ') { pos++; continue; }
            size_t n; result.push_back(stod(s.substr(pos), &n)); pos += n;
        }
        pos++;
        return result;
    }

    static Mat _parse_mat(const string& s, size_t& pos) {
        Mat result;
        pos++; // skip '['
        while (true) {
            while (s[pos] == ',' || s[pos] == ' ') pos++;
            if (s[pos] == ']') { pos++; break; }
            result.push_back(_parse_vec(s, pos));
        }
        return result;
    }

    double compute_phi(const vector<string>& y_train, int n) {
        int count = 0;
        for (int i = 0; i < n; i++)
            if (y_train[i] == positive_class)
                count++;
        return (double)count / n;
    }

    Vec compute_mu(const Mat& x_train, const vector<string>& y_train, int n, bool positive) {
        string target = positive ? positive_class : negative_class;
        double proportion = positive ? phi : (1.0 - phi);
        Vec init_arr(x_train[0].size(), 0.0);
        for (int i = 0; i < n; i++) {
            if (y_train[i] == target) {
                for (int j = 0; j < (int)init_arr.size(); j++)
                    init_arr[j] += x_train[i][j];
            }
        }
        for (int k = 0; k < (int)init_arr.size(); k++)
            init_arr[k] /= (proportion * n);
        return init_arr;
    }

    Mat compute_covariance(const Mat& x_train, const vector<string>& y_train, int n) {
        Mat res(dimension, Vec(dimension, 0.0));
        for (int i = 0; i < n; i++) {
            Vec& mu = (y_train[i] == positive_class) ? mu_one : mu_zero;
            Vec vector = calculate_vector(x_train[i], mu);
            Mat mat = matrix_product_from_vector_and_transpose(dimension, vector);
            for (int j = 0; j < dimension; j++)
                for (int k = 0; k < dimension; k++)
                    res[j][k] += mat[j][k] / n;
        }
        return res;
    }

    double multivariate_gaussian(const Vec& mean, const Mat& cov, const Vec& x) {
        Vec resultant = calculate_vector(x, mean);
        Mat cov_copy = cov;
        cov_copy = regularize(cov_copy, dimension);
        auto [lower, upper] = LU_decomposition(cov_copy, dimension);
        Mat inverse = matrix_inverse(lower, upper, dimension);
        Vec vector_prod = matrix_product_with_matrix_and_vector(inverse, resultant, dimension, dimension);
        double scalar_prod = scalar_product_from_transpose_and_vector(resultant, vector_prod);
        double determinant = calculate_determinant(upper, dimension);
        double exponential_term = exp(-0.5 * scalar_prod);
        double normalizer = 1.0 / (pow(2 * M_PI, dimension / 2.0) * sqrt(abs(determinant)));
        return normalizer * exponential_term;
    }
};
