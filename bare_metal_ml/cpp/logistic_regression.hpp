#pragma once
#include <vector>
#include <string>
#include <set>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace std;

class LogisticRegression {
public:
    string positive_class;
    vector<double> thetas;

    LogisticRegression(string positive_class = "") : positive_class(positive_class) {}

    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    double hypothesis(const vector<double>& params, const vector<double>& features) {
        double z = params[0];
        for (int k = 0; k < (int)features.size(); k++)
            z += params[k + 1] * features[k];
        return sigmoid(z);
    }

    void fit(const vector<vector<double>>& x_train, const vector<string>& y_train,
             double learning_rate = 0.001, int iterations = 1000) {
        if (positive_class.empty()) {
            set<string> classes(y_train.begin(), y_train.end());
            positive_class = *classes.begin();
        }

        int n = x_train.size();
        int n_features = x_train[0].size();
        vector<double> params(n_features + 1, 0.0);

        for (int iter = 0; iter < iterations; iter++) {
            vector<double> updates(n_features + 1, 0.0);
            for (int i = 0; i < n; i++) {
                double h = hypothesis(params, x_train[i]);
                double y = (y_train[i] == positive_class) ? 1.0 : 0.0;
                double error = y - h;
                updates[0] += error;
                for (int k = 0; k < n_features; k++)
                    updates[k + 1] += error * x_train[i][k];
            }
            for (int k = 0; k < (int)params.size(); k++)
                params[k] += (learning_rate * updates[k]) / n;
        }

        thetas = params;
    }

    vector<double> predict_proba(const vector<vector<double>>& x_test) {
        vector<double> probabilities;
        for (int i = 0; i < (int)x_test.size(); i++)
            probabilities.push_back(hypothesis(thetas, x_test[i]));
        return probabilities;
    }

    vector<string> predict(const vector<vector<double>>& x_test, double threshold = 0.5) {
        vector<double> probabilities = predict_proba(x_test);
        vector<string> predictions;
        for (int i = 0; i < (int)probabilities.size(); i++)
            predictions.push_back(probabilities[i] >= threshold ? positive_class : "");
        return predictions;
    }

    double accuracy(const vector<vector<double>>& x_test, const vector<string>& y_test,
                    double threshold = 0.5) {
        vector<string> predictions = predict(x_test, threshold);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++) {
            string actual = (y_test[i] == positive_class) ? positive_class : "";
            if (predictions[i] == actual)
                correct++;
        }
        return (double)correct / y_test.size() * 100.0;
    }

    void save(const string& path = "logistic_regression.json") const {
        ofstream f(path);
        f << setprecision(17);
        f << "{\"positive_class\":\"" << positive_class << "\",\"thetas\":[";
        for (int i = 0; i < (int)thetas.size(); i++) {
            if (i > 0) f << ",";
            f << thetas[i];
        }
        f << "]}";
    }

    void load(const string& path = "logistic_regression.json") {
        ifstream f(path);
        string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        size_t pos = s.find("\"positive_class\":\"") + 18;
        positive_class = s.substr(pos, s.find('"', pos) - pos);
        pos = s.find("\"thetas\":[") + 10;
        thetas.clear();
        while (s[pos] != ']') {
            if (s[pos] == ',' || s[pos] == ' ') { pos++; continue; }
            size_t n; thetas.push_back(stod(s.substr(pos), &n)); pos += n;
        }
    }
};
