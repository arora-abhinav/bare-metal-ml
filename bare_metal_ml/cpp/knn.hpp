#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <queue>
#include <map>
#include <algorithm>
#include <tuple>
#include <memory>
#include <limits>

using namespace std;

// ── Distance metrics ──────────────────────────────────────────────────────────

double euclidean(const vector<double>& x1, const vector<double>& x2) {
    double res = 0.0;
    for (int i = 0; i < (int)x1.size(); i++)
        res += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    return sqrt(res);
}

double manhattan(const vector<double>& x1, const vector<double>& x2) {
    double dist = 0.0;
    for (int i = 0; i < (int)x1.size(); i++)
        dist += abs(x1[i] - x2[i]);
    return dist;
}

double cosine(const vector<double>& x1, const vector<double>& x2) {
    double dot_prod = 0.0;
    double x1_len = 0.0;
    double x2_len = 0.0;
    for (int i = 0; i < (int)x1.size(); i++) {
        dot_prod += x1[i] * x2[i];
        x1_len += x1[i] * x1[i];
        x2_len += x2[i] * x2[i];
    }
    return 1.0 - dot_prod / (sqrt(x1_len) * sqrt(x2_len));
}


// ── KNN ───────────────────────────────────────────────────────────────────────

class KNN {
public:
    int k;
    string metric;

    KNN(int k = 5, string metric = "euclidean") : k(k), metric(metric) {}

    void fit(const vector<vector<double>>& x_train, const vector<string>& y_train) {
        this->x_train = x_train;
        this->y_train = y_train;
    }

    string predict_one(const vector<double>& query) {
        // Max-heap: stores (dist, index, label) — pops the largest distance
        priority_queue<tuple<double, int, string>> heap;

        for (int i = 0; i < (int)x_train.size(); i++) {
            double dist = compute_dist(x_train[i], query);
            if ((int)heap.size() < k) {
                heap.push({dist, i, y_train[i]});
            } else if (dist < get<0>(heap.top())) {
                heap.pop();
                heap.push({dist, i, y_train[i]});
            }
        }

        map<string, int> votes;
        while (!heap.empty()) {
            votes[get<2>(heap.top())]++;
            heap.pop();
        }

        string best_label;
        int best_count = 0;
        for (auto& pair : votes) {
            if (pair.second > best_count) {
                best_count = pair.second;
                best_label = pair.first;
            }
        }
        return best_label;
    }

    vector<string> predict(const vector<vector<double>>& x_test) {
        vector<string> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(predict_one(x_test[i]));
        return predictions;
    }

    double accuracy(const vector<vector<double>>& x_test, const vector<string>& y_test) {
        vector<string> predictions = predict(x_test);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++)
            if (predictions[i] == y_test[i])
                correct++;
        return (double)correct / y_test.size() * 100.0;
    }

private:
    vector<vector<double>> x_train;
    vector<string> y_train;

    double compute_dist(const vector<double>& a, const vector<double>& b) {
        if (metric == "manhattan") return manhattan(a, b);
        if (metric == "cosine")    return cosine(a, b);
        return euclidean(a, b);
    }
};


// ── KD-Tree ───────────────────────────────────────────────────────────────────

struct KDNode {
    KDNode* left = nullptr;
    KDNode* right = nullptr;
    pair<int, double> median;   // (dimension_index, median_value)
    vector<double> coordinate;
    int index;

    KDNode(pair<int,double> med, vector<double> coord, int idx)
        : median(med), coordinate(coord), index(idx) {}

    ~KDNode() {
        delete left;
        delete right;
    }
};

class KDTree {
public:
    KDNode* root = nullptr;
    vector<string> y_train;
    int n_features = 0;

    KDTree() {}
    ~KDTree() { delete root; }

    void fit(const vector<vector<double>>& x_train, const vector<string>& labels) {
        y_train = labels;
        n_features = x_train[0].size();
        vector<pair<vector<double>, int>> examples;
        for (int i = 0; i < (int)x_train.size(); i++)
            examples.push_back({x_train[i], i});
        root = construct(examples, 0);
    }

    string predict_one(const vector<double>& query, int k = 1) {
        priority_queue<tuple<double, int, KDNode*>> heap;
        double best_dist = numeric_limits<double>::infinity();
        search(root, query, 0, best_dist, heap, k);

        map<string, int> votes;
        while (!heap.empty()) {
            KDNode* node = get<2>(heap.top());
            votes[y_train[node->index]]++;
            heap.pop();
        }

        string best_label;
        int best_count = 0;
        for (auto& pair : votes) {
            if (pair.second > best_count) {
                best_count = pair.second;
                best_label = pair.first;
            }
        }
        return best_label;
    }

    vector<string> predict(const vector<vector<double>>& x_test, int k = 1) {
        vector<string> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(predict_one(x_test[i], k));
        return predictions;
    }

    double accuracy(const vector<vector<double>>& x_test, const vector<string>& y_test, int k = 1) {
        vector<string> predictions = predict(x_test, k);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++)
            if (predictions[i] == y_test[i])
                correct++;
        return (double)correct / y_test.size() * 100.0;
    }

private:
    pair<pair<vector<double>,int>, int> obtain_median(vector<pair<vector<double>,int>>& coords) {
        int median_index = (coords.size() - 1) / 2;
        return {coords[median_index], median_index};
    }

    KDNode* construct(vector<pair<vector<double>,int>> examples, int index) {
        if (examples.empty()) return nullptr;

        sort(examples.begin(), examples.end(), [&](const auto& a, const auto& b) {
            return a.first[index] < b.first[index];
        });

        auto [median, median_index] = obtain_median(examples);
        KDNode* node = new KDNode(
            {index, median.first[index]},
            median.first,
            median.second
        );

        if (examples.size() > 1) {
            vector<pair<vector<double>,int>> left_examples(examples.begin(), examples.begin() + median_index);
            vector<pair<vector<double>,int>> right_examples(examples.begin() + median_index + 1, examples.end());
            node->left  = construct(left_examples,  (index + 1) % n_features);
            node->right = construct(right_examples, (index + 1) % n_features);
        }
        return node;
    }

    void search(KDNode* node, const vector<double>& query, int ind,
                double& best_dist, priority_queue<tuple<double,int,KDNode*>>& heap, int k) {
        if (node == nullptr) return;

        double current_dist = euclidean(node->coordinate, query);

        if ((int)heap.size() < k) {
            heap.push({current_dist, node->index, node});
            best_dist = get<0>(heap.top());
        } else if (current_dist < best_dist) {
            heap.pop();
            heap.push({current_dist, node->index, node});
            best_dist = get<0>(heap.top());
        }

        // Descend the side the query falls on first
        KDNode* primary = (query[ind] > node->median.second) ? node->right : node->left;
        KDNode* other   = (query[ind] > node->median.second) ? node->left  : node->right;

        search(primary, query, (ind + 1) % n_features, best_dist, heap, k);

        // Explore other side only if the splitting plane is within best_dist
        if (abs(query[ind] - node->median.second) < best_dist)
            search(other, query, (ind + 1) % n_features, best_dist, heap, k);
    }
};
