#pragma once
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <fstream>
#include <iomanip>

using namespace std;


// ── Shared text utilities ─────────────────────────────────────────────────────

vector<string> tokenize(const string& text) {
    string lower = text;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    string clean;
    for (char c : lower) {
        if (isalpha(c) || isspace(c))
            clean += c;
        else
            clean += ' ';
    }
    istringstream iss(clean);
    vector<string> tokens;
    string token;
    while (iss >> token)
        tokens.push_back(token);
    return tokens;
}

unordered_map<string, int> build_vocab(const vector<vector<string>>& train_tokens, int vocab_size) {
    unordered_map<string, int> counter_map;
    for (int i = 0; i < (int)train_tokens.size(); i++)
        for (int j = 0; j < (int)train_tokens[i].size(); j++)
            counter_map[train_tokens[i][j]]++;

    vector<pair<string, int>> sorted_map(counter_map.begin(), counter_map.end());
    sort(sorted_map.begin(), sorted_map.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    unordered_map<string, int> final_map;
    int counter = 0;
    int limit = min((int)sorted_map.size(), vocab_size);
    for (int i = 0; i < limit; i++) {
        final_map[sorted_map[i].first] = counter;
        counter++;
    }
    return final_map;
}

map<string, int> calculate_class_ratio(const vector<string>& y_train) {
    map<string, int> m;
    for (int i = 0; i < (int)y_train.size(); i++)
        m[y_train[i]]++;
    return m;
}

vector<vector<int>> vectorize(const vector<string>& x, int vocab_size,
                               const vector<vector<string>>& tokens,
                               const unordered_map<string, int>& vocab_list, bool count) {
    vector<vector<int>> vectorized(x.size(), vector<int>(vocab_size, 0));
    for (int i = 0; i < (int)x.size(); i++) {
        for (int j = 0; j < (int)tokens[i].size(); j++) {
            auto it = vocab_list.find(tokens[i][j]);
            if (it != vocab_list.end()) {
                if (!count)
                    vectorized[i][it->second] = 1;
                else
                    vectorized[i][it->second]++;
            }
        }
    }
    return vectorized;
}


// ── Gaussian Naive Bayes ──────────────────────────────────────────────────────

class GaussianNaiveBayes {
public:
    GaussianNaiveBayes() : n(0) {}

    void fit(const vector<vector<double>>& x_train, const vector<string>& y_train) {
        n = y_train.size();
        class_ratio = calculate_class_ratio(y_train);

        int counter = 0;
        for (int i = 0; i < n; i++) {
            if (class_numbering.find(y_train[i]) == class_numbering.end()) {
                class_numbering[y_train[i]] = counter;
                counter++;
            }
        }
        calculate_mean(x_train, y_train);
        calculate_variance(x_train, y_train);
    }

    string predict_one(const vector<double>& x) {
        string prediction;
        double highest_probability = -numeric_limits<double>::infinity();
        for (auto& cls : class_numbering) {
            string class_type = cls.first;
            double probability_sum = log((double)class_ratio[class_type] / n);
            for (int j = 0; j < (int)x.size(); j++) {
                probability_sum += log(gaussian_probability(
                    x[j],
                    mean_dict[{class_type, j}],
                    variance_dict[{class_type, j}]
                ));
            }
            if (probability_sum > highest_probability) {
                prediction = class_type;
                highest_probability = probability_sum;
            }
        }
        return prediction;
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

    void save(const string& path = "gaussian_nb.json") const {
        ofstream f(path);
        f << setprecision(17);
        f << "{\"n\":" << n << ",";
        f << "\"class_ratio\":{";
        bool first = true;
        for (auto& kv : class_ratio) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"class_numbering\":{";
        first = true;
        for (auto& kv : class_numbering) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"mean\":{";
        first = true;
        for (auto& kv : mean_dict) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first.first << "|" << kv.first.second << "\":" << kv.second;
        }
        f << "},\"variance\":{";
        first = true;
        for (auto& kv : variance_dict) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first.first << "|" << kv.first.second << "\":" << kv.second;
        }
        f << "}}";
    }

    void load(const string& path = "gaussian_nb.json") {
        ifstream f(path);
        string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        n = stoi(s.substr(s.find("\"n\":") + 4));
        auto [cr_s, cr_e] = _obj_range(s, "class_ratio");
        class_ratio = _parse_int_map(s, cr_s, cr_e);
        auto [cn_s, cn_e] = _obj_range(s, "class_numbering");
        class_numbering = _parse_int_map(s, cn_s, cn_e);
        auto [m_s, m_e] = _obj_range(s, "mean");
        mean_dict.clear();
        for (auto& kv : _parse_dbl_map(s, m_s, m_e)) {
            size_t sep = kv.first.rfind('|');
            mean_dict[{kv.first.substr(0, sep), stoi(kv.first.substr(sep + 1))}] = kv.second;
        }
        auto [v_s, v_e] = _obj_range(s, "variance");
        variance_dict.clear();
        for (auto& kv : _parse_dbl_map(s, v_s, v_e)) {
            size_t sep = kv.first.rfind('|');
            variance_dict[{kv.first.substr(0, sep), stoi(kv.first.substr(sep + 1))}] = kv.second;
        }
    }

private:
    int n;
    map<string, int> class_ratio;
    map<string, int> class_numbering;
    map<pair<string,int>, double> mean_dict;
    map<pair<string,int>, double> variance_dict;

    void calculate_mean(const vector<vector<double>>& x_train, const vector<string>& y_train) {
        int feature_number = 0;
        for (int i = 0; i < (int)x_train[0].size(); i++) {
            map<string, double> auxiliary_dict;
            for (int j = 0; j < (int)x_train.size(); j++) {
                if (auxiliary_dict.find(y_train[j]) == auxiliary_dict.end())
                    auxiliary_dict[y_train[j]] = x_train[j][i] / class_ratio[y_train[j]];
                else
                    auxiliary_dict[y_train[j]] += x_train[j][i] / class_ratio[y_train[j]];
            }
            for (auto& element : auxiliary_dict)
                mean_dict[{element.first, feature_number}] = element.second;
            feature_number++;
        }
    }

    void calculate_variance(const vector<vector<double>>& x_train, const vector<string>& y_train) {
        // Var(X) = E[X^2] - E[X]^2
        int feature_number = 0;
        for (int i = 0; i < (int)x_train[0].size(); i++) {
            map<string, double> auxiliary_dict;
            for (int j = 0; j < (int)x_train.size(); j++) {
                if (auxiliary_dict.find(y_train[j]) == auxiliary_dict.end())
                    auxiliary_dict[y_train[j]] = (x_train[j][i] * x_train[j][i]) / class_ratio[y_train[j]];
                else
                    auxiliary_dict[y_train[j]] += (x_train[j][i] * x_train[j][i]) / class_ratio[y_train[j]];
            }
            for (auto& element : auxiliary_dict) {
                double mean = mean_dict[{element.first, feature_number}];
                variance_dict[{element.first, feature_number}] = element.second - mean * mean;
            }
            feature_number++;
        }
    }

    double gaussian_probability(double x, double mean, double variance) {
        variance = max(variance, 1e-10);
        double prob = (1.0 / sqrt(2 * M_PI * variance)) * exp(-((x - mean) * (x - mean)) / (2 * variance));
        return max(prob, 1e-300);
    }

    static pair<size_t,size_t> _obj_range(const string& s, const string& key) {
        string pat = "\"" + key + "\":{";
        size_t open = s.find(pat) + pat.size() - 1;
        int depth = 1; size_t i = open + 1;
        while (i < s.size() && depth > 0) { if (s[i]=='{') depth++; else if (s[i]=='}') depth--; i++; }
        return {open + 1, i - 1};
    }

    static map<string,int> _parse_int_map(const string& s, size_t st, size_t en) {
        map<string,int> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stoi(s.substr(pos), &n); pos += n;
        }
        return res;
    }

    static map<string,double> _parse_dbl_map(const string& s, size_t st, size_t en) {
        map<string,double> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stod(s.substr(pos), &n); pos += n;
        }
        return res;
    }
};


// ── Bernoulli Naive Bayes ─────────────────────────────────────────────────────

class BernoulliNaiveBayes {
public:
    int vocab_size;

    BernoulliNaiveBayes(int vocab_size = 1000) : vocab_size(vocab_size), n(0) {}

    void fit(const vector<string>& x_train, const vector<string>& y_train) {
        n = y_train.size();
        vector<vector<string>> train_tokens;
        for (int i = 0; i < (int)x_train.size(); i++)
            train_tokens.push_back(tokenize(x_train[i]));
        vocab = build_vocab(train_tokens, vocab_size);
        vector<vector<int>> vectorized = vectorize(x_train, vocab_size, train_tokens, vocab, false);
        class_ratio = calculate_class_ratio(y_train);
        calculate_phi(vectorized, y_train);
    }

    string predict_one(const string& doc) {
        vector<string> tokens = tokenize(doc);
        vector<vector<int>> vec = vectorize({doc}, vocab_size, {tokens}, vocab, false);
        string prediction;
        double best_score = -numeric_limits<double>::infinity();
        for (auto& cls : class_ratio) {
            string element = cls.first;
            double s = log((double)class_ratio[element] / n);
            for (int i = 0; i < vocab_size; i++)
                s += log(bernoulli_dist(vec[0][i], phi_dict[{element, i}]));
            if (s > best_score) {
                best_score = s;
                prediction = element;
            }
        }
        return prediction;
    }

    vector<string> predict(const vector<string>& x_test) {
        vector<string> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(predict_one(x_test[i]));
        return predictions;
    }

    double accuracy(const vector<string>& x_test, const vector<string>& y_test) {
        vector<string> predictions = predict(x_test);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++)
            if (predictions[i] == y_test[i])
                correct++;
        return (double)correct / y_test.size() * 100.0;
    }

    void save(const string& path = "bernoulli_nb.json") const {
        ofstream f(path);
        f << setprecision(17);
        f << "{\"vocab_size\":" << vocab_size << ",\"n\":" << n << ",";
        f << "\"vocab\":{";
        bool first = true;
        for (auto& kv : vocab) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"class_ratio\":{";
        first = true;
        for (auto& kv : class_ratio) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"phi\":{";
        first = true;
        for (auto& kv : phi_dict) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first.first << "|" << kv.first.second << "\":" << kv.second;
        }
        f << "}}";
    }

    void load(const string& path = "bernoulli_nb.json") {
        ifstream f(path);
        string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        vocab_size = stoi(s.substr(s.find("\"vocab_size\":") + 13));
        n = stoi(s.substr(s.find("\"n\":") + 4));
        auto [v_s, v_e] = _bnb_obj_range(s, "vocab");
        vocab.clear();
        for (auto& kv : _bnb_int_map(s, v_s, v_e)) vocab[kv.first] = kv.second;
        auto [cr_s, cr_e] = _bnb_obj_range(s, "class_ratio");
        class_ratio = _bnb_int_map(s, cr_s, cr_e);
        auto [p_s, p_e] = _bnb_obj_range(s, "phi");
        phi_dict.clear();
        for (auto& kv : _bnb_dbl_map(s, p_s, p_e)) {
            size_t sep = kv.first.rfind('|');
            phi_dict[{kv.first.substr(0, sep), stoi(kv.first.substr(sep + 1))}] = kv.second;
        }
    }

private:
    int n;
    unordered_map<string, int> vocab;
    map<string, int> class_ratio;
    map<pair<string,int>, double> phi_dict;

    void calculate_phi(const vector<vector<int>>& vectorized_x, const vector<string>& y_train) {
        int feature_number = 0;
        for (int i = 0; i < vocab_size; i++) {
            map<string, int> auxiliary_dict;
            for (auto& cls : class_ratio) auxiliary_dict[cls.first] = 0;
            for (int j = 0; j < (int)vectorized_x.size(); j++)
                auxiliary_dict[y_train[j]] += vectorized_x[j][i];
            for (auto& element : auxiliary_dict)
                phi_dict[{element.first, feature_number}] = (double)(element.second + 1) / (class_ratio[element.first] + 2);
            feature_number++;
        }
    }

    double bernoulli_dist(int x, double phi) {
        return pow(phi, x) * pow(1.0 - phi, 1 - x);
    }

    static pair<size_t,size_t> _bnb_obj_range(const string& s, const string& key) {
        string pat = "\"" + key + "\":{";
        size_t open = s.find(pat) + pat.size() - 1;
        int depth = 1; size_t i = open + 1;
        while (i < s.size() && depth > 0) { if (s[i]=='{') depth++; else if (s[i]=='}') depth--; i++; }
        return {open + 1, i - 1};
    }

    static map<string,int> _bnb_int_map(const string& s, size_t st, size_t en) {
        map<string,int> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stoi(s.substr(pos), &n); pos += n;
        }
        return res;
    }

    static map<string,double> _bnb_dbl_map(const string& s, size_t st, size_t en) {
        map<string,double> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stod(s.substr(pos), &n); pos += n;
        }
        return res;
    }
};


// ── Multinomial Naive Bayes ───────────────────────────────────────────────────

class MultinomialNaiveBayes {
public:
    int vocab_size;

    MultinomialNaiveBayes(int vocab_size = 1000) : vocab_size(vocab_size), n(0) {}

    void fit(const vector<string>& x_train, const vector<string>& y_train) {
        n = y_train.size();
        vector<vector<string>> train_tokens;
        for (int i = 0; i < (int)x_train.size(); i++)
            train_tokens.push_back(tokenize(x_train[i]));
        vocab = build_vocab(train_tokens, vocab_size);
        vector<vector<int>> vectorized = vectorize(x_train, vocab_size, train_tokens, vocab, true);
        class_ratio = calculate_class_ratio(y_train);
        calculate_phi(vectorized, y_train);
    }

    string predict_one(const string& doc) {
        vector<string> tokens = tokenize(doc);
        vector<vector<int>> vec = vectorize({doc}, vocab_size, {tokens}, vocab, true);
        string prediction;
        double best_score = -numeric_limits<double>::infinity();
        for (auto& cls : class_ratio) {
            string element = cls.first;
            double s = log((double)class_ratio[element] / n);
            for (int i = 0; i < vocab_size; i++)
                s += vec[0][i] * log(phi_dict[{element, i}]);
            if (s > best_score) {
                best_score = s;
                prediction = element;
            }
        }
        return prediction;
    }

    vector<string> predict(const vector<string>& x_test) {
        vector<string> predictions;
        for (int i = 0; i < (int)x_test.size(); i++)
            predictions.push_back(predict_one(x_test[i]));
        return predictions;
    }

    double accuracy(const vector<string>& x_test, const vector<string>& y_test) {
        vector<string> predictions = predict(x_test);
        int correct = 0;
        for (int i = 0; i < (int)predictions.size(); i++)
            if (predictions[i] == y_test[i])
                correct++;
        return (double)correct / y_test.size() * 100.0;
    }

    void save(const string& path = "multinomial_nb.json") const {
        ofstream f(path);
        f << setprecision(17);
        f << "{\"vocab_size\":" << vocab_size << ",\"n\":" << n << ",";
        f << "\"vocab\":{";
        bool first = true;
        for (auto& kv : vocab) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"class_ratio\":{";
        first = true;
        for (auto& kv : class_ratio) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "},\"phi\":{";
        first = true;
        for (auto& kv : phi_dict) {
            if (!first) f << ","; first = false;
            f << "\"" << kv.first.first << "|" << kv.first.second << "\":" << kv.second;
        }
        f << "}}";
    }

    void load(const string& path = "multinomial_nb.json") {
        ifstream f(path);
        string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        vocab_size = stoi(s.substr(s.find("\"vocab_size\":") + 13));
        n = stoi(s.substr(s.find("\"n\":") + 4));
        auto [v_s, v_e] = _mnb_obj_range(s, "vocab");
        vocab.clear();
        for (auto& kv : _mnb_int_map(s, v_s, v_e)) vocab[kv.first] = kv.second;
        auto [cr_s, cr_e] = _mnb_obj_range(s, "class_ratio");
        class_ratio = _mnb_int_map(s, cr_s, cr_e);
        auto [p_s, p_e] = _mnb_obj_range(s, "phi");
        phi_dict.clear();
        for (auto& kv : _mnb_dbl_map(s, p_s, p_e)) {
            size_t sep = kv.first.rfind('|');
            phi_dict[{kv.first.substr(0, sep), stoi(kv.first.substr(sep + 1))}] = kv.second;
        }
    }

private:
    int n;
    unordered_map<string, int> vocab;
    map<string, int> class_ratio;
    map<pair<string,int>, double> phi_dict;

    map<string, int> calculate_word_count(const vector<vector<int>>& vectorized_x,
                                           const vector<string>& y_train) {
        map<string, int> res;
        for (int i = 0; i < (int)vectorized_x.size(); i++)
            for (int j = 0; j < (int)vectorized_x[0].size(); j++)
                res[y_train[i]] += vectorized_x[i][j];
        return res;
    }

    void calculate_phi(const vector<vector<int>>& vectorized_x, const vector<string>& y_train) {
        map<string, int> total_words = calculate_word_count(vectorized_x, y_train);
        int feature_number = 0;
        for (int i = 0; i < vocab_size; i++) {
            map<string, int> auxiliary_dict;
            for (auto& cls : class_ratio) auxiliary_dict[cls.first] = 0;
            for (int j = 0; j < (int)vectorized_x.size(); j++)
                auxiliary_dict[y_train[j]] += vectorized_x[j][i];
            for (auto& element : auxiliary_dict)
                phi_dict[{element.first, feature_number}] = (double)(element.second + 1) / (total_words[element.first] + vocab_size);
            feature_number++;
        }
    }

    static pair<size_t,size_t> _mnb_obj_range(const string& s, const string& key) {
        string pat = "\"" + key + "\":{";
        size_t open = s.find(pat) + pat.size() - 1;
        int depth = 1; size_t i = open + 1;
        while (i < s.size() && depth > 0) { if (s[i]=='{') depth++; else if (s[i]=='}') depth--; i++; }
        return {open + 1, i - 1};
    }

    static map<string,int> _mnb_int_map(const string& s, size_t st, size_t en) {
        map<string,int> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stoi(s.substr(pos), &n); pos += n;
        }
        return res;
    }

    static map<string,double> _mnb_dbl_map(const string& s, size_t st, size_t en) {
        map<string,double> res;
        size_t pos = st;
        while (pos < en) {
            size_t ks = s.find('"', pos); if (ks >= en) break; ks++;
            size_t ke = s.find('"', ks);  if (ke >= en) break;
            string key = s.substr(ks, ke - ks);
            pos = s.find(':', ke) + 1;
            size_t n; res[key] = stod(s.substr(pos), &n); pos += n;
        }
        return res;
    }
};
