import math
import re


# ── shared text utilities ─────────────────────────────────────────────────────

def _tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

def _build_vocab(x_train, train_tokens, vocab_size):
    counter_map = {}
    for i in range(len(x_train)):
        for token in train_tokens[i]:
            if token not in counter_map:
                counter_map[token] = 1
            else:
                counter_map[token] += 1
    sorted_map = sorted(counter_map.items(), key=lambda item: item[1], reverse=True)
    top_words = sorted_map[:vocab_size]
    final_map = {}
    counter = 0
    for element in top_words:
        word, number = element
        final_map[word] = counter
        counter += 1
    return final_map

def _calculate_class_ratio(y_train):
    m = {}
    for i in range(len(y_train)):
        if y_train[i] not in m:
            m[y_train[i]] = 1
        else:
            m[y_train[i]] += 1
    return m

def _vectorize(x, vocab_size, tokens, vocab_list, count):
    vectorized = [[0] * vocab_size for _ in range(len(x))]
    for i in range(len(x)):
        for token in tokens[i]:
            if token in vocab_list:
                if not count:
                    vectorized[i][vocab_list[token]] = 1
                else:
                    vectorized[i][vocab_list[token]] += 1
    return vectorized


# ── Gaussian Naive Bayes ──────────────────────────────────────────────────────

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features.

    Per-class mean and variance are computed per feature. Variance is calculated
    as E[X^2] - E[X]^2, avoiding a second pass over the data. Log probabilities
    are used during prediction to prevent numerical underflow.

    Example
    -------
    >>> from bare_metal_ml.naive_bayes import GaussianNaiveBayes
    >>> model = GaussianNaiveBayes()
    >>> model.fit(x_train, y_train)
    >>> print(model.accuracy(x_test, y_test))
    """

    def fit(self, x_train, y_train):
        self._class_ratio = _calculate_class_ratio(y_train)
        self._class_numbering = {}
        count = 0
        for i in range(len(y_train)):
            if y_train[i] not in self._class_numbering:
                self._class_numbering[y_train[i]] = count
                count += 1
        self._mean = {}
        self._variance = {}
        self._n = len(y_train)
        self._calculate_mean(x_train, y_train)
        self._calculate_variance(x_train, y_train)

    def _calculate_mean(self, x_train, y_train):
        feature_number = 0
        for i in range(len(x_train[0])):
            auxiliary_dict = {}
            for j in range(len(x_train)):
                if y_train[j] not in auxiliary_dict:
                    auxiliary_dict[y_train[j]] = x_train[j][i] / self._class_ratio[y_train[j]]
                else:
                    auxiliary_dict[y_train[j]] += x_train[j][i] / self._class_ratio[y_train[j]]
            for element in auxiliary_dict:
                self._mean[(element, feature_number)] = auxiliary_dict[element]
            feature_number += 1

    def _calculate_variance(self, x_train, y_train):
        # Var(X) = E[X^2] - E[X]^2
        feature_number = 0
        for i in range(len(x_train[0])):
            auxiliary_dict = {}
            for j in range(len(x_train)):
                if y_train[j] not in auxiliary_dict:
                    auxiliary_dict[y_train[j]] = (x_train[j][i] ** 2) / self._class_ratio[y_train[j]]
                else:
                    auxiliary_dict[y_train[j]] += (x_train[j][i] ** 2) / self._class_ratio[y_train[j]]
            for element in auxiliary_dict:
                self._variance[(element, feature_number)] = auxiliary_dict[element] - self._mean[(element, feature_number)] ** 2
            feature_number += 1

    def _gaussian_probability(self, x, mean, variance):
        variance = max(variance, 1e-10)
        return max((1 / math.sqrt(2 * math.pi * variance)) * math.exp(-((x - mean) ** 2) / (2 * variance)), 1e-300)

    def predict_one(self, x):
        prediction = ''
        highest_probability = float('-inf')
        for class_type in self._class_numbering:
            probability_sum = math.log(self._class_ratio[class_type] / self._n)
            for j in range(len(x)):
                probability_sum += math.log(self._gaussian_probability(x[j], self._mean[(class_type, j)], self._variance[(class_type, j)]))
            if probability_sum > highest_probability:
                prediction = class_type
                highest_probability = probability_sum
        return prediction

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self.predict_one(x_test[i]))
        return predictions

    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return (correct / len(y_test)) * 100.0


# ── Bernoulli Naive Bayes ─────────────────────────────────────────────────────

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes text classifier.

    Each document is represented as a binary vector of word presence.
    Laplace smoothing: phi(j,k) = (count + 1) / (class_count + 2).
    Log probabilities are used to avoid underflow.

    Parameters
    ----------
    vocab_size : int
        Top-N most frequent words to keep as features.

    Example
    -------
    >>> from bare_metal_ml.naive_bayes import BernoulliNaiveBayes
    >>> model = BernoulliNaiveBayes(vocab_size=1000)
    >>> model.fit(x_train, y_train)   # x_train: list of raw text strings
    >>> print(model.accuracy(x_test, y_test))
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def fit(self, x_train, y_train):
        train_tokens = [_tokenize(x_train[i]) for i in range(len(x_train))]
        self._vocab = _build_vocab(x_train, train_tokens, self.vocab_size)
        vectorized_inputs = _vectorize(x_train, self.vocab_size, train_tokens, self._vocab, False)
        self._class_ratio = _calculate_class_ratio(y_train)
        self._n = len(y_train)
        self._phi = {}
        self._calculate_phi(vectorized_inputs, y_train)

    def _calculate_phi(self, vectorized_x, y_train):
        feature_number = 0
        for i in range(len(vectorized_x[0])):
            auxiliary_dict = {element: 0 for element in self._class_ratio}
            for j in range(len(vectorized_x)):
                auxiliary_dict[y_train[j]] += vectorized_x[j][i]
            for element in auxiliary_dict:
                # Laplace smoothing
                self._phi[(element, feature_number)] = (auxiliary_dict[element] + 1) / (self._class_ratio[element] + 2)
            feature_number += 1

    def _bernoulli_dist(self, x, phi):
        return (phi ** x) * ((1 - phi) ** (1 - x))

    def predict_one(self, doc):
        tokens = _tokenize(doc)
        vector = _vectorize([doc], self.vocab_size, [tokens], self._vocab, False)[0]
        prediction = ''
        best_score = float('-inf')
        for element in self._class_ratio:
            s = math.log(self._class_ratio[element] / self._n)
            for i in range(len(vector)):
                s += math.log(self._bernoulli_dist(vector[i], self._phi[(element, i)]))
            if s > best_score:
                best_score = s
                prediction = element
        return prediction

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self.predict_one(x_test[i]))
        return predictions

    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return (correct / len(y_test)) * 100.0


# ── Multinomial Naive Bayes ───────────────────────────────────────────────────

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes text classifier.

    Each document is represented as a word-count vector. The phi denominator
    normalizes by total word count per class, not example count — the key
    distinction from Bernoulli NB. Laplace smoothing prevents log(0).

    Parameters
    ----------
    vocab_size : int
        Top-N most frequent words to keep as features.

    Example
    -------
    >>> from bare_metal_ml.naive_bayes import MultinomialNaiveBayes
    >>> model = MultinomialNaiveBayes(vocab_size=1000)
    >>> model.fit(x_train, y_train)   # x_train: list of raw text strings
    >>> print(model.accuracy(x_test, y_test))
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def fit(self, x_train, y_train):
        train_tokens = [_tokenize(x_train[i]) for i in range(len(x_train))]
        self._vocab = _build_vocab(x_train, train_tokens, self.vocab_size)
        vectorized_inputs = _vectorize(x_train, self.vocab_size, train_tokens, self._vocab, True)
        self._class_ratio = _calculate_class_ratio(y_train)
        self._n = len(y_train)
        self._phi = {}
        self._calculate_phi(vectorized_inputs, y_train)

    def _calculate_word_count(self, vectorized_x, y_train):
        res_dict = {}
        for i in range(len(vectorized_x)):
            for j in range(len(vectorized_x[0])):
                if y_train[i] not in res_dict:
                    res_dict[y_train[i]] = vectorized_x[i][j]
                else:
                    res_dict[y_train[i]] += vectorized_x[i][j]
        return res_dict

    def _calculate_phi(self, vectorized_x, y_train):
        feature_number = 0
        total_words = self._calculate_word_count(vectorized_x, y_train)
        for i in range(len(vectorized_x[0])):
            auxiliary_dict = {element: 0 for element in self._class_ratio}
            for j in range(len(vectorized_x)):
                auxiliary_dict[y_train[j]] += vectorized_x[j][i]
            for element in auxiliary_dict:
                # Laplace smoothing: normalize by total word count per class
                self._phi[(element, feature_number)] = (auxiliary_dict[element] + 1) / (total_words[element] + self.vocab_size)
            feature_number += 1

    def predict_one(self, doc):
        tokens = _tokenize(doc)
        vector = _vectorize([doc], self.vocab_size, [tokens], self._vocab, True)[0]
        prediction = ''
        best_score = float('-inf')
        for element in self._class_ratio:
            s = math.log(self._class_ratio[element] / self._n)
            for i in range(len(vector)):
                s += vector[i] * math.log(self._phi[(element, i)])
            if s > best_score:
                best_score = s
                prediction = element
        return prediction

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self.predict_one(x_test[i]))
        return predictions

    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return (correct / len(y_test)) * 100.0
