import math


class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient ascent on log-likelihood.

    Model: h(x) = sigmoid(theta^T x)  where x_0 = 1 (bias term)
    Update: theta_k += lr * (1/m) * sum((y - h(x)) * x_k)

    Parameters
    ----------
    positive_class : any
        The label mapped to y=1. Inferred from data if not provided.

    Example
    -------
    >>> from bare_metal_ml.logistic_regression import LogisticRegression
    >>> model = LogisticRegression()
    >>> model.fit(x_train, y_train, learning_rate=0.001, iterations=1000)
    >>> print(model.accuracy(x_test, y_test))
    """

    def __init__(self, positive_class=None):
        self.positive_class = positive_class
        self.thetas = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def _hypothesis(self, thetas, features):
        z = thetas[0]
        for k in range(len(features)):
            z += thetas[k + 1] * features[k]
        return self._sigmoid(z)

    def fit(self, x_train, y_train, learning_rate=0.001, iterations=1000):
        classes = list(set(y_train))
        if self.positive_class is None:
            self.positive_class = classes[0]

        n = len(x_train)
        n_features = len(x_train[0])
        thetas = [0.0] * (n_features + 1)

        for _ in range(iterations):
            updates = [0.0] * (n_features + 1)
            for i in range(n):
                h = self._hypothesis(thetas, x_train[i])
                y = 1 if y_train[i] == self.positive_class else 0
                error = y - h
                updates[0] += error
                for k in range(n_features):
                    updates[k + 1] += error * x_train[i][k]

            for k in range(len(thetas)):
                thetas[k] += (learning_rate * updates[k]) / n

        self.thetas = thetas

    def predict_proba(self, x_test):
        probabilities = []
        for i in range(len(x_test)):
            probabilities.append(self._hypothesis(self.thetas, x_test[i]))
        return probabilities

    def predict(self, x_test, threshold=0.5):
        probabilities = self.predict_proba(x_test)
        predictions = []
        for i in range(len(probabilities)):
            if probabilities[i] >= threshold:
                predictions.append(self.positive_class)
            else:
                predictions.append(None)
        return predictions

    def accuracy(self, x_test, y_test, threshold=0.5):
        predictions = self.predict(x_test, threshold)
        correct = 0
        for i in range(len(predictions)):
            actual = self.positive_class if y_test[i] == self.positive_class else None
            if predictions[i] == actual:
                correct += 1
        return (correct / len(y_test)) * 100.0
