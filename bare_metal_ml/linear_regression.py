class LinearRegression:
    """
    Univariate linear regression trained with batch gradient descent.

    Model: h(x) = theta0 + theta1 * x
    Loss:  J = 0.5 * sum((h(x_i) - y_i)^2)

    Example
    -------
    >>> from bare_metal_ml.linear_regression import LinearRegression
    >>> model = LinearRegression()
    >>> model.fit(x_train, y_train, learning_rate=0.01, iterations=100)
    >>> print(model.predict([5.0, 7.5]))
    """

    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0

    def _hypothesis(self, theta0, theta1, x):
        return theta0 + theta1 * x

    def fit(self, x_train, y_train, learning_rate=0.01, iterations=100):
        theta0 = 0.0
        theta1 = 0.0
        n = len(x_train)

        for _ in range(iterations):
            sum_error_theta0 = 0.0
            sum_error_theta1 = 0.0

            for i in range(n):
                h = self._hypothesis(theta0, theta1, x_train[i])
                error = h - y_train[i]
                sum_error_theta0 += error
                sum_error_theta1 += error * x_train[i]

            theta0 -= (learning_rate / n) * sum_error_theta0
            theta1 -= (learning_rate / n) * sum_error_theta1

        self.theta0 = theta0
        self.theta1 = theta1

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self._hypothesis(self.theta0, self.theta1, x_test[i]))
        return predictions

    def mse(self, x_test, y_test):
        predictions = self.predict(x_test)
        total = 0.0
        for i in range(len(predictions)):
            total += (predictions[i] - y_test[i]) ** 2
        return total / len(y_test)
