import math
from bare_metal_ml import linalg


class GDA:
    """
    Gaussian Discriminant Analysis — generative binary classifier.

    Models p(x|y) as a multivariate Gaussian with a shared covariance matrix
    and uses Bayes' rule to classify. Parameters are fit via maximum likelihood.

    Parameters
    ----------
    positive_class : any
        The label treated as y=1. Inferred from data if not provided.

    Example
    -------
    >>> from bare_metal_ml.gda import GDA
    >>> model = GDA()
    >>> model.fit(x_train, y_train)
    >>> print(model.accuracy(x_test, y_test))
    """

    def __init__(self, positive_class=None):
        self.positive_class = positive_class
        self._negative_class = None
        self.phi = None
        self.mu_zero = None
        self.mu_one = None
        self.covariance = None
        self._dimension = None

    def fit(self, x_train, y_train):
        classes = list(set(y_train))
        if self.positive_class is None:
            self.positive_class = classes[0]
        self._negative_class = [c for c in classes if c != self.positive_class][0]

        n = len(x_train)
        self._dimension = len(x_train[0])
        self.phi = self._compute_phi(y_train, n)
        self.mu_zero = self._compute_mu(x_train, y_train, n, positive=False)
        self.mu_one = self._compute_mu(x_train, y_train, n, positive=True)
        self.covariance = self._compute_covariance(x_train, y_train, n)

    def _compute_phi(self, y_train, n):
        count = 0
        for i in range(n):
            if y_train[i] == self.positive_class:
                count += 1
        return count / n

    def _compute_mu(self, x_train, y_train, n, positive):
        target = self.positive_class if positive else self._negative_class
        proportion = self.phi if positive else (1 - self.phi)
        init_arr = [0.0] * len(x_train[0])
        for i in range(n):
            if y_train[i] == target:
                for j in range(len(init_arr)):
                    init_arr[j] += x_train[i][j]
        for k in range(len(init_arr)):
            init_arr[k] /= (proportion * n)
        return init_arr

    def _compute_covariance(self, x_train, y_train, n):
        d = self._dimension
        res = [[0.0] * d for _ in range(d)]
        for i in range(n):
            mu = self.mu_one if y_train[i] == self.positive_class else self.mu_zero
            vector = linalg.calculate_vector(list(x_train[i]), mu)
            mat = linalg.matrix_product_from_vector_and_transpose(d, vector)
            for j in range(d):
                for k in range(d):
                    res[j][k] += mat[j][k] / n
        return res

    def _multivariate_gaussian(self, mean, covariance, x):
        d = self._dimension
        resultant = linalg.calculate_vector(list(x), mean)
        cov_copy = [row[:] for row in covariance]
        cov_copy = linalg.regularize(cov_copy, d)
        lower, upper = linalg.LU_decomposition(cov_copy, d)
        inverse = linalg.matrix_inverse(lower, upper, d)
        vector_prod = linalg.matrix_product_with_matrix_and_vector(inverse, resultant, d, d)
        scalar_prod = linalg.scalar_product_from_transpose_and_vector(resultant, vector_prod)
        determinant = linalg.calculate_determinant(upper, d)
        exponential_term = math.exp(-0.5 * scalar_prod)
        normalizer = 1.0 / ((2 * math.pi) ** (d / 2) * math.sqrt(abs(determinant)))
        return normalizer * exponential_term

    def predict_one(self, x):
        log_p_one = math.log(max(self._multivariate_gaussian(self.mu_one, self.covariance, x), 1e-300)) + math.log(self.phi)
        log_p_zero = math.log(max(self._multivariate_gaussian(self.mu_zero, self.covariance, x), 1e-300)) + math.log(1 - self.phi)
        return self.positive_class if log_p_one >= log_p_zero else self._negative_class

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
