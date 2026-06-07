import math
import random
from enum import Enum
from bare_metal_ml import linalg


class FunctionType(Enum):
    RELU = 1
    SIGMOID = 2


class Layer:
    def __init__(self, neuron_num, input_size, function_type):
        self.function_type = function_type
        self.neuron_num = neuron_num
        self.input_size = input_size
        he_std = math.sqrt(2 / input_size)
        self.parameters = [[random.gauss(0, he_std) for _ in range(input_size)] for _ in range(neuron_num)]
        self.bias = [[0] for _ in range(neuron_num)]
        self.z = None
        self.a = None
        self.first_moment_weight = [[0] * input_size for _ in range(neuron_num)]
        self.second_moment_weight = [[0] * input_size for _ in range(neuron_num)]
        self.first_moment_bias = [[0] for _ in range(neuron_num)]
        self.second_moment_bias = [[0] for _ in range(neuron_num)]
        self.dropout_mask = None

    def _activate(self, x):
        if self.function_type == FunctionType.RELU:
            return x if x > 0 else 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def linearize(self, parameters, input, bias):
        product = linalg.matrix_with_matrix_multiplication(parameters, input)
        cols = len(product[0])
        broadcast = [[bias[r][0]] * cols for r in range(len(product))]
        return linalg.matrix_addition_and_sub(product, broadcast, "add")

    def hypothesis(self, linear, softmax=False):
        result = [[self._activate(linear[r][c]) for c in range(len(linear[0]))] for r in range(len(linear))]
        if not softmax:
            return result
        for col in range(len(linear[0])):
            exp_sum = sum(math.exp(result[r][col]) for r in range(len(linear)))
            for r in range(len(linear)):
                result[r][col] = math.exp(result[r][col]) / exp_sum
        return result


class Adam:
    beta1 = 0.9
    beta2 = 0.999

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.t = 0

    def update(self, layers, dw, db):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for i in range(len(layers)):
            layers[i].first_moment_weight = linalg.matrix_addition_and_sub(
                linalg.scalar_multiply_matrix(layers[i].first_moment_weight, self.beta1),
                linalg.scalar_multiply_matrix(dw[i], 1 - self.beta1), "add")
            layers[i].second_moment_weight = linalg.matrix_addition_and_sub(
                linalg.scalar_multiply_matrix(layers[i].second_moment_weight, self.beta2),
                linalg.scalar_multiply_matrix(linalg.element_wise_multiplication(dw[i], dw[i]), 1 - self.beta2), "add")
            layers[i].first_moment_bias = linalg.matrix_addition_and_sub(
                linalg.scalar_multiply_matrix(layers[i].first_moment_bias, self.beta1),
                linalg.scalar_multiply_matrix(db[i], 1 - self.beta1), "add")
            layers[i].second_moment_bias = linalg.matrix_addition_and_sub(
                linalg.scalar_multiply_matrix(layers[i].second_moment_bias, self.beta2),
                linalg.scalar_multiply_matrix(linalg.element_wise_multiplication(db[i], db[i]), 1 - self.beta2), "add")

            m_hat_w = linalg.scalar_multiply_matrix(layers[i].first_moment_weight, 1 / bc1)
            v_hat_w = linalg.scalar_multiply_matrix(layers[i].second_moment_weight, 1 / bc2)
            m_hat_b = linalg.scalar_multiply_matrix(layers[i].first_moment_bias, 1 / bc1)
            v_hat_b = linalg.scalar_multiply_matrix(layers[i].second_moment_bias, 1 / bc2)

            root_w = linalg.element_wise_roots(v_hat_w, 2)
            eps_w = [[1e-8] * len(root_w[0]) for _ in range(len(root_w))]
            step_w = linalg.scalar_multiply_matrix(
                linalg.element_wise_division_two_matrices(m_hat_w, linalg.matrix_addition_and_sub(root_w, eps_w, "add")),
                self.learning_rate)
            layers[i].parameters = linalg.matrix_addition_and_sub(layers[i].parameters, step_w, "sub")

            root_b = linalg.element_wise_roots(v_hat_b, 2)
            eps_b = [[1e-8] * len(root_b[0]) for _ in range(len(root_b))]
            step_b = linalg.scalar_multiply_matrix(
                linalg.element_wise_division_two_matrices(m_hat_b, linalg.matrix_addition_and_sub(root_b, eps_b, "add")),
                self.learning_rate)
            layers[i].bias = linalg.matrix_addition_and_sub(layers[i].bias, step_b, "sub")


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers, dw, db):
        for i in range(len(layers)):
            layers[i].parameters = linalg.matrix_addition_and_sub(
                layers[i].parameters, linalg.scalar_multiply_matrix(dw[i], self.learning_rate), "sub")
            layers[i].bias = linalg.matrix_addition_and_sub(
                layers[i].bias, linalg.scalar_multiply_matrix(db[i], self.learning_rate), "sub")


class Network:
    """
    Fully-connected neural network trained with mini-batch gradient descent.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer (excluding input). e.g. [128, 64, 10]
    input_size : int
        Dimensionality of each training example.
    function_type : FunctionType
        Activation function for hidden layers (RELU or SIGMOID).
    optimizer : Adam | SGD
        Optimizer instance.
    dropout_rate : float
        Fraction of neurons to drop during training (0 = no dropout).

    Example
    -------
    >>> from bare_metal_ml.neural_network import Network, Adam, FunctionType
    >>> net = Network([128, 64, 10], 784, FunctionType.RELU, Adam(0.001), dropout_rate=0.1)
    >>> net.train(x_train, y_train, epochs=30, batch_size=256)
    >>> print(net.accuracy(x_test, y_test))
    """

    def __init__(self, layer_sizes, input_size, function_type, optimizer, dropout_rate=0.0):
        self.initial_input = None
        self.layers = []
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.current_batch = None
        for i, size in enumerate(layer_sizes):
            fan_in = input_size if i == 0 else layer_sizes[i - 1]
            self.layers.append(Layer(size, fan_in, function_type))

    def _cross_entropy(self, y_hat, y, epsilon=1e-7):
        clipped = [max(epsilon, min(1.0 - epsilon, v)) for v in y_hat]
        return -sum(math.log(clipped[i]) * y[i] for i in range(len(y)))

    def _feedforward(self, layer_index, input, training=True):
        if layer_index >= len(self.layers):
            return
        layer = self.layers[layer_index]
        is_last = layer_index == len(self.layers) - 1
        layer.z = layer.linearize(layer.parameters, input, layer.bias)
        output = layer.hypothesis(layer.z, softmax=is_last)
        if training and self.dropout_rate > 0.0 and not is_last:
            mask = [[1.0 if random.random() >= self.dropout_rate else 0.0
                     for _ in range(len(output[0]))] for _ in range(len(output))]
            output = linalg.element_wise_multiplication(output, mask)
            output = linalg.scalar_multiply_matrix(output, 1.0 / (1.0 - self.dropout_rate))
            layer.dropout_mask = mask
        layer.a = output
        self._feedforward(layer_index + 1, output, training)

    def _total_loss(self, labels):
        out_T = linalg.transpose_matrix(self.layers[-1].a)
        lab_T = linalg.transpose_matrix(labels)
        return sum(self._cross_entropy(out_T[r], lab_T[r]) for r in range(len(out_T))) / len(lab_T)

    def _last_layer_backprop(self, labels, prev_layer=None):
        prev_a = linalg.transpose_matrix(prev_layer.a if prev_layer else self.current_batch)
        term = linalg.matrix_addition_and_sub(labels, self.layers[-1].a, "sub")
        res = linalg.scalar_multiply_matrix(linalg.matrix_with_matrix_multiplication(term, prev_a), -1 / len(labels[0]))
        return res, linalg.scalar_multiply_matrix(term, -1 / len(labels[0]))

    def _prev_layer_backprop(self, layer, next_layer, prev_product, prev_layer=None):
        W_T = linalg.transpose_matrix(next_layer.parameters)
        delta = linalg.matrix_with_matrix_multiplication(W_T, prev_product)
        product_two = linalg.element_wise_multiplication(delta, linalg.ReLU_derivative(layer.z))
        if self.dropout_rate > 0.0 and layer.dropout_mask:
            product_two = linalg.element_wise_multiplication(product_two, layer.dropout_mask)
        prev_a = linalg.transpose_matrix(prev_layer.a if prev_layer else self.current_batch)
        return linalg.matrix_with_matrix_multiplication(product_two, prev_a), product_two

    def train(self, x_train, y_train, epochs=30, batch_size=256):
        """Train the network with mini-batch gradient descent."""
        self.initial_input = x_train
        num_examples = len(x_train[0])
        for epoch in range(epochs):
            indices = list(range(num_examples))
            random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = num_examples // batch_size
            for b in range(num_batches):
                batch_idx = indices[b * batch_size: (b + 1) * batch_size]
                self.current_batch = [[x_train[r][j] for j in batch_idx] for r in range(len(x_train))]
                batch_labels = [[y_train[r][j] for j in batch_idx] for r in range(len(y_train))]
                self._feedforward(0, self.current_batch)
                epoch_loss += self._total_loss(batch_labels)
                dw = [None] * len(self.layers)
                db = [None] * len(self.layers)
                prev = self.layers[-2] if len(self.layers) > 1 else None
                dw[-1], bp = self._last_layer_backprop(batch_labels, prev)
                db[-1] = [[sum(bp[r])] for r in range(len(bp))]
                for i in range(len(self.layers) - 2, -1, -1):
                    prev = self.layers[i - 1] if i > 0 else None
                    dw[i], bp = self._prev_layer_backprop(self.layers[i], self.layers[i + 1], bp, prev)
                    db[i] = [[sum(bp[r])] for r in range(len(bp))]
                self.optimizer.update(self.layers, dw, db)
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}")

    def accuracy(self, x_test, y_test):
        """Run inference (dropout disabled) and return accuracy as a percentage."""
        self.dropout_rate, saved = 0.0, self.dropout_rate
        self._feedforward(0, x_test, training=False)
        self.dropout_rate = saved
        output = self.layers[-1].a
        correct = sum(
            1 for col in range(len(output[0]))
            if max(range(len(output)), key=lambda r: output[r][col]) ==
               max(range(len(y_test)), key=lambda r: y_test[r][col])
        )
        return correct / len(output[0]) * 100.0
