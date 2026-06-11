import io
import contextlib
import math
import random
import pytest

from bare_metal_ml.neural_network import (
    Network, Adam, SGD, FunctionType, ActivationFunction,
    ReLU, Sigmoid, Tanh, LossFunctions,
)
from bare_metal_ml.autograd import Matrix


# ── helpers ───────────────────────────────────────────────────────────────────

def _xor_data():
    """XOR: 2 features x 4 samples, 2 one-hot classes."""
    x = [[0.0, 0.0, 1.0, 1.0],
         [0.0, 1.0, 0.0, 1.0]]
    y = [[1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 1.0, 0.0]]
    return x, y


def _simple_separable_data():
    """Linearly separable 2-class data; easier to converge than XOR."""
    x = [[ 2.0,  1.9, -2.0, -1.9],
         [ 0.1, -0.1,  0.1, -0.1]]
    y = [[1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0]]
    return x, y


def _train_silent(net, y_train, epochs=50, batch_size=4):
    with contextlib.redirect_stdout(io.StringIO()):
        net.train_loop(epochs, y_train, batch_size)


def _get_loss(net, x, y):
    net.feedforward(0, Matrix(x))
    loss_fn = LossFunctions()
    return net.total_loss(net.layers[-1].a, loss_fn.cross_entropy_loss, y)


# ── import smoke test ─────────────────────────────────────────────────────────

def test_public_api_imports():
    from bare_metal_ml import Network, Adam, SGD, FunctionType, ActivationFunction, ReLU, Sigmoid, Tanh
    assert all(x is not None for x in [Network, Adam, SGD, FunctionType, ActivationFunction, ReLU, Sigmoid, Tanh])


# ── activation function unit tests ───────────────────────────────────────────

def test_relu_forward():
    r = ReLU()
    assert r.forward(3.0) == 3.0
    assert r.forward(-1.0) == 0.0
    assert r.forward(0.0) == 0.0


def test_relu_derivative():
    r = ReLU()
    assert r.derivative(3.0) == 1.0
    assert r.derivative(-1.0) == 0.0


def test_sigmoid_forward():
    s = Sigmoid()
    assert abs(s.forward(0.0) - 0.5) < 1e-9
    assert 0.0 < s.forward(-100.0) < 0.01
    assert s.forward(100.0) >= 0.99


def test_sigmoid_derivative():
    s = Sigmoid()
    # sigmoid'(0) = 0.25
    assert abs(s.derivative(0.0) - 0.25) < 1e-9


def test_tanh_forward():
    t = Tanh()
    assert abs(t.forward(0.0)) < 1e-9
    assert abs(t.forward(100.0) - 1.0) < 1e-6


def test_tanh_derivative():
    t = Tanh()
    # tanh'(0) = 1 - 0^2 = 1
    assert abs(t.derivative(0.0) - 1.0) < 1e-9


# ── network training tests ────────────────────────────────────────────────────

def test_loss_decreases_with_adam():
    random.seed(42)
    x, y = _simple_separable_data()
    net = Network(2, [8, 2], x, Adam(0.01), 0.0, function_type=FunctionType.RELU)
    initial_loss = _get_loss(net, x, y)
    _train_silent(net, y, epochs=100, batch_size=4)
    final_loss = _get_loss(net, x, y)
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_loss_decreases_with_sgd():
    random.seed(7)
    x, y = _simple_separable_data()
    net = Network(2, [8, 2], x, SGD(0.1), 0.0, function_type=FunctionType.SIGMOID)
    initial_loss = _get_loss(net, x, y)
    _train_silent(net, y, epochs=200, batch_size=4)
    final_loss = _get_loss(net, x, y)
    assert final_loss < initial_loss, f"SGD loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_accuracy_returns_valid_percentage():
    random.seed(0)
    x, y = _simple_separable_data()
    net = Network(2, [4, 2], x, Adam(0.01), 0.0, function_type=FunctionType.RELU)
    _train_silent(net, y, epochs=50, batch_size=4)
    acc = net.accuracy(x, y)
    assert 0.0 <= acc <= 100.0


def test_accuracy_after_training_beats_random():
    random.seed(3)
    x, y = _simple_separable_data()
    net = Network(2, [8, 2], x, Adam(0.01), 0.0, function_type=FunctionType.RELU)
    _train_silent(net, y, epochs=300, batch_size=4)
    acc = net.accuracy(x, y)
    assert acc > 50.0, f"Accuracy {acc:.1f}% is not better than random on separable data"


# ── custom activation injection tests ────────────────────────────────────────

def test_custom_activation_trains_without_error():
    random.seed(1)

    class LeakyReLU(ActivationFunction):
        def forward(self, x):
            return x if x > 0 else 0.01 * x
        def derivative(self, x):
            return 1.0 if x > 0 else 0.01

    x, y = _simple_separable_data()
    net = Network(2, [4, 2], x, Adam(0.01), 0.0, activation=LeakyReLU())
    _train_silent(net, y, epochs=10, batch_size=4)
    # No assertion needed beyond "it ran without error"


def test_custom_activation_loss_decreases():
    random.seed(99)

    class Swish(ActivationFunction):
        def forward(self, x):
            return x / (1.0 + math.exp(-x))
        def derivative(self, x):
            s = 1.0 / (1.0 + math.exp(-x))
            return s * (1.0 + x * (1.0 - s))

    x, y = _simple_separable_data()
    net = Network(2, [8, 2], x, Adam(0.01), 0.0, activation=Swish())
    initial_loss = _get_loss(net, x, y)
    _train_silent(net, y, epochs=100, batch_size=4)
    final_loss = _get_loss(net, x, y)
    assert final_loss < initial_loss, f"Custom activation loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_custom_activation_takes_priority_over_function_type():
    """When both activation and function_type are given, activation wins."""
    random.seed(5)
    call_log = []

    class TrackedActivation(ActivationFunction):
        def forward(self, x):
            call_log.append("forward")
            return max(0.0, x)
        def derivative(self, x):
            call_log.append("deriv")
            return 1.0 if x > 0 else 0.0

    x, y = _simple_separable_data()
    net = Network(2, [4, 2], x, Adam(0.01), 0.0,
                  activation=TrackedActivation(), function_type=FunctionType.SIGMOID)
    _train_silent(net, y, epochs=1, batch_size=4)
    assert len(call_log) > 0, "Custom activation was never called"
