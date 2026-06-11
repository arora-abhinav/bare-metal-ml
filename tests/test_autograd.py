import pytest
from bare_metal_ml.autograd import Scalar, Matrix


# ── Scalar gradient tests ─────────────────────────────────────────────────────

def test_scalar_add_backward():
    a = Scalar(3.0)
    b = Scalar(2.0)
    c = a + b
    c.gradient = 1.0
    c.backprop()
    assert a.gradient == 1.0
    assert b.gradient == 1.0


def test_scalar_mul_backward():
    a = Scalar(3.0)
    b = Scalar(4.0)
    c = a * b
    c.gradient = 1.0
    c.backprop()
    assert a.gradient == 4.0  # dc/da = b = 4
    assert b.gradient == 3.0  # dc/db = a = 3


def test_scalar_chain_backward():
    """z = (a * b) + (c * d); gradients propagate through both branches."""
    a, b, c, d = Scalar(2.0), Scalar(3.0), Scalar(4.0), Scalar(5.0)
    z = (a * b) + (c * d)
    z.gradient = 1.0
    z.backprop()
    assert a.gradient == 3.0
    assert b.gradient == 2.0
    assert c.gradient == 5.0
    assert d.gradient == 4.0


def test_scalar_sub_backward():
    a = Scalar(5.0)
    b = Scalar(2.0)
    c = a - b
    c.gradient = 1.0
    c.backprop()
    assert a.gradient == 1.0
    assert b.gradient == -1.0


def test_scalar_div_backward():
    a = Scalar(6.0)
    b = Scalar(3.0)
    c = a / b
    c.gradient = 1.0
    c.backprop()
    assert abs(a.gradient - 1.0 / 3.0) < 1e-9   # dc/da = 1/b
    assert abs(b.gradient - (-6.0 / 9.0)) < 1e-9  # dc/db = -a/b^2


def test_scalar_relu_positive():
    a = Scalar(3.0)
    r = a.relu()
    r.gradient = 1.0
    r.backprop()
    assert a.gradient == 1.0


def test_scalar_relu_negative():
    a = Scalar(-2.0)
    r = a.relu()
    r.gradient = 1.0
    r.backprop()
    assert a.gradient == 0.0


def test_scalar_sigmoid_value():
    a = Scalar(0.0)
    s = a.sigmoid()
    assert abs(s.digit - 0.5) < 1e-9


def test_scalar_sigmoid_backward():
    a = Scalar(0.0)
    s = a.sigmoid()
    s.gradient = 1.0
    s.backprop()
    # sigmoid'(0) = 0.5 * 0.5 = 0.25
    assert abs(a.gradient - 0.25) < 1e-9


# ── Matrix gradient tests ─────────────────────────────────────────────────────

def test_matrix_add_backward():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = Matrix([[5.0, 6.0], [7.0, 8.0]])
    A.gradient = [[0.0, 0.0], [0.0, 0.0]]
    B.gradient = [[0.0, 0.0], [0.0, 0.0]]
    C = A + B
    C.gradient = [[1.0, 1.0], [1.0, 1.0]]
    C.backprop()
    # dC/dA = I, dC/dB = I
    assert A.gradient == [[1.0, 1.0], [1.0, 1.0]]
    assert B.gradient == [[1.0, 1.0], [1.0, 1.0]]


def test_matrix_relu_backward():
    """ReLU passes gradient only where the input was positive."""
    M = Matrix([[2.0, -1.0], [-3.0, 4.0]])
    M.gradient = [[0.0, 0.0], [0.0, 0.0]]
    R = M.relu()
    R.gradient = [[1.0, 1.0], [1.0, 1.0]]
    R.backprop()
    assert M.gradient[0][0] == 1.0  # positive → passes gradient
    assert M.gradient[0][1] == 0.0  # negative → blocked
    assert M.gradient[1][0] == 0.0  # negative → blocked
    assert M.gradient[1][1] == 1.0  # positive → passes gradient


def test_matrix_mul_forward():
    """Matrix multiplication produces the correct output matrix."""
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = Matrix([[1.0, 0.0], [0.0, 1.0]])  # identity
    C = A * B
    assert C.matrix == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_backprop_runs():
    """Full forward + backward pass completes without error and produces non-zero gradients."""
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = Matrix([[1.0, 1.0], [1.0, 1.0]])
    A.gradient = [[0.0, 0.0], [0.0, 0.0]]
    B.gradient = [[0.0, 0.0], [0.0, 0.0]]
    C = A * B
    C.gradient = [[1.0, 1.0], [1.0, 1.0]]
    C.backprop()
    # At least one gradient entry should be non-zero
    assert any(A.gradient[r][c] != 0.0 for r in range(2) for c in range(2))
    assert any(B.gradient[r][c] != 0.0 for r in range(2) for c in range(2))
