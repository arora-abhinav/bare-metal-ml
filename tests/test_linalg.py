import math
import pytest
from bare_metal_ml import linalg


def _approx_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def _mat_approx_equal(m1, m2, tol=1e-9):
    for r in range(len(m1)):
        for c in range(len(m1[0])):
            if abs(m1[r][c] - m2[r][c]) > tol:
                return False
    return True


# ── element_wise_roots ───────────────────────────────────────────────────────

def test_element_wise_roots_sqrt():
    """square-root of a matrix; directly validates the ^ → ** fix."""
    result = linalg.element_wise_roots([[4.0, 9.0], [16.0, 25.0]], 2)
    assert _approx_equal(result[0][0], 2.0)
    assert _approx_equal(result[0][1], 3.0)
    assert _approx_equal(result[1][0], 4.0)
    assert _approx_equal(result[1][1], 5.0)


def test_element_wise_roots_cube():
    result = linalg.element_wise_roots([[8.0, 27.0]], 3)
    assert _approx_equal(result[0][0], 2.0)
    assert _approx_equal(result[0][1], 3.0)


# ── scalar_multiply_matrix ───────────────────────────────────────────────────

def test_scalar_multiply_does_not_mutate():
    original = [[1.0, 2.0], [3.0, 4.0]]
    copy = [row[:] for row in original]
    result = linalg.scalar_multiply_matrix(original, 3.0)
    assert result == [[3.0, 6.0], [9.0, 12.0]]
    assert original == copy  # input must not be mutated


def test_scalar_multiply_values():
    result = linalg.scalar_multiply_matrix([[2.0, -1.0]], 2.0)
    assert _approx_equal(result[0][0], 4.0)
    assert _approx_equal(result[0][1], -2.0)


# ── matrix_with_matrix_multiplication ────────────────────────────────────────

def test_matmul_identity():
    I = [[1.0, 0.0], [0.0, 1.0]]
    A = [[3.0, 7.0], [2.0, 5.0]]
    result = linalg.matrix_with_matrix_multiplication(I, A)
    assert _mat_approx_equal(result, A)


def test_matmul_known():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    result = linalg.matrix_with_matrix_multiplication(A, B)
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert _mat_approx_equal(result, expected)


# ── matrix_addition_and_sub ───────────────────────────────────────────────────

def test_matrix_add():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    result = linalg.matrix_addition_and_sub(A, B, "add")
    assert _mat_approx_equal(result, [[6.0, 8.0], [10.0, 12.0]])


def test_matrix_sub():
    A = [[5.0, 6.0], [7.0, 8.0]]
    B = [[1.0, 2.0], [3.0, 4.0]]
    result = linalg.matrix_addition_and_sub(A, B, "sub")
    assert _mat_approx_equal(result, [[4.0, 4.0], [4.0, 4.0]])


# ── transpose_matrix ─────────────────────────────────────────────────────────

def test_transpose():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = linalg.transpose_matrix(A)
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert result == expected


def test_transpose_double():
    A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert linalg.transpose_matrix(linalg.transpose_matrix(A)) == A


# ── element_wise_multiplication ───────────────────────────────────────────────

def test_element_wise_mult():
    A = [[2.0, 3.0], [4.0, 5.0]]
    B = [[1.0, 2.0], [3.0, 4.0]]
    result = linalg.element_wise_multiplication(A, B)
    assert _mat_approx_equal(result, [[2.0, 6.0], [12.0, 20.0]])


# ── LU decomposition & inverse ────────────────────────────────────────────────

def test_lu_roundtrip():
    """L @ U should reproduce the original matrix (up to tolerance)."""
    A = [[4.0, 3.0], [6.0, 3.0]]
    lower, upper = linalg.LU_decomposition([row[:] for row in A], 2)
    reconstructed = linalg.matrix_with_matrix_multiplication(lower, upper)
    assert _mat_approx_equal(reconstructed, A, tol=1e-7)


def test_matrix_inverse():
    """A @ inv(A) should be close to the identity matrix."""
    A = [[2.0, 1.0], [5.0, 3.0]]
    lower, upper = linalg.LU_decomposition([row[:] for row in A], 2)
    inv = linalg.matrix_inverse(lower, upper, 2)
    product = linalg.matrix_with_matrix_multiplication(A, inv)
    I = [[1.0, 0.0], [0.0, 1.0]]
    assert _mat_approx_equal(product, I, tol=1e-7)
