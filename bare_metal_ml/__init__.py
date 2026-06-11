import sys
import types as _types

from bare_metal_ml._cpp import (
    # algorithms
    GDA,
    LinearRegression,
    LogisticRegression,
    KNN, KDTree,
    GaussianNaiveBayes, BernoulliNaiveBayes, MultinomialNaiveBayes,
    # neural network
    Network, Adam, SGD,
    FunctionType, ActivationFunction,
    ReLU, Sigmoid, Tanh,
    # autograd
    Element, Scalar, Matrix,
    # distance metrics
    euclidean, manhattan, cosine,
    # linalg — all functions, core + utilities
    matrix_with_matrix_multiplication,
    matrix_addition_and_sub,
    scalar_multiply_matrix,
    element_wise_multiplication,
    element_wise_division_two_matrices,
    element_wise_roots,
    transpose_matrix,
    ReLU_derivative,
    sum_across_column,
    matrix_product_from_vector_and_transpose,
    calculate_vector,
    LU_decomposition,
    calculate_determinant,
    matrix_inverse,
    matrix_product_with_matrix_and_vector,
    scalar_product_from_transpose_and_vector,
    regularize,
)

# bare_metal_ml.linalg — pure C++ namespace, no Python layer
linalg = _types.ModuleType("bare_metal_ml.linalg")
for _name in [
    "matrix_with_matrix_multiplication", "matrix_addition_and_sub",
    "scalar_multiply_matrix", "element_wise_multiplication",
    "element_wise_division_two_matrices", "element_wise_roots",
    "transpose_matrix", "ReLU_derivative", "sum_across_column",
    "matrix_product_from_vector_and_transpose", "calculate_vector",
    "LU_decomposition", "calculate_determinant", "matrix_inverse",
    "matrix_product_with_matrix_and_vector",
    "scalar_product_from_transpose_and_vector", "regularize",
]:
    setattr(linalg, _name, globals()[_name])
sys.modules["bare_metal_ml.linalg"] = linalg

__all__ = [
    "GDA",
    "LinearRegression",
    "LogisticRegression",
    "KNN", "KDTree", "euclidean", "manhattan", "cosine",
    "GaussianNaiveBayes", "BernoulliNaiveBayes", "MultinomialNaiveBayes",
    "linalg",
    "Network", "Adam", "SGD",
    "FunctionType", "ActivationFunction", "ReLU", "Sigmoid", "Tanh",
    "Element", "Scalar", "Matrix",
]
