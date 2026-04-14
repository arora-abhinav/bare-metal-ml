#pragma once
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Vector operations
Vector subtract_vectors(const Vector& v1, const Vector& v2);
double dot_product(const Vector& v1, const Vector& v2, int dimension);

// Matrix operations
Matrix outer_product(const Vector& v, int dimension);
Vector matrix_vector_product(const Matrix& matrix, const Vector& vector, int dimension);

// LU Decomposition (Doolittle's method)
std::pair<Matrix, Matrix> lu_decomposition(const Matrix& matrix, int dimension);

// Matrix inverse via forward/backward substitution
Matrix matrix_inverse(const Matrix& lower, const Matrix& upper, int dimension);

// Determinant from upper triangular matrix
double calculate_determinant(const Matrix& upper, int dimension);

// Regularize covariance matrix to prevent division by zero
Matrix regularize(const Matrix& matrix, int dimension, double epsilon = 1e-6);
