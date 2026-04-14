#include "LinearAlgebra.h"
#include <cmath>
#include <stdexcept>

// Subtracts v2 from v1 element-wise
Vector subtract_vectors(const Vector& v1, const Vector& v2) {
    Vector res = v1;
    for (int i = 0; i < (int)res.size(); i++)
        res[i] -= v2[i];
    return res;
}

// Dot product of two vectors: v1^T * v2
double dot_product(const Vector& v1, const Vector& v2, int dimension) {
    double res = 0.0;
    for (int i = 0; i < dimension; i++)
        res += v1[i] * v2[i];
    return res;
}

// Outer product: v * v^T -> dimension x dimension matrix
// res[i][j] = v[i] * v[j]
Matrix outer_product(const Vector& v, int dimension) {
    Matrix res(dimension, Vector(dimension, 0.0));
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            res[i][j] = v[i] * v[j];
    return res;
}

// Matrix-vector product: matrix * vector
// Intermediate mat[j][i] = matrix[j][i] * vector[i], then sum rows
Vector matrix_vector_product(const Matrix& matrix, const Vector& vector, int dimension) {
    Matrix mat(dimension, Vector(dimension, 0.0));
    Vector res(dimension, 0.0);

    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            mat[j][i] = matrix[j][i] * vector[i];

    for (int k = 0; k < dimension; k++) {
        double row_sum = 0.0;
        for (int l = 0; l < dimension; l++)
            row_sum += mat[k][l];
        res[k] = row_sum;
    }
    return res;
}

// LU Decomposition using Doolittle's method
// L has 1s on diagonal, U is upper triangular
// Returns {lower, upper}
std::pair<Matrix, Matrix> lu_decomposition(const Matrix& matrix, int dimension) {
    Matrix lower(dimension, Vector(dimension, 0.0));
    Matrix upper(dimension, Vector(dimension, 0.0));

    // Initialize: diagonal of L = 1, first row of U, first column of L
    for (int i = 0; i < dimension; i++) {
        lower[i][i] = 1.0;
        upper[0][i] = matrix[0][i];
        if (std::abs(upper[0][0]) < 1e-12)
            throw std::runtime_error("Zero pivot in LU decomposition");
        lower[i][0] = matrix[i][0] / upper[0][0];
    }

    for (int j = 1; j < dimension; j++) {
        // Compute row j of U (upper triangular entries)
        for (int k = 0; k <= j; k++) {
            double sum_u = 0.0;
            for (int s = 0; s < k; s++)
                sum_u += lower[k][s] * upper[s][j];
            upper[k][j] = matrix[k][j] - sum_u;
        }

        // Compute column j of L (lower triangular entries below diagonal)
        for (int k = j + 1; k < dimension; k++) {
            double sum_l = 0.0;
            for (int s = 0; s < j; s++)
                sum_l += lower[k][s] * upper[s][j];
            if (std::abs(upper[j][j]) < 1e-12)
                throw std::runtime_error("Zero pivot in LU decomposition");
            lower[k][j] = (matrix[k][j] - sum_l) / upper[j][j];
        }
    }

    return {lower, upper};
}

// Compute determinant as product of U's diagonal
double calculate_determinant(const Matrix& upper, int dimension) {
    double det = 1.0;
    for (int i = 0; i < dimension; i++)
        det *= upper[i][i];
    return det;
}

// Compute inverse of A = LU via:
// Step 1: Forward substitution  -> solve LY = I
// Step 2: Backward substitution -> solve UX = Y
// Result X = A^-1
Matrix matrix_inverse(const Matrix& lower, const Matrix& upper, int dimension) {
    Matrix Y(dimension, Vector(dimension, 0.0));
    Matrix X(dimension, Vector(dimension, 0.0));

    // Identity matrix
    Matrix identity(dimension, Vector(dimension, 0.0));
    for (int m = 0; m < dimension; m++)
        identity[m][m] = 1.0;

    // Forward substitution: solve LY = I
    // L has 1s on diagonal so no division needed
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            double sum_Y = 0.0;
            for (int k = 0; k < i; k++)
                sum_Y += lower[i][k] * Y[k][j];
            Y[i][j] = identity[i][j] - sum_Y;
        }
    }

    // Backward substitution: solve UX = Y (bottom to top)
    for (int i = dimension - 1; i >= 0; i--) {
        for (int j = 0; j < dimension; j++) {
            double sum_X = 0.0;
            for (int k = i + 1; k < dimension; k++)
                sum_X += upper[i][k] * X[k][j];
            X[i][j] = (Y[i][j] - sum_X) / upper[i][i];
        }
    }

    return X;
}

// Add epsilon to diagonal to prevent singular matrix issues
Matrix regularize(const Matrix& matrix, int dimension, double epsilon) {
    Matrix result = matrix;
    for (int i = 0; i < dimension; i++)
        result[i][i] += epsilon;
    return result;
}
