#pragma once
#include <vector>
#include <string>
#include <cmath>

using namespace std;
typedef vector<double> Vec;
typedef vector<vector<double>> Mat;

Mat matrix_product_from_vector_and_transpose(int dimension, Vec vector) {
    Mat res(dimension, Vec(dimension, 0.0));
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            res[i][j] = vector[i] * vector[j];
    return res;
}

Vec calculate_vector(Vec v1, Vec v2) {
    Vec res = v1;
    for (int i = 0; i < (int)res.size(); i++)
        res[i] -= v2[i];
    return res;
}

Vec sum_across_column(Mat matrix) {
    Vec res(matrix.size(), 0.0);
    for (int k = 0; k < (int)matrix.size(); k++) {
        double row_sum = 0.0;
        for (int l = 0; l < (int)matrix[0].size(); l++)
            row_sum += matrix[k][l];
        res[k] = row_sum;
    }
    return res;
}

pair<Mat, Mat> LU_decomposition(Mat matrix, int dimension) {
    Mat lower(dimension, Vec(dimension, 0.0));
    Mat upper(dimension, Vec(dimension, 0.0));
    for (int i = 0; i < dimension; i++) {
        lower[i][i] = 1.0;
        upper[0][i] = matrix[0][i];
        lower[i][0] = matrix[i][0] / upper[0][0];
    }
    for (int j = 1; j < dimension; j++) {
        for (int k = 0; k < j + 1; k++) {
            double sum_u = 0.0;
            for (int s = 0; s < k; s++)
                sum_u += lower[k][s] * upper[s][j];
            upper[k][j] = matrix[k][j] - sum_u;
        }
        for (int k = j + 1; k < dimension; k++) {
            double sum_l = 0.0;
            for (int s = 0; s < j; s++)
                sum_l += lower[k][s] * upper[s][j];
            lower[k][j] = (1.0 / upper[j][j]) * (matrix[k][j] - sum_l);
        }
    }
    return {lower, upper};
}

double calculate_determinant(Mat upper, int dimension) {
    double det = 1.0;
    for (int i = 0; i < dimension; i++)
        det *= upper[i][i];
    return det;
}

Mat matrix_inverse(Mat lower, Mat upper, int dimension) {
    Mat Y_res(dimension, Vec(dimension, 0.0));
    Mat X_res(dimension, Vec(dimension, 0.0));
    Mat identity(dimension, Vec(dimension, 0.0));
    for (int m = 0; m < dimension; m++)
        identity[m][m] = 1.0;
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            double sum_Y = 0.0;
            for (int k = 0; k < i; k++)
                sum_Y += lower[i][k] * Y_res[k][j];
            Y_res[i][j] = identity[i][j] - sum_Y;
        }
    }
    for (int i = dimension - 1; i >= 0; i--) {
        for (int j = 0; j < dimension; j++) {
            double sum_X = 0.0;
            for (int k = i + 1; k < dimension; k++)
                sum_X += upper[i][k] * X_res[k][j];
            X_res[i][j] = (1.0 / upper[i][i]) * (Y_res[i][j] - sum_X);
        }
    }
    return X_res;
}

Vec matrix_product_with_matrix_and_vector(Mat matrix, Vec vector, int row, int col) {
    Mat mat(row, Vec(col, 0.0));
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
            mat[j][i] = matrix[j][i] * vector[i];
    return sum_across_column(mat);
}

double scalar_product_from_transpose_and_vector(Vec v1, Vec v2) {
    double res = 0.0;
    for (int i = 0; i < (int)v1.size(); i++)
        res += v1[i] * v2[i];
    return res;
}

Mat regularize(Mat matrix, int dimension, double epsilon = 1e-6) {
    for (int i = 0; i < dimension; i++)
        matrix[i][i] += epsilon;
    return matrix;
}

Mat matrix_with_matrix_multiplication(const Mat& A, const Mat& B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    Mat res(M, Vec(N, 0.0));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                res[i][j] += A[i][k] * B[k][j];
    return res;
}

Mat transpose_matrix(Mat matrix) {
    Mat res(matrix[0].size(), Vec(matrix.size(), 0.0));
    for (int row = 0; row < (int)matrix.size(); row++)
        for (int col = 0; col < (int)matrix[0].size(); col++)
            res[col][row] = matrix[row][col];
    return res;
}

Mat scalar_multiply_matrix(Mat matrix, double scalar) {
    for (int row = 0; row < (int)matrix.size(); row++)
        for (int col = 0; col < (int)matrix[0].size(); col++)
            matrix[row][col] *= scalar;
    return matrix;
}

Mat matrix_addition_and_sub(Mat matrix_one, Mat matrix_two, string operation) {
    Mat res(matrix_one.size(), Vec(matrix_one[0].size(), 0.0));
    for (int row = 0; row < (int)matrix_one.size(); row++) {
        for (int col = 0; col < (int)matrix_two[0].size(); col++) {
            if (operation == "add")
                res[row][col] = matrix_one[row][col] + matrix_two[row][col];
            else
                res[row][col] = matrix_one[row][col] - matrix_two[row][col];
        }
    }
    return res;
}

Mat element_wise_multiplication(Mat matrix_one, Mat matrix_two) {
    Mat res(matrix_one.size(), Vec(matrix_one[0].size(), 0.0));
    for (int row = 0; row < (int)matrix_one.size(); row++)
        for (int col = 0; col < (int)matrix_one[0].size(); col++)
            res[row][col] = matrix_one[row][col] * matrix_two[row][col];
    return res;
}

Mat element_wise_division_two_matrices(Mat matrix_one, Mat matrix_two) {
    Mat res(matrix_one.size(), Vec(matrix_one[0].size(), 0.0));
    for (int row = 0; row < (int)matrix_one.size(); row++)
        for (int col = 0; col < (int)matrix_one[0].size(); col++)
            res[row][col] = matrix_one[row][col] / matrix_two[row][col];
    return res;
}

Mat element_wise_roots(Mat matrix, double root_of) {
    Mat res(matrix.size(), Vec(matrix[0].size(), 0.0));
    for (int row = 0; row < (int)matrix.size(); row++)
        for (int col = 0; col < (int)matrix[0].size(); col++)
            res[row][col] = pow(matrix[row][col], 1.0 / root_of);
    return res;
}
