import math

def matrix_product_from_vector_and_transpose(dimension, vector):
    res = [[0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            res[i][j] = vector[i] * vector[j]
    return res

def calculate_vector(v1, v2):
    res = v1.copy()
    for i in range(len(res)):
        res[i] -= v2[i]
    return res

def LU_decomposition(matrix, dimension):
    lower = [[0] * dimension for _ in range(dimension)]
    upper = [[0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        lower[i][i] = 1
        upper[0][i] = matrix[0][i]
        lower[i][0] = matrix[i][0] / upper[0][0]
    for j in range(1, dimension):
        for k in range(j + 1):
            sum_u = 0
            for s in range(k):
                sum_u += lower[k][s] * upper[s][j]
            upper[k][j] = matrix[k][j] - sum_u
        for k in range(j + 1, dimension):
            sum_l = 0
            for s in range(j):
                sum_l += lower[k][s] * upper[s][j]
            lower[k][j] = (1 / upper[j][j]) * (matrix[k][j] - sum_l)
    return lower, upper

def calculate_determinant(upper, dimension):
    det = 1
    for i in range(dimension):
        det *= upper[i][i]
    return det

def matrix_inverse(lower, upper, dimension):
    Y_res = [[0] * dimension for _ in range(dimension)]
    X_res = [[0] * dimension for _ in range(dimension)]
    identity = [[0] * dimension for _ in range(dimension)]
    for m in range(dimension):
        identity[m][m] = 1
    for i in range(dimension):
        for j in range(dimension):
            sum_Y = 0
            for k in range(i):
                sum_Y += lower[i][k] * Y_res[k][j]
            Y_res[i][j] = identity[i][j] - sum_Y
    for i in range(dimension - 1, -1, -1):
        for j in range(dimension):
            sum_X = 0
            for k in range(i + 1, dimension):
                sum_X += upper[i][k] * X_res[k][j]
            X_res[i][j] = (1 / upper[i][i]) * (Y_res[i][j] - sum_X)
    return X_res

def sum_across_column(matrix):
    res = [0] * len(matrix)
    for k in range(len(matrix)):
        row_sum = 0
        for l in range(len(matrix[0])):
            row_sum += matrix[k][l]
        res[k] = row_sum
    return res

def matrix_product_with_matrix_and_vector(matrix, vector, row, col):
    mat = [[0] * col for _ in range(row)]
    for i in range(col):
        for j in range(row):
            mat[j][i] = matrix[j][i] * vector[i]
    return sum_across_column(mat)

def scalar_product_from_transpose_and_vector(v1, v2):
    res = 0
    for i in range(len(v1)):
        res += v1[i] * v2[i]
    return res

def regularize(matrix, dimension, epsilon=1e-6):
    for i in range(dimension):
        matrix[i][i] += epsilon
    return matrix

def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    res = [[0] * rows for _ in range(cols)]
    for row in range(rows):
        for col in range(cols):
            res[col][row] = matrix[row][col]
    return res

def matrix_with_matrix_multiplication(matrix_one, matrix_two):
    rows_A = len(matrix_one)
    cols_A = len(matrix_one[0])
    cols_B = len(matrix_two[0])
    res = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                res[i][j] += matrix_one[i][k] * matrix_two[k][j]
    return res

def matrix_addition_and_sub(matrix_one, matrix_two, operation):
    res = [[0] * len(matrix_one[0]) for _ in range(len(matrix_one))]
    for row in range(len(matrix_one)):
        for col in range(len(matrix_one[0])):
            if operation == "add":
                res[row][col] = matrix_one[row][col] + matrix_two[row][col]
            else:
                res[row][col] = matrix_one[row][col] - matrix_two[row][col]
    return res

def scalar_multiply_matrix(matrix, scalar):
    res = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            res[row][col] = matrix[row][col] * scalar
    return res

def element_wise_multiplication(matrix_one, matrix_two):
    res = [[0] * len(matrix_one[0]) for _ in range(len(matrix_one))]
    for row in range(len(matrix_one)):
        for col in range(len(matrix_one[0])):
            res[row][col] = matrix_one[row][col] * matrix_two[row][col]
    return res

def element_wise_division_two_matrices(matrix_one, matrix_two):
    res = [[0] * len(matrix_one[0]) for _ in range(len(matrix_one))]
    for row in range(len(matrix_one)):
        for col in range(len(matrix_one[0])):
            res[row][col] = matrix_one[row][col] / matrix_two[row][col]
    return res

def element_wise_roots(matrix, root_of):
    res = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            res[row][col] = matrix[row][col] ** (1 / root_of)
    return res

def ReLU_derivative(matrix):
    res = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            res[row][col] = 1 if matrix[row][col] > 0 else 0
    return res
