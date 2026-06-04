#This is a custom library for math functions since I am not using numpy.
#Makes an easier C port

#This function calculates the matrix derived from the product of a vector and its transpose
def matrix_product_from_vector_and_transpose(dimension, vector):
    res = [[0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):

            #Transpose product
            res [i][j] = vector[i] * vector[j]
    
    return res

#This calculates the resultant vector (v1 - v2)
def calculate_vector(v1, v2):
    res = v1.copy()
    for i in range(len(res)):
        res[i] -= v2[i]
    return res


def LU_decomposition(matrix, dimension):
    #Rewriting matrix as the product of an upper triangle and a lower triangle matrix
    #Using the doolittle method of initialization
    lower = [[0] * dimension for _ in range(dimension)]
    upper = [[0] * dimension for _ in range(dimension)]

    # Initialize first row of U and first column of L
    for i in range(dimension):
        lower[i][i] = 1
        upper[0][i] = matrix[0][i]
        lower[i][0] = matrix[i][0] / upper[0][0]

    for j in range(1, dimension):

        # Calculating rows of upper matrix
        for k in range(j + 1):
            sum_u = 0
            for s in range(k):
                sum_u += lower[k][s] * upper[s][j]
            upper[k][j] = matrix[k][j] - sum_u

        # Calculating columns of lower matrix
        for k in range(j + 1, dimension):
            sum_l = 0
            for s in range(j):
                sum_l += lower[k][s] * upper[s][j]
            lower[k][j] = (1/upper[j][j])*(matrix[k][j] - sum_l)

    
    return lower, upper

def calculate_determinant(upper, dimension):
    # Determinant is product of U's diagonal
    det = 1
    for i in range(dimension):
        det *= upper[i][i]

    return det


#Since we have the LU Decomposition, this uses forward and backward 
#substitution to obtain the inverse of the matrix. So, L * Y = I
#where we solve for Y whereas U*X = Y where we solve for X. Here,
#X is the inverse of the original matrix
def matrix_inverse(lower, upper, dimension):
    Y_res = [[0] * dimension for _ in range(dimension)]
    X_res = [[0] * dimension for _ in range(dimension)]
    identity = [[0] * dimension for _ in range (dimension)]
    #Building the identity matrix
    for m in range(dimension):
        identity [m][m] = 1

    
    #Obtaining Y
    for i in range(dimension):
        for j in range(dimension):
            sum_Y = 0
            for k in range(i):
                sum_Y += lower[i][k] * Y_res[k][j]
            Y_res[i][j] = identity[i][j] - sum_Y
    
    #Obtaining X
    for i in range(dimension - 1, -1, -1):  
        for j in range(dimension):
            sum_X = 0
            for k in range(i + 1, dimension):
                sum_X += upper[i][k] * X_res[k][j]
            X_res[i][j] = (1 / upper[i][i]) * (Y_res[i][j] - sum_X)
    
    #X_res = inverse of the matrix 
    return X_res

#Calculating the product of a matrix and vector 
def matrix_product_with_matrix_and_vector(matrix, vector, row, col):
    mat = [[0] * col for _ in range(row)]
    for i in range(col):
        for j in range(row):
            mat[j][i] = matrix[j][i] * vector[i]

    return sum_across_column(mat)


def scalar_product_from_transpose_and_vector(v1, v2, dimension):
    res = 0
    for i in range(dimension):
        res += v1[i] * v2[i]
    return res

#Regularising a matrix to help prevent division by zero error
def regularize(matrix, dimension, epsilon=1e-6):
    for i in range(dimension):
        matrix[i][i] += epsilon
    return matrix

def matrix_with_matrix_multiplication(matrix_one, matrix_two):
    #Obtaining the column vectors from each matrix
    vectors = []
    for col in range(len(matrix_two[0])):
        vector = []
        for row in range(len(matrix_two)):
            vector.append(matrix_two[row][col])
        vectors.append(vector)
    
    res = []
    for vector in vectors:
        #Since this appends each column vector, it is interpretted as a row instead of a column
        res.append(matrix_product_with_matrix_and_vector(matrix_one, vector, len(matrix_one), len(matrix_one[0])))
    
    #Therefore, the transpose must be returned
    return transpose_matrix(res)

#Calculates the transpose of the matrix
def transpose_matrix(matrix):
    #Shape of the transposed matrix
    res = [[0] * len(matrix) for _ in range(len(matrix[0]))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            res[col][row] = matrix[row][col]
    
    return res

#Applies the activation function to each element in the matrix
def element_wise_activation(matrix, activation_function):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix[row][col] = activation_function(matrix[row][col])
    
    return matrix

#Sums all the elements on the same row
def sum_across_column(matrix):
    res = [0] * len(matrix)
    for k in range(len(matrix)):
        row_sum = 0
        for l in range(len(matrix[0])):
            row_sum += matrix[k][l]
        res[k] = row_sum
    
    return res

#Multiples a scalar with a matrix
def scalar_multiply_matrix(matrix, scalar):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix[row][col] *= scalar
    
    return matrix

#Matrix addition and subtraction
def matrix_addition_and_sub(matrix_one, matrix_two, operation):
    res = [[0] * len(matrix_one[0]) for _ in range(len(matrix_one))]
    for row in range(len(matrix_one)):
        for col in range(len(matrix_two)):
            if operation == "add":
                res[row][col] = matrix_one[row][col] + matrix_two[row][col]
            else:
                res[row][col] = matrix_one[row][col] - matrix_two[row][col]
    
    return res

def element_wise_multiplication(matrix_one, matrix_two):
    res = [[0] * len(matrix_one[0]) for _ in range(len(matrix_one))]
    for row in range(len(matrix_one)):
        for col in range(len(matrix_one[0])):
            res[row][col] = matrix_one[row][col] * matrix_two[row][col]
    
    return res