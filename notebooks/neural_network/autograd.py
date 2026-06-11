import math
from collections import deque
import custom_math
#This class purely exists so this can be passed in as a parameter into the backprop function
#Element is the base class that both Scalar and Matrix inherit from. It defines the interface
#that every node in the computation graph must support, regardless of whether the data it holds
#is a single number or a full matrix. Each subclass overrides these with its own forward value
#computation and its own back() closure that knows the correct local derivative rule for that type.
class Element:
    def __init__(self, data, children: set = (), operation = ""):
        self.data = data
        self.children = children
        self.operation = operation
        self.back = lambda: None
        self.gradient = 0

    #Elementwise or matrix addition depending on the subclass
    def __add__(self, other):
        raise NotImplementedError

    #Elementwise multiplication for Scalar, matrix multiplication for Matrix
    def __mul__(self, other):
        raise NotImplementedError

    #Elementwise or matrix subtraction depending on the subclass
    def __sub__(self, other):
        raise NotImplementedError

    #Elementwise division for Scalar, elementwise division for Matrix
    def __truediv__(self, other):
        raise NotImplementedError

    #Negation — flips the sign of every element
    def __neg__(self):
        raise NotImplementedError

    #Reverse operators — called by Python when the left operand cannot handle the operation.
    #Each subclass wraps the raw value in its own type and delegates to the forward operator.
    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    #Activation functions — applied elementwise in both subclasses
    def relu(self):
        raise NotImplementedError

    def sigmoid(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

class TopologicalSort():
    def __init__(self):
        pass
    #This is to prevent calling backprop manually. Kahns topological sort flattens the graph into a list that can be traversed in order
    #That way back can be called in the correct order
    def topological_sort(self, root_node:Element):
        num_dependancies = {}
        
        def collect_nodes(node:Element):
            if node not in num_dependancies:
                num_dependancies[node] = 0
            for c in node.children:
                collect_nodes(c)
        
        def obtain_dependancies():
            for n in num_dependancies:
                for c in n.children:
                    num_dependancies[c] += 1

        final_res = []
        collect_nodes(root_node)
        obtain_dependancies()
        print(num_dependancies)

        q = deque()
        for n in num_dependancies:
            if num_dependancies[n] == 0:
                q.append(n)

        while len(q) > 0:
            qLen = len(q)
            for _ in range(qLen):
                node:Element = q.pop()
                final_res.append(node)
                for c in node.children:
                    num_dependancies[c] -= 1
                    if num_dependancies[c] == 0:
                        q.append(c)
        
        return final_res
    
    #Separated out the backprop and the topological sort since the topological sort is only required once
    #The backprop must be actually called for each epoch, not the topological sort. The topological sort will be 
    #cached in the network
    def backprop(final_res):
        for node in final_res:
            node.back()


#Using a scalar class that represents the node in each graph for any kind of mathematical equation
#The whole point of the autograd is that each sort of individual operation has its own unique way of taking the derivative
#For example: if a = b + c, then da/db = 1 or da/dc = 1. However, if a = b*c, then da/db = c and da/dc = b. Each of these operations
#Have a different way of computing their local derivatives. In a bigger more convoluted equation like a = b*c + d/f - c+b,
#The equation can be reconstrcuted by the correct order of operations and the chain rule can be applied starting from the local derivative
#all the way up to the desired result.

class Scalar(Element):
    #Each scalar has a set of children its DIRECTLY derived from. NOT the indirect. The indirect children 
    #can be obtained via traversing backwards in the graph
    def __init__(self, digit, children:set = (), operation = ""):
        self.digit = digit
        self.children = children
        self.operation = operation
        self.back = lambda: None
        #By default, the gradient is taken of a final expression that will be specified
        self.gradient = 0
        self.topo = TopologicalSort()
    
    #Building out the basic operations.
    def __add__(self, other:Scalar):
        res = Scalar(self.digit + other.digit, {self, other}, "add")
        def back():
            #The local derivative of an added expression w.r.t 1 element (ex: d(a + b + c)/da = 1) is 1
            #res.gradient is the derivative of the FINAL output with respect to this current res
            self.gradient += res.gradient
            #Same logic applies for other
            other.gradient += res.gradient
        res.back = back
        return res
    
    def __mul__(self, other:Scalar):
        res = Scalar(self.digit * other.digit, {self, other}, "mul")
        def back():
            #Local derivative (d(ab)/da = b) and (d(ab)/db = a) 
            #Multiply local with global derivative
            self.gradient += other.digit * res.gradient
            other.gradient += self.digit * res.gradient
        res.back = back
        return res
    
    #Using true div since we want floats
    def __truediv__(self, other:Scalar):
        res = Scalar(self.digit / other.digit, {self, other}, "div")
        def back():
            #Local derivative (d(a/b)/da = 1/b) and (d(a/b)/db = -a*b^-2)
            #Multiply with global derivative
            self.gradient += res.gradient * (1/other.digit)
            other.gradient += self.digit * (-(other.digit**-2)) * res.gradient
        res.back = back
        return res
    
    def __sub__(self, other:Scalar):
        res = Scalar(self.digit - other.digit, {self, other}, "sub")
        def back():
            #Local derivative (d(a-b)/da = 1) and (d(a-b)/db = -1)
            #Multiply with global derivative
            self.gradient += res.gradient
            other.gradient += -res.gradient
        res.back = back
        return res

    def __neg__(self):
        res = Scalar(-self.digit, {self}, "neg")
        def back():
            #Local derivative (d(-a)/da = -1)
            #Multiply with global derivative
            self.gradient += -1 * res.gradient
        res.back = back
        return res

    #arctan
    def tanh(self):
        res = Scalar(math.tanh(self.digit), {self}, "tanh")
        def back():
            #Local derivative (d(tanh(a))/da = 1 - tanh^2(a)) 
            #Multiply with global derivative
            self.gradient += (1 - (math.tanh(self.digit))**2) * res.gradient
        res.back = back
        return res

    def exp(self):
        res = Scalar(math.exp(self.digit), {self}, "exp")
        def back():
            #Local derivative (d(e^a)/da = e^a)
            #Multiply with global derivative
            self.gradient += math.exp(self.digit) * res.gradient
        res.back = back
        return res

    def log(self):
        res = Scalar(math.log(self.digit), {self}, "log")
        def back():
            #Local derivative (d(ln(a))/da = 1/a)
            #Multiply with global derivative
            self.gradient += (1 / self.digit) * res.gradient
        res.back = back
        return res

    def relu(self):
        res = Scalar(max(0, self.digit), {self}, "relu")
        def back():
            #Local derivative (d(relu(a))/da = 1 if a > 0, else 0)
            #Multiply with global derivative
            self.gradient += (1 if self.digit > 0 else 0) * res.gradient
        res.back = back
        return res

    def sigmoid(self):
        sig = 1 / (1 + math.exp(-self.digit))
        res = Scalar(sig, {self}, "sigmoid")
        def back():
            #Local derivative (d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a)))
            #Multiply with global derivative
            self.gradient += (sig * (1 - sig)) * res.gradient
        res.back = back
        return res

    def pow(self, other:Scalar):
        res = Scalar(math.pow(self.digit, other.digit), {self, other}, "pow")
        def back():
            #Local derivative (d(a^b)/da = b * a^(b-1)) and (d(a^b)/db = a^b * ln(a))
            #Multiply with global derivative
            self.gradient += (other.digit * math.pow(self.digit, other.digit - 1)) * res.gradient
            other.gradient += (res.digit * math.log(self.digit)) * res.gradient
        res.back = back
        return res

    #Reverse operations handle cases where a raw number is on the left (e.g. 2 + scalar, 3 * scalar).
    #Python falls back to these when the left operand doesn't know how to handle a Scalar.
    def __radd__(self, other):
        return Scalar(other) + self

    def __rmul__(self, other):
        return Scalar(other) * self

    def __rsub__(self, other):
        return Scalar(other) - self

    def __rtruediv__(self, other):
        return Scalar(other) / self
    
    def topo_sort(self):
        self.topo.topological_sort(self)
    
    def backprop(self, sorted_graph):
        self.topo.backprop(sorted_graph)

#Support for tensors (matrices for now) instead of just scalar values.
#The core idea is identical to the Scalar class — each Matrix node records which operation
#created it and which matrices it was directly derived from. The difference is that gradients
#are now matrices themselves, so every local derivative rule operates element-wise or via
#matrix calculus rather than scalar calculus.
class Matrix(Element):
    #Each matrix node stores the data, its direct parents in the graph, and the operation that produced it.
    #gradient is initialised to 0 and will be populated as a matrix during the backward pass.
    def __init__(self, matrix, children:set = (), operation = ""):
        self.matrix = matrix
        self.children = children
        self.operation = operation
        self.back = lambda: None
        #By default, the gradient is taken of a final expression that will be specified
        self.gradient = 0
        self.topo = TopologicalSort()

    #Building out the basic operations.
    def __add__(self, other:Matrix):
        res = Matrix(custom_math.matrix_addition_and_sub(self.matrix, other.matrix, "add"), {self, other}, "add")
        def back():
            #The local derivative of matrix addition w.r.t either operand is the identity — d(A+B)/dA = I and d(A+B)/dB = I
            #So the upstream gradient flows through unchanged to both operands
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, res.gradient, "add")
            #Same logic applies for other
            other.gradient = custom_math.matrix_addition_and_sub(other.gradient, res.gradient, "add")
        res.back = back
        return res
    
    def __mul__(self, other:Matrix):
        res = Matrix(custom_math.matrix_with_matrix_multiplication(self.matrix, other.matrix), {self, other}, "mul")
        def back():
            #For matrix multiplication C = A @ B, the local derivatives are dC/dA = B^T and dC/dB = A^T
            #Multiply local with global (upstream) derivative via the chain rule
            #dL/dA = dL/dC @ B^T — upstream gradient times B transposed on the right
            #dL/dB = A^T @ dL/dC — A transposed times upstream gradient on the left
            term = custom_math.matrix_with_matrix_multiplication(res.gradient, custom_math.transpose_matrix(other.matrix))
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, term, "add")
            term = custom_math.matrix_with_matrix_multiplication(custom_math.transpose_matrix(self.matrix), res.gradient)
            other.gradient = custom_math.matrix_addition_and_sub(other.gradient, term, "add")
        res.back = back
        return res
    
    def __sub__(self, other:Matrix):
        res = Matrix(custom_math.matrix_addition_and_sub(self.matrix, other.matrix, "sub"), {self, other}, "sub")
        def back():
            #Local derivative (d(A-B)/dA = I) and (d(A-B)/dB = -I)
            #Multiply with global derivative
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, res.gradient, "add")
            term = custom_math.scalar_multiply_matrix(res.gradient, -1)
            other.gradient = custom_math.matrix_addition_and_sub(other.gradient, term, "add")
        res.back = back
        return res

    def __neg__(self):
        res = Matrix(custom_math.scalar_multiply_matrix(self.matrix, -1), {self}, "neg")
        def back():
            #Local derivative (d(-A)/dA = -I)
            #Multiply with global derivative
            term = custom_math.scalar_multiply_matrix(res.gradient, -1)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, term, "add")
        res.back = back
        return res
    
    def element_wise_mult(self, other:Matrix):
        res = Matrix(custom_math.element_wise_multiplication(self.matrix, other.matrix), {self, other}, "elt_mul")
        def back():
            #Local derivative of element-wise (Hadamard) multiplication: d(A⊙B)/dA_ij = B_ij and d(A⊙B)/dB_ij = A_ij
            #Multiply with global derivative
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(other.matrix, res.gradient), "add")
            other.gradient = custom_math.matrix_addition_and_sub(other.gradient, custom_math.element_wise_multiplication(self.matrix, res.gradient), "add")
        res.back = back
        return res
    
    def element_wise_div(self, other:Matrix):
        res = Matrix(custom_math.element_wise_division_two_matrices(self.matrix, other.matrix), {self, other}, "elt_div")
        def back():
            #Local derivative of element-wise division: d(A/B)/dA_ij = 1/B_ij and d(A/B)/dB_ij = -A_ij / B_ij^2
            #Multiply with global derivative
            ones = [[1] * len(other.matrix[0]) for _ in range(len(other.matrix))]
            inv_other = custom_math.element_wise_division_two_matrices(ones, other.matrix)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(inv_other, res.gradient), "add")
            b_sq = custom_math.element_wise_multiplication(other.matrix, other.matrix)
            neg_a_over_b_sq = custom_math.scalar_multiply_matrix(custom_math.element_wise_division_two_matrices(self.matrix, b_sq), -1)
            other.gradient = custom_math.matrix_addition_and_sub(other.gradient, custom_math.element_wise_multiplication(neg_a_over_b_sq, res.gradient), "add")
        res.back = back
        return res

    #Multiplying every element of the matrix by a scalar value.
    #The scalar is a raw number, not a Matrix node, so only self receives a gradient.
    def scalar_multiply(self, scalar):
        res = Matrix(custom_math.scalar_multiply_matrix(self.matrix, scalar), {self}, "scalar_mul")
        def back():
            #Local derivative (d(sA)/dA_ij = s) — the scalar passes straight through
            #Multiply with global derivative
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.scalar_multiply_matrix(res.gradient, scalar), "add")
        res.back = back
        return res

    def transpose(self):
        res = Matrix(custom_math.transpose_matrix(self.matrix), {self}, "transpose")
        def back():
            #Transposing twice returns the original, so the upstream gradient just needs to be transposed back
            #Local derivative (d(A^T)/dA = upstream gradient transposed)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.transpose_matrix(res.gradient), "add")
        res.back = back
        return res

    def relu(self):
        relu_matrix = [[max(0, self.matrix[r][c]) for c in range(len(self.matrix[0]))] for r in range(len(self.matrix))]
        res = Matrix(relu_matrix, {self}, "relu")
        def back():
            #Local derivative (d(relu(A))/dA_ij = 1 if A_ij > 0, else 0) — the ReLU derivative mask
            #Multiply with global derivative
            mask = custom_math.ReLU_derivative(self.matrix)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(mask, res.gradient), "add")
        res.back = back
        return res

    def sigmoid(self):
        sig_matrix = [[1 / (1 + math.exp(-self.matrix[r][c])) for c in range(len(self.matrix[0]))] for r in range(len(self.matrix))]
        res = Matrix(sig_matrix, {self}, "sigmoid")
        def back():
            #Local derivative (d(sigmoid(A))/dA_ij = sigmoid(A_ij) * (1 - sigmoid(A_ij)))
            #res.matrix already holds the sigmoid values so we reuse them directly
            #Multiply with global derivative
            ones = [[1] * len(res.matrix[0]) for _ in range(len(res.matrix))]
            one_minus_sig = custom_math.matrix_addition_and_sub(ones, res.matrix, "sub")
            local_grad = custom_math.element_wise_multiplication(res.matrix, one_minus_sig)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(local_grad, res.gradient), "add")
        res.back = back
        return res

    def tanh(self):
        tanh_matrix = [[math.tanh(self.matrix[r][c]) for c in range(len(self.matrix[0]))] for r in range(len(self.matrix))]
        res = Matrix(tanh_matrix, {self}, "tanh")
        def back():
            #Local derivative (d(tanh(A))/dA_ij = 1 - tanh^2(A_ij))
            #res.matrix already holds the tanh values so we square them and subtract from 1
            #Multiply with global derivative
            tanh_sq = custom_math.element_wise_multiplication(res.matrix, res.matrix)
            ones = [[1] * len(res.matrix[0]) for _ in range(len(res.matrix))]
            local_grad = custom_math.matrix_addition_and_sub(ones, tanh_sq, "sub")
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(local_grad, res.gradient), "add")
        res.back = back
        return res

    def exp(self):
        exp_matrix = [[math.exp(self.matrix[r][c]) for c in range(len(self.matrix[0]))] for r in range(len(self.matrix))]
        res = Matrix(exp_matrix, {self}, "exp")
        def back():
            #Local derivative (d(e^A)/dA_ij = e^A_ij) — the exponential is its own derivative
            #res.matrix already holds the exponential values so we reuse them directly
            #Multiply with global derivative
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(res.matrix, res.gradient), "add")
        res.back = back
        return res

    def log(self):
        log_matrix = [[math.log(self.matrix[r][c]) for c in range(len(self.matrix[0]))] for r in range(len(self.matrix))]
        res = Matrix(log_matrix, {self}, "log")
        def back():
            #Local derivative (d(ln(A))/dA_ij = 1/A_ij)
            #Multiply with global derivative
            ones = [[1] * len(self.matrix[0]) for _ in range(len(self.matrix))]
            inv_self = custom_math.element_wise_division_two_matrices(ones, self.matrix)
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, custom_math.element_wise_multiplication(inv_self, res.gradient), "add")
        res.back = back
        return res

    #Sums each row across all its columns, producing a (rows x 1) column vector.
    #This is the matrix equivalent of summing a vector — used mainly to collapse batch gradients into bias gradients.
    def sum_cols(self):
        summed = [[sum(self.matrix[r])] for r in range(len(self.matrix))]
        res = Matrix(summed, {self}, "sum_cols")
        def back():
            #Local derivative: each element of the output res_i = sum_j(A_ij), so d(res_i)/d(A_ij) = 1 for all j
            #The upstream gradient res.gradient_i therefore broadcasts back to every column in row i
            broadcast = [[res.gradient[r][0]] * len(self.matrix[0]) for r in range(len(self.matrix))]
            self.gradient = custom_math.matrix_addition_and_sub(self.gradient, broadcast, "add")
        res.back = back
        return res
    
    def topo_sort(self):
        self.topo.topological_sort(self)
    
    def backprop(self, sorted_graph):
        self.topo.backprop(sorted_graph)