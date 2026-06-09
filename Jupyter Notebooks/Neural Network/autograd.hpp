#pragma once
#include <vector>
#include <string>
#include <set>
#include <map>
#include <deque>
#include <functional>
#include <memory>
#include <cmath>
#include "../../custom_math.cpp"

using namespace std;

class Element;
class Scalar;
class Matrix;

using ElemPtr    = shared_ptr<Element>;
using ScalarPtr  = shared_ptr<Scalar>;
using MatrixPtr  = shared_ptr<Matrix>;


// ── Element ───────────────────────────────────────────────────────────────────

//Element is the base class that both Scalar and Matrix inherit from. It defines the interface
//that every node in the computation graph must support, regardless of whether the data it holds
//is a single number or a full matrix. Each subclass overrides these with its own forward value
//computation and its own back() closure that knows the correct local derivative rule for that type.
class Element : public enable_shared_from_this<Element> {
public:
    set<ElemPtr>         children;
    string               operation;
    function<void()>     back;

    Element(set<ElemPtr> children = {}, string op = "")
        : children(children), operation(op), back([](){}) {}

    virtual ~Element() = default;

    //Elementwise or matrix addition depending on the subclass
    virtual ElemPtr add(ElemPtr other) = 0;

    //Elementwise multiplication for Scalar, matrix multiplication for Matrix
    virtual ElemPtr mul(ElemPtr other) = 0;

    //Elementwise or matrix subtraction depending on the subclass
    virtual ElemPtr sub(ElemPtr other) = 0;

    //Elementwise division for Scalar, elementwise division for Matrix
    virtual ElemPtr truediv(ElemPtr other) = 0;

    //Negation — flips the sign of every element
    virtual ElemPtr neg() = 0;

    //Activation functions — applied elementwise in both subclasses
    virtual ElemPtr relu()    = 0;
    virtual ElemPtr sigmoid() = 0;
    virtual ElemPtr tanh_op() = 0;
    virtual ElemPtr exp_op()  = 0;
    virtual ElemPtr log_op()  = 0;
};


// ── TopologicalSort ───────────────────────────────────────────────────────────

//Kahn's topological sort flattens the DAG into a list ordered from root to leaves.
//back() is then called in that order so each node receives its upstream gradient
//before it propagates it to its own children.
class TopologicalSort {
public:
    void backprop(ElemPtr root_node) {
        map<Element*, int>    num_dependencies;
        map<Element*, ElemPtr> node_ptrs;

        //Collect every node reachable from the root and initialise its dependency count to 0
        function<void(ElemPtr)> collect_nodes = [&](ElemPtr node) {
            if (num_dependencies.find(node.get()) == num_dependencies.end()) {
                num_dependencies[node.get()] = 0;
                node_ptrs[node.get()] = node;
                for (auto& c : node->children)
                    collect_nodes(c);
            }
        };
        collect_nodes(root_node);

        //Count how many parents each node has (how many nodes depend on it)
        for (auto& [ptr, count] : num_dependencies)
            for (auto& c : node_ptrs[ptr]->children)
                num_dependencies[c.get()]++;

        //Nodes with 0 dependants are roots — start the BFS from them
        deque<Element*> q;
        for (auto& [ptr, count] : num_dependencies)
            if (count == 0)
                q.push_back(ptr);

        vector<Element*> final_res;
        while (!q.empty()) {
            int q_len = q.size();
            for (int i = 0; i < q_len; i++) {
                Element* node = q.back();
                q.pop_back();
                final_res.push_back(node);
                for (auto& c : node->children) {
                    num_dependencies[c.get()]--;
                    if (num_dependencies[c.get()] == 0)
                        q.push_back(c.get());
                }
            }
        }

        for (auto* node : final_res)
            node->back();
    }
};


// ── Scalar ────────────────────────────────────────────────────────────────────

//Using a scalar class that represents the node in each graph for any kind of mathematical equation.
//The whole point of the autograd is that each sort of individual operation has its own unique way of
//taking the derivative. Each of these operations have a different way of computing their local
//derivatives. In a bigger more convoluted equation the chain rule can be applied starting from the
//local derivative all the way up to the desired result.
class Scalar : public Element {
public:
    double digit;
    double gradient;

    //Each scalar has a set of children it is DIRECTLY derived from. NOT the indirect children.
    //The indirect children can be obtained via traversing backwards in the graph.
    Scalar(double digit, set<ElemPtr> children = {}, string op = "")
        : Element(children, op), digit(digit), gradient(0.0) {}

    //Building out the basic operations.
    ElemPtr add(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit + other->digit, set<ElemPtr>{self, other}, "add");
        res->back = [res, self, other]() {
            //The local derivative of an added expression w.r.t 1 element is 1
            //res.gradient is the derivative of the FINAL output with respect to this current res
            self->gradient  += res->gradient;
            //Same logic applies for other
            other->gradient += res->gradient;
        };
        return res;
    }

    ElemPtr mul(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit * other->digit, set<ElemPtr>{self, other}, "mul");
        res->back = [res, self, other]() {
            //Local derivative (d(ab)/da = b) and (d(ab)/db = a)
            //Multiply local with global derivative
            self->gradient  += other->digit * res->gradient;
            other->gradient += self->digit  * res->gradient;
        };
        return res;
    }

    ElemPtr truediv(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit / other->digit, set<ElemPtr>{self, other}, "div");
        res->back = [res, self, other]() {
            //Local derivative (d(a/b)/da = 1/b) and (d(a/b)/db = -a*b^-2)
            //Multiply with global derivative
            self->gradient  += res->gradient * (1.0 / other->digit);
            other->gradient += self->digit * -(pow(other->digit, -2)) * res->gradient;
        };
        return res;
    }

    ElemPtr sub(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit - other->digit, set<ElemPtr>{self, other}, "sub");
        res->back = [res, self, other]() {
            //Local derivative (d(a-b)/da = 1) and (d(a-b)/db = -1)
            //Multiply with global derivative
            self->gradient  += res->gradient;
            other->gradient += -res->gradient;
        };
        return res;
    }

    ElemPtr neg() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(-self->digit, set<ElemPtr>{self}, "neg");
        res->back = [res, self]() {
            //Local derivative (d(-a)/da = -1)
            //Multiply with global derivative
            self->gradient += -1.0 * res->gradient;
        };
        return res;
    }

    ElemPtr tanh_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(tanh(self->digit), set<ElemPtr>{self}, "tanh");
        res->back = [res, self]() {
            //Local derivative (d(tanh(a))/da = 1 - tanh^2(a))
            //Multiply with global derivative
            self->gradient += (1.0 - pow(tanh(self->digit), 2)) * res->gradient;
        };
        return res;
    }

    ElemPtr exp_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(exp(self->digit), set<ElemPtr>{self}, "exp");
        res->back = [res, self]() {
            //Local derivative (d(e^a)/da = e^a)
            //Multiply with global derivative
            self->gradient += exp(self->digit) * res->gradient;
        };
        return res;
    }

    ElemPtr log_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(log(self->digit), set<ElemPtr>{self}, "log");
        res->back = [res, self]() {
            //Local derivative (d(ln(a))/da = 1/a)
            //Multiply with global derivative
            self->gradient += (1.0 / self->digit) * res->gradient;
        };
        return res;
    }

    ElemPtr relu() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(max(0.0, self->digit), set<ElemPtr>{self}, "relu");
        res->back = [res, self]() {
            //Local derivative (d(relu(a))/da = 1 if a > 0, else 0)
            //Multiply with global derivative
            self->gradient += (self->digit > 0 ? 1.0 : 0.0) * res->gradient;
        };
        return res;
    }

    ElemPtr sigmoid() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        double sig = 1.0 / (1.0 + exp(-self->digit));
        auto res = make_shared<Scalar>(sig, set<ElemPtr>{self}, "sigmoid");
        res->back = [res, self, sig]() {
            //Local derivative (d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a)))
            //Multiply with global derivative
            self->gradient += (sig * (1.0 - sig)) * res->gradient;
        };
        return res;
    }

    ScalarPtr pow_op(ScalarPtr other) {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(pow(self->digit, other->digit), set<ElemPtr>{self, other}, "pow");
        res->back = [res, self, other]() {
            //Local derivative (d(a^b)/da = b * a^(b-1)) and (d(a^b)/db = a^b * ln(a))
            //Multiply with global derivative
            self->gradient  += (other->digit * pow(self->digit, other->digit - 1)) * res->gradient;
            other->gradient += (res->digit * log(self->digit)) * res->gradient;
        };
        return res;
    }

    //Reverse operators — called when the left operand cannot handle the operation
    ScalarPtr radd(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->add(shared_from_this())); }
    ScalarPtr rmul(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->mul(shared_from_this())); }
    ScalarPtr rsub(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->sub(shared_from_this())); }
    ScalarPtr rdiv(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->truediv(shared_from_this())); }

    //This is to prevent calling backprop manually. Kahn's topological sort flattens the graph
    //into a list that can be traversed in order so back() is called in the correct sequence.
    void backprop() {
        TopologicalSort topo;
        topo.backprop(shared_from_this());
    }
};


// ── Matrix ────────────────────────────────────────────────────────────────────

//Support for tensors (matrices for now) instead of just scalar values.
//The core idea is identical to the Scalar class — each Matrix node records which operation
//created it and which matrices it was directly derived from. The difference is that gradients
//are now matrices themselves, so every local derivative rule operates element-wise or via
//matrix calculus rather than scalar calculus.
class Matrix : public Element {
public:
    Mat matrix;
    Mat gradient;

    //Each matrix node stores the data, its direct parents in the graph, and the operation that produced it.
    //gradient is initialised as an empty Mat and must be set to a zero matrix of the correct shape
    //before backprop runs so that matrix_addition_and_sub has a valid matrix to accumulate into.
    Matrix(Mat matrix, set<ElemPtr> children = {}, string op = "")
        : Element(children, op), matrix(matrix) {}

    //Building out the basic operations.
    ElemPtr add(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Matrix>(other_elem);
        auto self  = static_pointer_cast<Matrix>(shared_from_this());
        auto res   = make_shared<Matrix>(matrix_addition_and_sub(self->matrix, other->matrix, "add"),
                                         set<ElemPtr>{self, other}, "add");
        res->back = [res, self, other]() {
            //The local derivative of matrix addition w.r.t either operand is the identity — d(A+B)/dA = I
            //So the upstream gradient flows through unchanged to both operands
            self->gradient  = matrix_addition_and_sub(self->gradient,  res->gradient, "add");
            other->gradient = matrix_addition_and_sub(other->gradient, res->gradient, "add");
        };
        return res;
    }

    ElemPtr mul(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Matrix>(other_elem);
        auto self  = static_pointer_cast<Matrix>(shared_from_this());
        auto res   = make_shared<Matrix>(matrix_with_matrix_multiplication(self->matrix, other->matrix),
                                         set<ElemPtr>{self, other}, "mul");
        res->back = [res, self, other]() {
            //For matrix multiplication C = A @ B, the local derivatives are dC/dA = B^T and dC/dB = A^T
            //Multiply local with global (upstream) derivative via the chain rule
            Mat term = matrix_with_matrix_multiplication(transpose_matrix(other->matrix), res->gradient);
            self->gradient  = matrix_addition_and_sub(self->gradient, term, "add");
            term = matrix_with_matrix_multiplication(transpose_matrix(self->matrix), res->gradient);
            other->gradient = matrix_addition_and_sub(other->gradient, term, "add");
        };
        return res;
    }

    ElemPtr sub(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Matrix>(other_elem);
        auto self  = static_pointer_cast<Matrix>(shared_from_this());
        auto res   = make_shared<Matrix>(matrix_addition_and_sub(self->matrix, other->matrix, "sub"),
                                         set<ElemPtr>{self, other}, "sub");
        res->back = [res, self, other]() {
            //Local derivative (d(A-B)/dA = I) and (d(A-B)/dB = -I)
            //Multiply with global derivative
            self->gradient  = matrix_addition_and_sub(self->gradient, res->gradient, "add");
            Mat term = scalar_multiply_matrix(res->gradient, -1.0);
            other->gradient = matrix_addition_and_sub(other->gradient, term, "add");
        };
        return res;
    }

    ElemPtr neg() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(scalar_multiply_matrix(self->matrix, -1.0),
                                        set<ElemPtr>{self}, "neg");
        res->back = [res, self]() {
            //Local derivative (d(-A)/dA = -I)
            //Multiply with global derivative
            Mat term = scalar_multiply_matrix(res->gradient, -1.0);
            self->gradient = matrix_addition_and_sub(self->gradient, term, "add");
        };
        return res;
    }

    ElemPtr truediv(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Matrix>(other_elem);
        auto self  = static_pointer_cast<Matrix>(shared_from_this());
        auto res   = make_shared<Matrix>(element_wise_division_two_matrices(self->matrix, other->matrix),
                                         set<ElemPtr>{self, other}, "elt_div");
        res->back = [res, self, other]() {
            //Local derivative of element-wise division: d(A/B)/dA_ij = 1/B_ij and d(A/B)/dB_ij = -A_ij/B_ij^2
            //Multiply with global derivative
            int rows = other->matrix.size(), cols = other->matrix[0].size();
            Mat ones(rows, Vec(cols, 1.0));
            Mat inv_other = element_wise_division_two_matrices(ones, other->matrix);
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(inv_other, res->gradient), "add");
            Mat b_sq = element_wise_multiplication(other->matrix, other->matrix);
            Mat neg_a_over_b_sq = scalar_multiply_matrix(
                element_wise_division_two_matrices(self->matrix, b_sq), -1.0);
            other->gradient = matrix_addition_and_sub(other->gradient,
                element_wise_multiplication(neg_a_over_b_sq, res->gradient), "add");
        };
        return res;
    }

    MatrixPtr element_wise_mult(MatrixPtr other) {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(element_wise_multiplication(self->matrix, other->matrix),
                                        set<ElemPtr>{self, other}, "elt_mul");
        res->back = [res, self, other]() {
            //Local derivative of element-wise (Hadamard) multiplication: d(A⊙B)/dA_ij = B_ij
            //Multiply with global derivative
            self->gradient  = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(other->matrix, res->gradient), "add");
            other->gradient = matrix_addition_and_sub(other->gradient,
                element_wise_multiplication(self->matrix, res->gradient), "add");
        };
        return res;
    }

    //Multiplying every element of the matrix by a scalar value.
    //The scalar is a raw number, not a Matrix node, so only self receives a gradient.
    MatrixPtr scalar_multiply(double scalar) {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(scalar_multiply_matrix(self->matrix, scalar),
                                        set<ElemPtr>{self}, "scalar_mul");
        res->back = [res, self, scalar]() {
            //Local derivative (d(sA)/dA_ij = s) — the scalar passes straight through
            //Multiply with global derivative
            self->gradient = matrix_addition_and_sub(self->gradient,
                scalar_multiply_matrix(res->gradient, scalar), "add");
        };
        return res;
    }

    MatrixPtr transpose_op() {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(transpose_matrix(self->matrix), set<ElemPtr>{self}, "transpose");
        res->back = [res, self]() {
            //Transposing twice returns the original, so the upstream gradient just needs to be transposed back
            self->gradient = matrix_addition_and_sub(self->gradient,
                transpose_matrix(res->gradient), "add");
        };
        return res;
    }

    ElemPtr relu() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat relu_mat(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                relu_mat[r][c] = max(0.0, self->matrix[r][c]);
        auto res = make_shared<Matrix>(relu_mat, set<ElemPtr>{self}, "relu");
        res->back = [res, self]() {
            //Local derivative (d(relu(A))/dA_ij = 1 if A_ij > 0, else 0) — the ReLU derivative mask
            //Multiply with global derivative
            Mat mask = ReLU_derivative(self->matrix);
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(mask, res->gradient), "add");
        };
        return res;
    }

    ElemPtr sigmoid() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat sig_mat(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                sig_mat[r][c] = 1.0 / (1.0 + exp(-self->matrix[r][c]));
        auto res = make_shared<Matrix>(sig_mat, set<ElemPtr>{self}, "sigmoid");
        res->back = [res, self]() {
            //Local derivative (d(sigmoid(A))/dA_ij = sigmoid(A_ij) * (1 - sigmoid(A_ij)))
            //res->matrix already holds the sigmoid values so we reuse them directly
            //Multiply with global derivative
            int rows = res->matrix.size(), cols = res->matrix[0].size();
            Mat ones(rows, Vec(cols, 1.0));
            Mat one_minus_sig = matrix_addition_and_sub(ones, res->matrix, "sub");
            Mat local_grad = element_wise_multiplication(res->matrix, one_minus_sig);
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(local_grad, res->gradient), "add");
        };
        return res;
    }

    ElemPtr tanh_op() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat tanh_mat(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                tanh_mat[r][c] = tanh(self->matrix[r][c]);
        auto res = make_shared<Matrix>(tanh_mat, set<ElemPtr>{self}, "tanh");
        res->back = [res, self]() {
            //Local derivative (d(tanh(A))/dA_ij = 1 - tanh^2(A_ij))
            //res->matrix already holds the tanh values so we square them and subtract from 1
            //Multiply with global derivative
            int rows = res->matrix.size(), cols = res->matrix[0].size();
            Mat tanh_sq = element_wise_multiplication(res->matrix, res->matrix);
            Mat ones(rows, Vec(cols, 1.0));
            Mat local_grad = matrix_addition_and_sub(ones, tanh_sq, "sub");
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(local_grad, res->gradient), "add");
        };
        return res;
    }

    ElemPtr exp_op() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat exp_mat(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                exp_mat[r][c] = exp(self->matrix[r][c]);
        auto res = make_shared<Matrix>(exp_mat, set<ElemPtr>{self}, "exp");
        res->back = [res, self]() {
            //Local derivative (d(e^A)/dA_ij = e^A_ij) — the exponential is its own derivative
            //res->matrix already holds the exponential values so we reuse them directly
            //Multiply with global derivative
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(res->matrix, res->gradient), "add");
        };
        return res;
    }

    ElemPtr log_op() override {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat log_mat(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                log_mat[r][c] = log(self->matrix[r][c]);
        auto res = make_shared<Matrix>(log_mat, set<ElemPtr>{self}, "log");
        res->back = [res, self]() {
            //Local derivative (d(ln(A))/dA_ij = 1/A_ij)
            //Multiply with global derivative
            int rows = self->matrix.size(), cols = self->matrix[0].size();
            Mat ones(rows, Vec(cols, 1.0));
            Mat inv_self = element_wise_division_two_matrices(ones, self->matrix);
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(inv_self, res->gradient), "add");
        };
        return res;
    }

    //Sums each row across all its columns, producing a (rows x 1) column vector.
    //This is the matrix equivalent of summing a vector — used mainly to collapse batch gradients into bias gradients.
    MatrixPtr sum_cols() {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat summed(rows, Vec(1, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                summed[r][0] += self->matrix[r][c];
        auto res = make_shared<Matrix>(summed, set<ElemPtr>{self}, "sum_cols");
        res->back = [res, self, cols]() {
            //Local derivative: d(res_i)/d(A_ij) = 1 for all j
            //The upstream gradient res->gradient_i broadcasts back to every column in row i
            int rows = res->gradient.size();
            Mat broadcast(rows, Vec(cols, 0.0));
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    broadcast[r][c] = res->gradient[r][0];
            self->gradient = matrix_addition_and_sub(self->gradient, broadcast, "add");
        };
        return res;
    }

    //This is to prevent calling backprop manually. Kahn's topological sort flattens the graph
    //into a list that can be traversed in order so back() is called in the correct sequence.
    void backprop() {
        TopologicalSort topo;
        topo.backprop(shared_from_this());
    }
};
