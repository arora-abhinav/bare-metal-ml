#pragma once
#include <vector>
#include <string>
#include <set>
#include <map>
#include <deque>
#include <functional>
#include <memory>
#include <cmath>
#include "linalg.hpp"

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

    virtual ElemPtr add(ElemPtr other)    = 0;
    virtual ElemPtr mul(ElemPtr other)    = 0;
    virtual ElemPtr sub(ElemPtr other)    = 0;
    virtual ElemPtr truediv(ElemPtr other)= 0;
    virtual ElemPtr neg()                 = 0;
    virtual ElemPtr relu()                = 0;
    virtual ElemPtr sigmoid()             = 0;
    virtual ElemPtr tanh_op()             = 0;
    virtual ElemPtr exp_op()              = 0;
    virtual ElemPtr log_op()              = 0;
};


// ── TopologicalSort ───────────────────────────────────────────────────────────

//Kahn's topological sort flattens the DAG into a list ordered from root to leaves.
//back() is then called in that order so each node receives its upstream gradient
//before it propagates it to its own children.
class TopologicalSort {
public:
    //Kahn's topological sort — returns the sorted node list without running back()
    vector<ElemPtr> topological_sort(ElemPtr root_node) {
        map<Element*, int>     num_dependencies;
        map<Element*, ElemPtr> node_ptrs;

        function<void(ElemPtr)> collect_nodes = [&](ElemPtr node) {
            if (num_dependencies.find(node.get()) == num_dependencies.end()) {
                num_dependencies[node.get()] = 0;
                node_ptrs[node.get()] = node;
                for (auto& c : node->children)
                    collect_nodes(c);
            }
        };
        collect_nodes(root_node);

        for (auto& [ptr, count] : num_dependencies)
            for (auto& c : node_ptrs[ptr]->children)
                num_dependencies[c.get()]++;

        deque<Element*> q;
        for (auto& [ptr, count] : num_dependencies)
            if (count == 0)
                q.push_back(ptr);

        //Store ElemPtr so sorted_graph keeps nodes alive across batches —
        //raw Element* would dangle once feedforward recreates intermediate nodes
        vector<ElemPtr> final_res;
        while (!q.empty()) {
            int q_len = q.size();
            for (int i = 0; i < q_len; i++) {
                Element* node = q.back();
                q.pop_back();
                final_res.push_back(node_ptrs[node]);
                for (auto& c : node->children) {
                    num_dependencies[c.get()]--;
                    if (num_dependencies[c.get()] == 0)
                        q.push_back(c.get());
                }
            }
        }

        return final_res;
    }

    //Runs back() on a pre-sorted node list — call this with the cached result of topological_sort()
    void backprop(vector<ElemPtr>& final_res) {
        for (auto& node : final_res)
            node->back();
    }
};


// ── Scalar ────────────────────────────────────────────────────────────────────

class Scalar : public Element {
public:
    double digit;
    double gradient;

    Scalar(double digit, set<ElemPtr> children = {}, string op = "")
        : Element(children, op), digit(digit), gradient(0.0) {}

    ElemPtr add(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit + other->digit, set<ElemPtr>{self, other}, "add");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //The local derivative of an added expression w.r.t 1 element is 1
            self->gradient  += res->gradient;
            other->gradient += res->gradient;
        };
        return res;
    }

    ElemPtr mul(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit * other->digit, set<ElemPtr>{self, other}, "mul");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(ab)/da = b) and (d(ab)/db = a)
            self->gradient  += other->digit * res->gradient;
            other->gradient += self->digit  * res->gradient;
        };
        return res;
    }

    ElemPtr truediv(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit / other->digit, set<ElemPtr>{self, other}, "div");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(a/b)/da = 1/b) and (d(a/b)/db = -a*b^-2)
            self->gradient  += res->gradient * (1.0 / other->digit);
            other->gradient += self->digit * -(pow(other->digit, -2)) * res->gradient;
        };
        return res;
    }

    ElemPtr sub(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Scalar>(other_elem);
        auto self  = static_pointer_cast<Scalar>(shared_from_this());
        auto res   = make_shared<Scalar>(self->digit - other->digit, set<ElemPtr>{self, other}, "sub");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(a-b)/da = 1) and (d(a-b)/db = -1)
            self->gradient  += res->gradient;
            other->gradient += -res->gradient;
        };
        return res;
    }

    ElemPtr neg() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(-self->digit, set<ElemPtr>{self}, "neg");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(-a)/da = -1)
            self->gradient += -1.0 * res->gradient;
        };
        return res;
    }

    ElemPtr tanh_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(tanh(self->digit), set<ElemPtr>{self}, "tanh");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(tanh(a))/da = 1 - tanh^2(a))
            self->gradient += (1.0 - pow(tanh(self->digit), 2)) * res->gradient;
        };
        return res;
    }

    ElemPtr exp_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(exp(self->digit), set<ElemPtr>{self}, "exp");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(e^a)/da = e^a)
            self->gradient += exp(self->digit) * res->gradient;
        };
        return res;
    }

    ElemPtr log_op() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(log(self->digit), set<ElemPtr>{self}, "log");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(ln(a))/da = 1/a)
            self->gradient += (1.0 / self->digit) * res->gradient;
        };
        return res;
    }

    ElemPtr relu() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(max(0.0, self->digit), set<ElemPtr>{self}, "relu");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(relu(a))/da = 1 if a > 0, else 0)
            self->gradient += (self->digit > 0 ? 1.0 : 0.0) * res->gradient;
        };
        return res;
    }

    ElemPtr sigmoid() override {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        double sig = 1.0 / (1.0 + exp(-self->digit));
        auto res = make_shared<Scalar>(sig, set<ElemPtr>{self}, "sigmoid");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, sig]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a)))
            self->gradient += (sig * (1.0 - sig)) * res->gradient;
        };
        return res;
    }

    ScalarPtr pow_op(ScalarPtr other) {
        auto self = static_pointer_cast<Scalar>(shared_from_this());
        auto res  = make_shared<Scalar>(pow(self->digit, other->digit), set<ElemPtr>{self, other}, "pow");
        weak_ptr<Scalar> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(a^b)/da = b * a^(b-1)) and (d(a^b)/db = a^b * ln(a))
            self->gradient  += (other->digit * pow(self->digit, other->digit - 1)) * res->gradient;
            other->gradient += (res->digit * log(self->digit)) * res->gradient;
        };
        return res;
    }

    ScalarPtr radd(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->add(shared_from_this())); }
    ScalarPtr rmul(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->mul(shared_from_this())); }
    ScalarPtr rsub(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->sub(shared_from_this())); }
    ScalarPtr rdiv(double other) { return static_pointer_cast<Scalar>(make_shared<Scalar>(other)->truediv(shared_from_this())); }

    vector<ElemPtr> topo_sort() {
        TopologicalSort ts;
        return ts.topological_sort(shared_from_this());
    }

    void backprop(vector<ElemPtr>& sorted_graph) {
        TopologicalSort ts;
        ts.backprop(sorted_graph);
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

    //gradient is auto-initialised to a zero matrix of the same shape as matrix so that
    //every back() closure can safely accumulate into it via matrix_addition_and_sub.
    Matrix(Mat matrix, set<ElemPtr> children = {}, string op = "")
        : Element(children, op), matrix(matrix),
          gradient(matrix.empty() ? Mat{} : Mat(matrix.size(), Vec(matrix[0].size(), 0.0))) {}

    ElemPtr add(ElemPtr other_elem) override {
        auto other = static_pointer_cast<Matrix>(other_elem);
        auto self  = static_pointer_cast<Matrix>(shared_from_this());
        auto res   = make_shared<Matrix>(matrix_addition_and_sub(self->matrix, other->matrix, "add"),
                                         set<ElemPtr>{self, other}, "add");
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //The local derivative of matrix addition w.r.t either operand is the identity
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //dL/dA = dL/dC @ B^T — upstream gradient times B transposed on the right
            //dL/dB = A^T @ dL/dC — A transposed times upstream gradient on the left
            Mat term = matrix_with_matrix_multiplication(res->gradient, transpose_matrix(other->matrix));
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(A-B)/dA = I) and (d(A-B)/dB = -I)
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(-A)/dA = -I)
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative: d(A/B)/dA_ij = 1/B_ij and d(A/B)/dB_ij = -A_ij/B_ij^2
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, other]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative of Hadamard multiplication: d(A⊙B)/dA_ij = B_ij and d(A⊙B)/dB_ij = A_ij
            self->gradient  = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(other->matrix, res->gradient), "add");
            other->gradient = matrix_addition_and_sub(other->gradient,
                element_wise_multiplication(self->matrix, res->gradient), "add");
        };
        return res;
    }

    //The scalar is a raw number, not a Matrix node, so only self receives a gradient.
    MatrixPtr scalar_multiply(double scalar) {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(scalar_multiply_matrix(self->matrix, scalar),
                                        set<ElemPtr>{self}, "scalar_mul");
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, scalar]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(sA)/dA_ij = s)
            self->gradient = matrix_addition_and_sub(self->gradient,
                scalar_multiply_matrix(res->gradient, scalar), "add");
        };
        return res;
    }

    MatrixPtr transpose_op() {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        auto res  = make_shared<Matrix>(transpose_matrix(self->matrix), set<ElemPtr>{self}, "transpose");
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Transposing twice returns the original, so the upstream gradient is transposed back
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(relu(A))/dA_ij = 1 if A_ij > 0, else 0)
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(sigmoid(A))/dA_ij = sigmoid(A_ij) * (1 - sigmoid(A_ij)))
            //res->matrix already holds the sigmoid values so we reuse them directly
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(tanh(A))/dA_ij = 1 - tanh^2(A_ij))
            //res->matrix already holds the tanh values so we square them and subtract from 1
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(e^A)/dA_ij = e^A_ij) — the exponential is its own derivative
            //res->matrix already holds the exponential values so we reuse them directly
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
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self]() {
            auto res = res_weak.lock(); if (!res) return;
            //Local derivative (d(ln(A))/dA_ij = 1/A_ij)
            int rows = self->matrix.size(), cols = self->matrix[0].size();
            Mat ones(rows, Vec(cols, 1.0));
            Mat inv_self = element_wise_division_two_matrices(ones, self->matrix);
            self->gradient = matrix_addition_and_sub(self->gradient,
                element_wise_multiplication(inv_self, res->gradient), "add");
        };
        return res;
    }

    //Sums each row across all its columns, producing a (rows x 1) column vector.
    //Used mainly to collapse batch gradients into bias gradients.
    MatrixPtr sum_cols() {
        auto self = static_pointer_cast<Matrix>(shared_from_this());
        int rows = self->matrix.size(), cols = self->matrix[0].size();
        Mat summed(rows, Vec(1, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                summed[r][0] += self->matrix[r][c];
        auto res = make_shared<Matrix>(summed, set<ElemPtr>{self}, "sum_cols");
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, self, cols]() {
            auto res = res_weak.lock(); if (!res) return;
            //d(res_i)/d(A_ij) = 1 for all j — broadcast gradient back across the batch dimension
            int rows = res->gradient.size();
            Mat broadcast(rows, Vec(cols, 0.0));
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    broadcast[r][c] = res->gradient[r][0];
            self->gradient = matrix_addition_and_sub(self->gradient, broadcast, "add");
        };
        return res;
    }

    vector<ElemPtr> topo_sort() {
        TopologicalSort ts;
        return ts.topological_sort(shared_from_this());
    }

    void backprop(vector<ElemPtr>& sorted_graph) {
        TopologicalSort ts;
        ts.backprop(sorted_graph);
    }
};
