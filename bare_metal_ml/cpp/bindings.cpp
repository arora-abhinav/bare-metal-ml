#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "gda.hpp"
#include "linear_regression.hpp"
#include "logistic_regression.hpp"
#include "knn.hpp"
#include "naive_bayes.hpp"
#include "neural_network.hpp"

namespace py = pybind11;

//Trampoline class — lets Python subclass ActivationFunction and override forward() and derivative().
//PYBIND11_OVERRIDE_PURE calls back into Python when the virtual method is invoked from C++,
//so custom activations defined in Python work transparently inside the C++ training loop.
struct PyActivationFunction : ActivationFunction {
    using ActivationFunction::ActivationFunction;
    double forward(double x) override {
        PYBIND11_OVERRIDE_PURE(double, ActivationFunction, forward, x);
    }
    double derivative(double x) override {
        PYBIND11_OVERRIDE_PURE(double, ActivationFunction, derivative, x);
    }
};

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "bare-metal-ml C++ backend";

    // ── linalg ────────────────────────────────────────────────────────────────
    m.def("matrix_with_matrix_multiplication",    &matrix_with_matrix_multiplication);
    m.def("matrix_addition_and_sub",              &matrix_addition_and_sub);
    m.def("scalar_multiply_matrix",               &scalar_multiply_matrix);
    m.def("element_wise_multiplication",          &element_wise_multiplication);
    m.def("element_wise_division_two_matrices",   &element_wise_division_two_matrices);
    m.def("element_wise_roots",                   &element_wise_roots);
    m.def("transpose_matrix",                     &transpose_matrix);
    m.def("ReLU_derivative",                      &ReLU_derivative);
    m.def("sum_across_column",                    &sum_across_column);
    m.def("matrix_product_from_vector_and_transpose", &matrix_product_from_vector_and_transpose);
    m.def("calculate_vector",                     &calculate_vector);
    m.def("LU_decomposition",                     &LU_decomposition);
    m.def("calculate_determinant",                &calculate_determinant);
    m.def("matrix_inverse",                       &matrix_inverse);
    m.def("matrix_product_with_matrix_and_vector",&matrix_product_with_matrix_and_vector);
    m.def("scalar_product_from_transpose_and_vector", &scalar_product_from_transpose_and_vector);
    m.def("regularize",                           &regularize);

    // ── distance metrics ──────────────────────────────────────────────────────
    m.def("euclidean", &euclidean);
    m.def("manhattan", &manhattan);
    m.def("cosine",    &cosine);

    // ── autograd ──────────────────────────────────────────────────────────────
    py::class_<Element, shared_ptr<Element>>(m, "Element")
        .def_readonly("operation", &Element::operation);

    py::class_<Scalar, Element, shared_ptr<Scalar>>(m, "Scalar")
        .def(py::init<double>(), py::arg("digit"))
        .def_readwrite("digit",     &Scalar::digit)
        .def_readwrite("gradient",  &Scalar::gradient)
        .def("__add__",      [](shared_ptr<Scalar> s, shared_ptr<Scalar> o) {
            return static_pointer_cast<Scalar>(s->add(o)); })
        .def("__mul__",      [](shared_ptr<Scalar> s, shared_ptr<Scalar> o) {
            return static_pointer_cast<Scalar>(s->mul(o)); })
        .def("__sub__",      [](shared_ptr<Scalar> s, shared_ptr<Scalar> o) {
            return static_pointer_cast<Scalar>(s->sub(o)); })
        .def("__truediv__",  [](shared_ptr<Scalar> s, shared_ptr<Scalar> o) {
            return static_pointer_cast<Scalar>(s->truediv(o)); })
        .def("__neg__",      [](shared_ptr<Scalar> s) {
            return static_pointer_cast<Scalar>(s->neg()); })
        .def("relu",         [](shared_ptr<Scalar> s) { return static_pointer_cast<Scalar>(s->relu()); })
        .def("sigmoid",      [](shared_ptr<Scalar> s) { return static_pointer_cast<Scalar>(s->sigmoid()); })
        .def("tanh_op",      [](shared_ptr<Scalar> s) { return static_pointer_cast<Scalar>(s->tanh_op()); })
        .def("exp_op",       [](shared_ptr<Scalar> s) { return static_pointer_cast<Scalar>(s->exp_op()); })
        .def("log_op",       [](shared_ptr<Scalar> s) { return static_pointer_cast<Scalar>(s->log_op()); })
        .def("pow_op",       &Scalar::pow_op)
        .def("__radd__",     [](shared_ptr<Scalar> s, double o) { return s->radd(o); })
        .def("__rmul__",     [](shared_ptr<Scalar> s, double o) { return s->rmul(o); })
        .def("__rsub__",     [](shared_ptr<Scalar> s, double o) { return s->rsub(o); })
        .def("__rtruediv__", [](shared_ptr<Scalar> s, double o) { return s->rdiv(o); })
        .def("topo_sort",    &Scalar::topo_sort)
        .def("backprop",     [](shared_ptr<Scalar> s, vector<ElemPtr> g) { s->backprop(g); });

    py::class_<Matrix, Element, shared_ptr<Matrix>>(m, "Matrix")
        .def(py::init<Mat>(), py::arg("matrix"))
        .def_readwrite("matrix",    &Matrix::matrix)
        .def_readwrite("gradient",  &Matrix::gradient)
        .def("__add__",         [](shared_ptr<Matrix> s, shared_ptr<Matrix> o) {
            return static_pointer_cast<Matrix>(s->add(o)); })
        .def("__mul__",         [](shared_ptr<Matrix> s, shared_ptr<Matrix> o) {
            return static_pointer_cast<Matrix>(s->mul(o)); })
        .def("__sub__",         [](shared_ptr<Matrix> s, shared_ptr<Matrix> o) {
            return static_pointer_cast<Matrix>(s->sub(o)); })
        .def("__truediv__",     [](shared_ptr<Matrix> s, shared_ptr<Matrix> o) {
            return static_pointer_cast<Matrix>(s->truediv(o)); })
        .def("__neg__",         [](shared_ptr<Matrix> s) {
            return static_pointer_cast<Matrix>(s->neg()); })
        .def("relu",            [](shared_ptr<Matrix> s) { return static_pointer_cast<Matrix>(s->relu()); })
        .def("sigmoid",         [](shared_ptr<Matrix> s) { return static_pointer_cast<Matrix>(s->sigmoid()); })
        .def("tanh_op",         [](shared_ptr<Matrix> s) { return static_pointer_cast<Matrix>(s->tanh_op()); })
        .def("exp_op",          [](shared_ptr<Matrix> s) { return static_pointer_cast<Matrix>(s->exp_op()); })
        .def("log_op",          [](shared_ptr<Matrix> s) { return static_pointer_cast<Matrix>(s->log_op()); })
        .def("element_wise_mult", &Matrix::element_wise_mult)
        .def("scalar_multiply",   &Matrix::scalar_multiply)
        .def("transpose_op",      &Matrix::transpose_op)
        .def("sum_cols",          &Matrix::sum_cols)
        .def("topo_sort",         &Matrix::topo_sort)
        .def("backprop",          [](shared_ptr<Matrix> s, vector<ElemPtr> g) { s->backprop(g); });

    // ── ActivationFunction ────────────────────────────────────────────────────
    //Exposed with the PyActivationFunction trampoline so Python can subclass it.
    //Subclass, override forward() and derivative() as scalar element-wise functions,
    //and pass an instance to Network(..., custom_activation=my_act).
    py::class_<ActivationFunction, PyActivationFunction>(m, "ActivationFunction")
        .def(py::init<>())
        .def("forward",     &ActivationFunction::forward)
        .def("derivative",  &ActivationFunction::derivative);

    py::class_<ReLU, ActivationFunction>(m, "ReLU")
        .def(py::init<>());

    py::class_<Sigmoid, ActivationFunction>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<Tanh, ActivationFunction>(m, "Tanh")
        .def(py::init<>());

    py::enum_<FunctionType>(m, "FunctionType")
        .value("RELU",    FunctionType::RELU)
        .value("SIGMOID", FunctionType::SIGMOID)
        .value("TANH",    FunctionType::TANH)
        .export_values();

    // ── Optimizers ────────────────────────────────────────────────────────────
    py::class_<Adam>(m, "Adam")
        .def(py::init<double>(), py::arg("learning_rate") = 0.001)
        .def_readwrite("learning_rate", &Adam::learning_rate);

    py::class_<SGD>(m, "SGD")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01)
        .def_readwrite("learning_rate", &SGD::learning_rate);

    // ── Network ───────────────────────────────────────────────────────────────
    //Arg order matches the Python Network API: optimizer before dropout_rate before function_type.
    //keep_alive<1,5> keeps optimizer alive (5th arg after self,layer_num,neurons,x_train).
    //keep_alive<1,8> keeps activation alive (8th arg after ...,dropout,function_type,activation).
    py::class_<Network>(m, "Network")
        .def(py::init([](int layer_num, vector<int> neurons, Mat x_train,
                         Adam* adam, double dropout,
                         FunctionType ft, ActivationFunction* custom_act,
                         string weights_dir) {
                auto* net = new Network(layer_num, neurons, x_train, ft, adam, dropout, custom_act);
                net->weights_dir = weights_dir;
                return net;
             }),
             py::arg("layer_num"), py::arg("neurons_in_layers"), py::arg("initial_input"),
             py::arg("optimizer"),
             py::arg("dropout_rate") = 0.0,
             py::arg("function_type") = FunctionType::RELU,
             py::arg("activation") = nullptr,
             py::arg("weights_dir") = "",
             py::keep_alive<1, 5>(),
             py::keep_alive<1, 8>())
        .def(py::init([](int layer_num, vector<int> neurons, Mat x_train,
                         SGD* sgd, double dropout,
                         FunctionType ft, ActivationFunction* custom_act,
                         string weights_dir) {
                auto* net = new Network(layer_num, neurons, x_train, ft, sgd, dropout, custom_act);
                net->weights_dir = weights_dir;
                return net;
             }),
             py::arg("layer_num"), py::arg("neurons_in_layers"), py::arg("initial_input"),
             py::arg("optimizer"),
             py::arg("dropout_rate") = 0.0,
             py::arg("function_type") = FunctionType::RELU,
             py::arg("activation") = nullptr,
             py::arg("weights_dir") = "",
             py::keep_alive<1, 5>(),
             py::keep_alive<1, 8>())
        .def("train_loop",    &Network::train_loop,
             py::arg("epochs"), py::arg("train_labels"), py::arg("batch_size"))
        .def("accuracy",      &Network::test_accuracy,
             py::arg("x_test"), py::arg("y_test"))
        .def("predict",       &Network::predict,
             py::arg("x_test"))
        .def("save_weights",  &Network::save_weights,
             py::arg("path") = "weights.json")
        .def("load_weights",  &Network::load_weights,
             py::arg("path") = "weights.json")
        .def_readwrite("dropout_rate", &Network::dropout_rate);

    // ── GDA ──────────────────────────────────────────────────────────────────
    py::class_<GDA>(m, "GDA")
        .def(py::init<string>(), py::arg("positive_class") = "")
        .def("fit",         &GDA::fit)
        .def("predict_one", &GDA::predict_one)
        .def("predict",     &GDA::predict)
        .def("accuracy",    &GDA::accuracy);

    // ── Linear Regression ─────────────────────────────────────────────────────
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit",     &LinearRegression::fit,
             py::arg("x_train"), py::arg("y_train"),
             py::arg("learning_rate") = 0.01, py::arg("iterations") = 100)
        .def("predict", &LinearRegression::predict)
        .def("mse",     &LinearRegression::mse);

    // ── Logistic Regression ───────────────────────────────────────────────────
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<string>(), py::arg("positive_class") = "")
        .def("fit",           &LogisticRegression::fit,
             py::arg("x_train"), py::arg("y_train"),
             py::arg("learning_rate") = 0.001, py::arg("iterations") = 1000)
        .def("predict_proba", &LogisticRegression::predict_proba)
        .def("predict",       &LogisticRegression::predict,
             py::arg("x_test"), py::arg("threshold") = 0.5)
        .def("accuracy",      &LogisticRegression::accuracy,
             py::arg("x_test"), py::arg("y_test"), py::arg("threshold") = 0.5);

    // ── KNN ───────────────────────────────────────────────────────────────────
    py::class_<KNN>(m, "KNN")
        .def(py::init<int, string>(),
             py::arg("k") = 5, py::arg("metric") = "euclidean")
        .def("fit",         &KNN::fit)
        .def("predict_one", &KNN::predict_one)
        .def("predict",     &KNN::predict)
        .def("accuracy",    &KNN::accuracy);

    // ── KDTree ────────────────────────────────────────────────────────────────
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<>())
        .def("fit",         &KDTree::fit)
        .def("predict_one", &KDTree::predict_one, py::arg("query"), py::arg("k") = 1)
        .def("predict",     &KDTree::predict,     py::arg("x_test"), py::arg("k") = 1)
        .def("accuracy",    &KDTree::accuracy,    py::arg("x_test"), py::arg("y_test"), py::arg("k") = 1);

    // ── Gaussian Naive Bayes ──────────────────────────────────────────────────
    py::class_<GaussianNaiveBayes>(m, "GaussianNaiveBayes")
        .def(py::init<>())
        .def("fit",         &GaussianNaiveBayes::fit)
        .def("predict_one", &GaussianNaiveBayes::predict_one)
        .def("predict",     &GaussianNaiveBayes::predict)
        .def("accuracy",    &GaussianNaiveBayes::accuracy);

    // ── Bernoulli Naive Bayes ─────────────────────────────────────────────────
    py::class_<BernoulliNaiveBayes>(m, "BernoulliNaiveBayes")
        .def(py::init<int>(), py::arg("vocab_size") = 1000)
        .def("fit",         &BernoulliNaiveBayes::fit)
        .def("predict_one", &BernoulliNaiveBayes::predict_one)
        .def("predict",     &BernoulliNaiveBayes::predict)
        .def("accuracy",    &BernoulliNaiveBayes::accuracy);

    // ── Multinomial Naive Bayes ───────────────────────────────────────────────
    py::class_<MultinomialNaiveBayes>(m, "MultinomialNaiveBayes")
        .def(py::init<int>(), py::arg("vocab_size") = 1000)
        .def("fit",         &MultinomialNaiveBayes::fit)
        .def("predict_one", &MultinomialNaiveBayes::predict_one)
        .def("predict",     &MultinomialNaiveBayes::predict)
        .def("accuracy",    &MultinomialNaiveBayes::accuracy);
}
