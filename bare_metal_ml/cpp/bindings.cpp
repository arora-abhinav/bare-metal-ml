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
    //py::keep_alive<1,6> keeps the optimizer Python object alive for the lifetime
    //of the Network, since Network stores a raw Optimizer* internally.
    //py::keep_alive<1,8> does the same for a custom ActivationFunction instance.
    py::class_<Network>(m, "Network")
        .def(py::init([](int layer_num, vector<int> neurons, Mat x_train,
                         FunctionType ft, Adam* adam,
                         double dropout, ActivationFunction* custom_act) {
                return new Network(layer_num, neurons, x_train, ft, adam, dropout, custom_act);
             }),
             py::arg("layer_num"), py::arg("neurons_in_layers"), py::arg("initial_input"),
             py::arg("function_type") = FunctionType::RELU,
             py::arg("optimizer"),
             py::arg("dropout_rate") = 0.0,
             py::arg("custom_activation") = nullptr,
             py::keep_alive<1, 6>(),
             py::keep_alive<1, 8>())
        .def(py::init([](int layer_num, vector<int> neurons, Mat x_train,
                         FunctionType ft, SGD* sgd,
                         double dropout, ActivationFunction* custom_act) {
                return new Network(layer_num, neurons, x_train, ft, sgd, dropout, custom_act);
             }),
             py::arg("layer_num"), py::arg("neurons_in_layers"), py::arg("initial_input"),
             py::arg("function_type") = FunctionType::RELU,
             py::arg("optimizer"),
             py::arg("dropout_rate") = 0.0,
             py::arg("custom_activation") = nullptr,
             py::keep_alive<1, 6>(),
             py::keep_alive<1, 8>())
        .def("train_loop",    &Network::train_loop,
             py::arg("epochs"), py::arg("train_labels"), py::arg("batch_size"))
        .def("accuracy",      &Network::test_accuracy,
             py::arg("x_test"), py::arg("y_test"))
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
