#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gda.hpp"
#include "linear_regression.hpp"
#include "logistic_regression.hpp"
#include "knn.hpp"
#include "naive_bayes.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "bare-metal-ml C++ backend";

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
