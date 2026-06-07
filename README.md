# bare-metal-ml

**A machine learning library built from mathematical foundations — classical algorithms and a fully-connected neural network implemented from scratch in Python and C++, without any high-level ML libraries.**

---

## Overview

- Implements classical ML algorithms and a neural network entirely from scratch
- Custom linear algebra engine (LU decomposition, matrix inverse, determinant) — no NumPy in algorithm code
- Unified Python package (`bare_metal_ml`) with a clean import API
- C++ port of the neural network with OpenBLAS matrix multiplication
- Benchmarked against scikit-learn reference implementations

---

## Algorithms

### Implemented

| Algorithm | Description |
|---|---|
| **Neural Network** | Fully-connected network with ReLU, softmax output, mini-batch training, He initialization, inverted dropout, and Adam optimizer. Achieved **97.85% accuracy on MNIST** |
| **Gaussian Discriminant Analysis (GDA)** | Generative classifier using multivariate Gaussian distributions. Achieved ~97% accuracy on WDBC |
| **Gaussian Naive Bayes** | Generative classifier assuming feature independence. Achieved ~93% accuracy on WDBC |
| **Bernoulli Naive Bayes** | Binary bag-of-words text classifier with Laplace smoothing |
| **Multinomial Naive Bayes** | Count-based text classifier for spam detection |
| **Logistic Regression** | Discriminative binary classifier trained via gradient descent on binary cross-entropy loss |
| **Linear Regression** | Regression model trained via gradient descent |
| **K-Nearest Neighbours (KNN)** | Non-parametric classifier with Euclidean, Manhattan and cosine distance metrics |
| **KD-Tree** | Space-partitioning data structure for accelerated nearest-neighbour search |

### Coming Soon

- Support Vector Machine (SVM)
- Decision Trees
- Random Forest

---

## Neural Network — C++ Port

The neural network is also implemented in C++ (`Neural Network/Neural Network.cpp`) with:

- **OpenBLAS** (`cblas_dgemm`) for matrix multiplication — replaces the naive O(n³) implementation
- **Adam optimizer** with correct bias correction (`m̂ = m / (1 − β₁ᵗ)`, `v̂ = v / (1 − β₂ᵗ)`)
- **He initialization** (`std = sqrt(2 / fan_in)`) for stable ReLU gradients
- **Inverted dropout** with a configurable dropout rate
- **Mini-batch training** with per-epoch shuffling
- Weight save/load to CSV

Compile with:
```bash
g++ -O2 -std=c++17 "Neural Network/Neural Network.cpp" -lopenblas -o neural_net
```

---

## Mathematical Foundations

Everything implemented from scratch — no high-level ML library calls:

- LU decomposition via Doolittle's algorithm
- Matrix inverse via forward and backward substitution
- Determinant computation from the upper triangular factor
- Multivariate Gaussian PDF
- Gradient descent and Adam update rule
- Maximum likelihood estimation derivations for all generative models
- Softmax + cross-entropy gradient derivation for the neural network

---

## Unified API

```python
from bare_metal_ml import Network, Adam, FunctionType
from bare_metal_ml import KNN, KDTree
from bare_metal_ml import GaussianNaiveBayes, BernoulliNaiveBayes, MultinomialNaiveBayes

# Neural network
net = Network([128, 64, 10], input_size=784, function_type=FunctionType.RELU,
              optimizer=Adam(0.001), dropout_rate=0.1)
net.train(x_train, y_train, epochs=30, batch_size=256)
print(net.accuracy(x_test, y_test))

# KNN
knn = KNN(k=5)
knn.fit(x_train, y_train)
print(knn.accuracy(x_test, y_test))

# Naive Bayes
nb = GaussianNaiveBayes()
nb.fit(x_train, y_train)
print(nb.accuracy(x_test, y_test))
```

---

## Results

| Algorithm | Dataset | Accuracy |
|---|---|---|
| Neural Network | MNIST (10-class) | **97.85%** |
| GDA | WDBC (binary) | ~97% |
| Gaussian Naive Bayes | WDBC (binary) | ~93% |
| Bernoulli Naive Bayes | SMS Spam | — |
| Multinomial Naive Bayes | SMS Spam | — |
| Logistic Regression | UCI Iris | — |

---

## Datasets

| Dataset | Description | Link |
|---|---|---|
| **MNIST** | 70,000 handwritten digit images, 10 classes | [Yann LeCun](http://yann.lecun.com/exdb/mnist/) |
| **Wisconsin Breast Cancer (WDBC)** | 569 samples, 30 features, binary classification | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| **UCI Iris** | 150 samples, 4 features, multiclass | [UCI ML Repository](https://archive.ics.uci.edu/dataset/53/iris) |
| **SMS Spam Collection** | 5,572 SMS messages, binary spam/ham labels | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |

---

## How to Run

### Requirements

```
Python 3.10+
pandas    # data loading only
numpy     # data loading only
```

No high-level ML libraries (scikit-learn, TensorFlow, PyTorch, etc.) are used in any algorithm implementation.

### Running the Notebooks

```bash
git clone https://github.com/arora-abhinav/bare-metal-ml.git
cd bare-metal-ml
jupyter notebook
```

| Notebook | Algorithm |
|---|---|
| `Neural Network/Neural Network.ipynb` | Neural Network (MNIST) |
| `GDA/GDA.ipynb` | Gaussian Discriminant Analysis |
| `Naive Bayes/NaiveBayes.ipynb` | Gaussian Naive Bayes |
| `Naive Bayes/bernoulliNaiveBayes.ipynb` | Bernoulli Naive Bayes |
| `Naive Bayes/multinomial_naive_bayes.ipynb` | Multinomial Naive Bayes |
| `Logistic Regression/logistic-regression.ipynb` | Logistic Regression |
| `Linear Regression/l_regression.ipynb` | Linear Regression |
| `KNN/KNN.ipynb` | K-Nearest Neighbours |
| `KNN/KDTree.ipynb` | KD-Tree |

### Using the Package

```bash
pip install -e .
python -c "from bare_metal_ml import Network, Adam, FunctionType"
```

---

## Project Structure

```
bare-metal-ml/
├── bare_metal_ml/
│   ├── __init__.py
│   ├── neural_network.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── linalg.py
├── Neural Network/
│   ├── Neural Network.ipynb
│   └── Neural Network.cpp
├── GDA/
│   └── GDA.ipynb
├── Naive Bayes/
│   ├── NaiveBayes.ipynb
│   ├── bernoulliNaiveBayes.ipynb
│   └── multinomial_naive_bayes.ipynb
├── Logistic Regression/
│   └── logistic-regression.ipynb
├── Linear Regression/
│   └── l_regression.ipynb
├── KNN/
│   ├── KNN.ipynb
│   └── KDTree.ipynb
├── custom_math.py
└── custom_math.cpp
```

---

## Author

**Abhinav Arora**
University of Maryland — Computer Science
