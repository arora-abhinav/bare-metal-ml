# bare-metal-ml

A machine learning library built from mathematical foundations — classical algorithms and a fully-connected neural network with a custom autograd engine, implemented from scratch in C++ with a clean Python API. No NumPy, no PyTorch, no scikit-learn in any algorithm code.

Every Python call runs C++ under the hood via a compiled pybind11 extension, with BLAS-accelerated matrix multiplication on Apple Silicon and x86.

---

## Table of Contents

1. [Installation](#installation)
2. [Neural Network](#neural-network)
   - [Data Format](#data-format)
   - [Optimizers](#optimizers)
   - [Built-in Activation Functions](#built-in-activation-functions)
   - [Custom Activation Functions](#custom-activation-functions)
   - [Building and Training](#building-and-training)
   - [Evaluation and Prediction](#evaluation-and-prediction)
   - [Saving and Loading Weights](#saving-and-loading-weights)
   - [Recommended Configurations](#recommended-configurations)
3. [Autograd Engine](#autograd-engine)
   - [Scalar](#scalar)
   - [Matrix](#matrix)
4. [Classical Algorithms](#classical-algorithms)
   - [Gaussian Discriminant Analysis](#gaussian-discriminant-analysis)
   - [K-Nearest Neighbours and KD-Tree](#k-nearest-neighbours-and-kd-tree)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [Naive Bayes](#naive-bayes)
5. [Linear Algebra Utilities](#linear-algebra-utilities)
6. [Project Structure](#project-structure)
7. [Benchmarks](#benchmarks)

---

## Installation

**Requirements:** Python 3.10+, a C++17 compiler, and pybind11.

```bash
git clone https://github.com/arora-abhinav/bare-metal-ml.git
cd bare-metal-ml
pip install -e .
```

The build step compiles the C++ extension automatically. Verify the installation:

```python
import bare_metal_ml as bml
print(bml.Network)   # <class 'bare_metal_ml._cpp.Network'>
```

All classes shown as `bare_metal_ml._cpp.*` are running pure C++.

---

## Neural Network

A fully-connected feedforward network with:
- Mini-batch training with per-epoch shuffling
- He initialization (`std = sqrt(2 / fan_in)`) for stable ReLU gradients
- Inverted dropout
- Softmax output with cross-entropy loss
- Adam and SGD optimizers
- Topo-sort cached autograd graph for efficient backpropagation
- Weight persistence (save / load JSON)

### Data Format

**This library uses column-major layout.** Data must be shaped `(features × samples)`, not the conventional `(samples × features)`.

```python
import numpy as np

# x_train shape: (samples, features) — standard layout
# Transpose before passing to bare_metal_ml
x_train_col = x_train.T.tolist()   # shape becomes (features, samples)

# Labels must be one-hot encoded, shape (classes, samples)
def one_hot(labels, n_classes=10):
    result = [[0.0] * len(labels) for _ in range(n_classes)]
    for i, label in enumerate(labels):
        result[label][i] = 1.0
    return result

y_train_oh = one_hot(y_train)
```

For inference, `predict()` and `accuracy()` also expect column-major input:

```python
x_test_col = x_test.T.tolist()
```

---

### Optimizers

Two optimizers are available. Pass one instance to `Network` at construction time.

#### Adam (recommended)

Adaptive moment estimation. Maintains per-parameter first and second moment estimates with bias correction.

```python
from bare_metal_ml import Adam

optimizer = Adam(learning_rate=0.001)   # default: 0.001
optimizer = Adam(0.01)
```

Hyperparameters β₁=0.9, β₂=0.999, ε=1e-8 are fixed at their standard values.

#### SGD

Vanilla stochastic gradient descent.

```python
from bare_metal_ml import SGD

optimizer = SGD(learning_rate=0.01)    # default: 0.01
```

---

### Built-in Activation Functions

Three activation functions are available as `FunctionType` enum values.

```python
from bare_metal_ml import FunctionType

FunctionType.RELU      # max(0, x) — default, recommended for deep networks
FunctionType.SIGMOID   # 1 / (1 + e^-x)
FunctionType.TANH      # tanh(x)
```

Pass to `Network` via the `function_type` keyword argument. All hidden layers use the chosen activation; the output layer always uses softmax.

---

### Custom Activation Functions

You can inject any element-wise activation function by subclassing `ActivationFunction` and implementing two methods: `forward(x)` for the forward pass and `derivative(x)` for the local derivative used during backpropagation. Both operate on a single scalar `x`.

```python
from bare_metal_ml import ActivationFunction, Network, Adam

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: float) -> float:
        return x if x > 0 else self.alpha * x

    def derivative(self, x: float) -> float:
        return 1.0 if x > 0 else self.alpha


class Swish(ActivationFunction):
    """x * sigmoid(x)"""
    def __init__(self):
        super().__init__()

    def forward(self, x: float) -> float:
        import math
        s = 1.0 / (1.0 + math.exp(-x))
        return x * s

    def derivative(self, x: float) -> float:
        import math
        s = 1.0 / (1.0 + math.exp(-x))
        return s + x * s * (1.0 - s)


# Pass via the `activation` argument — overrides `function_type`
my_act = LeakyReLU(alpha=0.1)
net = Network(
    layer_num        = 3,
    neurons_in_layers= [128, 64, 10],
    initial_input    = x_train_col,
    optimizer        = Adam(0.001),
    dropout_rate     = 0.2,
    activation       = my_act,        # custom activation takes priority
)
```

The C++ training loop calls back into your Python `forward()` and `derivative()` methods transparently via a pybind11 virtual dispatch trampoline, so any Python-level logic (math, conditional branches) works as expected.

---

### Building and Training

```python
from bare_metal_ml import Network, Adam, FunctionType

adam = Adam(0.001)

net = Network(
    layer_num         = 3,              # number of layers (including output)
    neurons_in_layers = [128, 64, 10],  # neurons per layer
    initial_input     = x_train_col,    # (features × samples) list-of-lists
    optimizer         = adam,
    dropout_rate      = 0.2,            # fraction of neurons to drop (0.0 = no dropout)
    function_type     = FunctionType.RELU,
)

net.train_loop(
    epochs     = 20,
    train_labels = y_train_oh,  # one-hot (classes × samples)
    batch_size = 64,
)
```

`dropout_rate` is applied during training only. Inference automatically disables dropout.

---

### Evaluation and Prediction

```python
# accuracy() returns a float in [0, 1]
acc = net.accuracy(x_test_col, y_test_labels)
print(f"Test accuracy: {acc * 100:.2f}%")

# predict() returns a flat list of integer class indices
predictions = net.predict(x_test_col)
```

`y_test_labels` passed to `accuracy()` is a flat list of integer class indices (not one-hot).

---

### Saving and Loading Weights

```python
net.save_weights("weights.json")       # saves W and b for every layer

net.load_weights("weights.json")       # restores weights in-place
```

Weights are serialised as JSON arrays. The file path defaults to `"weights.json"` if omitted.

---

### Recommended Configurations

Based on benchmarks against PyTorch and Keras on MNIST (48 000 train / 12 000 test):

| Task | Architecture | Optimizer | Dropout | Notes |
|---|---|---|---|---|
| Image classification (MNIST-scale) | `[256, 128, n_classes]` | Adam 0.001 | 0.2 | Strong baseline |
| Tabular data, small dataset | `[64, 32, n_classes]` | Adam 0.001 | 0.0–0.1 | Avoid heavy dropout on small data |
| Tabular data, large dataset | `[256, 128, 64, n_classes]` | Adam 0.001 | 0.2–0.3 | He init handles depth well |
| Binary classification | `[64, 32, 2]` | Adam 0.001 | 0.1 | Or use LogisticRegression for linear problems |
| Fast prototyping | `[128, n_classes]` | SGD 0.01 | 0.0 | Fewer parameters, faster iteration |

General rules:
- **Adam over SGD** for most tasks — faster convergence, less sensitive to learning rate.
- **ReLU over Sigmoid/Tanh** for hidden layers — He init is matched to ReLU; vanishing gradients are less of an issue.
- **Dropout 0.1–0.3** for larger networks on image data; reduce or remove for tabular data with fewer features.
- **Batch size 64–256** — smaller batches generalise better but train slower.

---

## Autograd Engine

`Scalar` and `Matrix` are first-class computation graph nodes. Every arithmetic operation creates a new node that records its children and a backward closure. Calling `topo_sort()` then `backprop()` propagates gradients through the graph.

### Scalar

Operates on single floating-point values.

```python
from bare_metal_ml import Scalar

a = Scalar(2.0)
b = Scalar(3.0)

# Forward pass — builds the computation graph
c = a * b        # 6.0
d = c + Scalar(1.0)   # 7.0

# Seed the root gradient and backpropagate
d.gradient = 1.0
graph = d.topo_sort()
d.backprop(graph)

print(a.gradient)   # 3.0  (d(d)/d(a) = b = 3)
print(b.gradient)   # 2.0  (d(d)/d(b) = a = 2)
```

**Available operations:**

| Python syntax | Method | Notes |
|---|---|---|
| `a + b` | `__add__` | |
| `a * b` | `__mul__` | |
| `a - b` | `__sub__` | |
| `a / b` | `__truediv__` | |
| `-a` | `__neg__` | |
| `a.pow_op(b)` | `pow_op` | aᵇ |
| `a.relu()` | `relu` | max(0, x) |
| `a.sigmoid()` | `sigmoid` | 1/(1+e⁻ˣ) |
| `a.tanh_op()` | `tanh_op` | tanh(x) |
| `a.exp_op()` | `exp_op` | eˣ |
| `a.log_op()` | `log_op` | ln(x) |
| `3.0 + a` | `__radd__` | scalar on left |
| `3.0 * a` | `__rmul__` | scalar on left |

**Attributes:**
- `a.digit` — the scalar value (read/write)
- `a.gradient` — accumulated gradient (read/write, initialised to 0.0)
- `a.operation` — string name of the op that created this node (read-only)

---

### Matrix

Operates on 2-D matrices (list-of-lists). Gradients are matrices of the same shape.

```python
from bare_metal_ml import Matrix

A = Matrix([[1.0, 2.0],
            [3.0, 4.0]])

B = Matrix([[5.0, 6.0],
            [7.0, 8.0]])

# Matrix multiplication (not element-wise)
C = A * B

# Seed and backpropagate
C.gradient = [[1.0, 1.0], [1.0, 1.0]]
graph = C.topo_sort()
C.backprop(graph)

print(A.gradient)   # dL/dA = dL/dC @ B^T
print(B.gradient)   # dL/dB = A^T @ dL/dC
```

**Available operations:**

| Python syntax / method | Behaviour |
|---|---|
| `A + B` | Element-wise addition |
| `A * B` | **Matrix multiplication** (not Hadamard) |
| `A - B` | Element-wise subtraction |
| `A / B` | Element-wise division |
| `-A` | Negate all elements |
| `A.element_wise_mult(B)` | Hadamard (element-wise) product |
| `A.scalar_multiply(s)` | Multiply every element by scalar `s` |
| `A.transpose_op()` | Transpose |
| `A.sum_cols()` | Sum across columns → (rows × 1) vector |
| `A.relu()` | Element-wise ReLU |
| `A.sigmoid()` | Element-wise sigmoid |
| `A.tanh_op()` | Element-wise tanh |
| `A.exp_op()` | Element-wise eˣ |
| `A.log_op()` | Element-wise ln(x) |

**Attributes:**
- `A.matrix` — the 2-D list of values (read/write)
- `A.gradient` — 2-D list of gradients, same shape (read/write)
- `A.operation` — string op name (read-only)

---

## Classical Algorithms

### Gaussian Discriminant Analysis

Generative classifier. Fits a multivariate Gaussian per class and classifies by maximum likelihood.

```python
from bare_metal_ml import GDA

gda = GDA(positive_class="M")   # label of the positive class (binary classification)
gda.fit(x_train, y_train)

prediction  = gda.predict_one(x_sample)
predictions = gda.predict(x_test)
acc         = gda.accuracy(x_test, y_test)
```

`x_train` is a list of feature vectors; `y_train` is a list of string labels.

---

### K-Nearest Neighbours and KD-Tree

```python
from bare_metal_ml import KNN, KDTree, euclidean, manhattan, cosine

# KNN — brute-force, O(n) per query
knn = KNN(k=5, metric="euclidean")   # metric: "euclidean" | "manhattan" | "cosine"
knn.fit(x_train, y_train)

label       = knn.predict_one(x_sample)
predictions = knn.predict(x_test)
acc         = knn.accuracy(x_test, y_test)

# KD-Tree — O(log n) average per query
kdt = KDTree()
kdt.fit(x_train, y_train)

label       = kdt.predict_one(x_sample, k=5)
predictions = kdt.predict(x_test, k=5)
acc         = kdt.accuracy(x_test, y_test, k=5)

# Distance functions are also available standalone
d = euclidean([1.0, 2.0], [4.0, 6.0])   # 5.0
d = manhattan([1.0, 2.0], [4.0, 6.0])   # 7.0
d = cosine([1.0, 0.0], [0.0, 1.0])      # 1.0 (maximally dissimilar)
```

---

### Linear Regression

Trained via gradient descent on mean squared error.

```python
from bare_metal_ml import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train, learning_rate=0.01, iterations=1000)

predictions = lr.predict(x_test)
mse         = lr.mse(x_test, y_test)
```

`x_train` is a list of feature vectors; `y_train` is a list of scalar targets.

---

### Logistic Regression

Binary classifier trained via gradient descent on binary cross-entropy.

```python
from bare_metal_ml import LogisticRegression

logr = LogisticRegression(positive_class="spam")
logr.fit(x_train, y_train, learning_rate=0.001, iterations=1000)

probabilities = logr.predict_proba(x_test)
predictions   = logr.predict(x_test, threshold=0.5)
acc           = logr.accuracy(x_test, y_test, threshold=0.5)
```

---

### Naive Bayes

Three variants for different data types.

```python
from bare_metal_ml import GaussianNaiveBayes, BernoulliNaiveBayes, MultinomialNaiveBayes

# Gaussian — continuous features (e.g., measurements)
gnb = GaussianNaiveBayes()
gnb.fit(x_train, y_train)
acc = gnb.accuracy(x_test, y_test)

# Bernoulli — binary bag-of-words features (text classification)
bnb = BernoulliNaiveBayes(vocab_size=1000)
bnb.fit(x_train, y_train)   # x_train: list of raw text strings
acc = bnb.accuracy(x_test, y_test)

# Multinomial — word count features (text classification)
mnb = MultinomialNaiveBayes(vocab_size=1000)
mnb.fit(x_train, y_train)
acc = mnb.accuracy(x_test, y_test)
```

All three share the same interface: `fit`, `predict_one`, `predict`, `accuracy`.

---

## Linear Algebra Utilities

All functions are C++ and available under the `bare_metal_ml.linalg` namespace.

```python
from bare_metal_ml import linalg

# Core matrix operations
C   = linalg.matrix_with_matrix_multiplication(A, B)
S   = linalg.matrix_addition_and_sub(A, B, "add")   # "add" or "sub"
S   = linalg.scalar_multiply_matrix(A, 3.0)
H   = linalg.element_wise_multiplication(A, B)
D   = linalg.element_wise_division_two_matrices(A, B)
R   = linalg.element_wise_roots(A, 2.0)             # element-wise sqrt
T   = linalg.transpose_matrix(A)
M   = linalg.ReLU_derivative(A)                     # 1 where A > 0, else 0
v   = linalg.sum_across_column(A)                   # row-wise sum → vector

# Utility functions
outer = linalg.matrix_product_from_vector_and_transpose(n, v)  # outer product v @ v^T
diff  = linalg.calculate_vector(v1, v2)             # v1 - v2
dot   = linalg.scalar_product_from_transpose_and_vector(v1, v2)  # dot product
mv    = linalg.matrix_product_with_matrix_and_vector(A, v, rows, cols)

# Matrix decomposition and inverse
L, U  = linalg.LU_decomposition(A, n)              # Doolittle LU factorisation
det   = linalg.calculate_determinant(U, n)          # determinant from upper triangular
A_inv = linalg.matrix_inverse(L, U, n)             # inverse via forward/back substitution
A_reg = linalg.regularize(A, n, epsilon=1e-6)      # add ε to diagonal for numerical stability
```

All inputs and outputs are Python `list[list[float]]` for matrices and `list[float]` for vectors.

---

## Project Structure

```
bare-metal-ml/
├── bare_metal_ml/
│   ├── __init__.py          # public API — imports everything from _cpp
│   ├── _cpp.*.so            # compiled C++ extension (built on install)
│   └── cpp/
│       ├── autograd.hpp     # Scalar, Matrix, TopologicalSort
│       ├── linalg.hpp       # all math operations (BLAS matmul)
│       ├── neural_network.hpp
│       ├── gda.hpp
│       ├── knn.hpp
│       ├── linear_regression.hpp
│       ├── logistic_regression.hpp
│       ├── naive_bayes.hpp
│       └── bindings.cpp     # pybind11 module definition
├── notebooks/
│   ├── neural_network/      # reference implementation + MNIST data
│   ├── gda/
│   ├── knn/
│   ├── linear_regression/
│   ├── logistic_regression/
│   └── naive_bayes/
├── benchmarks/
│   ├── benchmark_neural_network.py
│   ├── benchmark_classifiers.py
│   ├── benchmark_linear_regression.py
│   └── benchmark_naive_bayes.py
├── pyproject.toml
└── setup.py
```

The `notebooks/` directory contains the original Python reference implementations. They are not used by the library but document the mathematical derivations behind each algorithm.

---

## Benchmarks

Benchmarked on MNIST (48 000 train / 12 000 test), architecture `784 → 128 → 64 → 10`, Adam lr=0.01, dropout=0.2, 10 epochs, batch size 64:

| Model | Accuracy | Time |
|---|---|---|
| bare-metal-ml | ~96% | ~43s |
| PyTorch | ~96% | ~7s |
| Keras (PyTorch backend) | ~96% | ~49s |

Accuracy is on par with PyTorch and Keras. The speed gap comes from the Python↔C++ boundary: each matrix operation in the autograd graph is a separate pybind11 dispatch. The flexibility of the autograd design (arbitrary activation functions, custom graph topologies) is the deliberate trade-off.

---

## Author

**Abhinav Arora**  
University of Maryland — Computer Science
