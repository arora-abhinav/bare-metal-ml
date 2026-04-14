# bare-metal-ml

**A machine learning framework built from mathematical foundations, implementing classical ML algorithms from scratch in Python and C++ without the use of any high-level ML libraries.**

---

## Overview

- Educational ML framework implementing algorithms from scratch
- All linear algebra, probability distributions and optimization techniques derived and implemented manually
- Benchmarked against scikit-learn reference implementations
- C++ port with pybind11 Python bindings coming soon

---

## Algorithms

### Currently Implemented

| Algorithm | Description |
|---|---|
| **Gaussian Discriminant Analysis (GDA)** | Generative classifier using multivariate Gaussian distributions. Achieved ~97% accuracy on WDBC |
| **Gaussian Naive Bayes** | Generative classifier assuming feature independence. Achieved ~93% accuracy on WDBC |
| **Logistic Regression** | Discriminative classifier trained via gradient descent on binary cross entropy loss. Applied to UCI Iris dataset |
| **Linear Regression** | Regression model trained via gradient descent |

### Coming Soon

- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Support Vector Machine
- Decision Trees
- Random Forest
- Neural Network

---

## Mathematical Foundations

Everything implemented from scratch — no high-level ML library calls:

- LU Decomposition via Doolittle's algorithm
- Matrix inverse via forward and backward substitution
- Determinant computation
- Multivariate Gaussian PDF
- Gradient descent
- Maximum likelihood estimation derivations for all models

---

## Datasets

| Dataset | Description | Link |
|---|---|---|
| **Wisconsin Breast Cancer Dataset (WDBC)** | 569 samples, 30 features, binary classification | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| **UCI Iris** | 150 samples, 4 features, binary classification | [UCI ML Repository](https://archive.ics.uci.edu/dataset/53/iris) |

---

## Results

| Algorithm | Dataset | Accuracy |
|---|---|---|
| GDA | WDBC | ~97% |
| Gaussian Naive Bayes | WDBC | ~93% |
| Logistic Regression | Iris | xx% |
| Linear Regression | — | — |

---

## How to Run

### Requirements

- Python 3
- `pandas` — data loading only
- `numpy` — data loading only

> No high-level ML libraries (scikit-learn, TensorFlow, PyTorch, etc.) are used in the algorithm implementations.

### Running the Notebooks

1. Clone this repository:
   ```bash
   git clone https://github.com/arora-abhinav/bare-metal-ml.git
   cd bare-metal-ml
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open any notebook under `python/`:
   - `python/GDA.ipynb` — Gaussian Discriminant Analysis
   - `python/GaussianNaiveBayes.ipynb` — Gaussian Naive Bayes
   - `python/LogisticRegression.ipynb` — Logistic Regression
   - `python/LinearRegression.ipynb` — Linear Regression

4. Run all cells (Kernel → Restart & Run All)

---

## Project Structure

```
bare-metal-ml/
├── README.md
├── .gitignore
├── python/
│   ├── GDA.ipynb
│   ├── GaussianNaiveBayes.ipynb
│   ├── LogisticRegression.ipynb
│   └── LinearRegression.ipynb
└── datasets/
    ├── wdbc.csv
    └── iris.csv
```

---

## Coming Soon

- C++ port with generic framework design
- pybind11 Python bindings
- REST API deployment via FastAPI and AWS
- Additional algorithms — SVM, Decision Trees, Random Forest, Neural Network

---

## Author

**Abhinav Arora**
University of Maryland — Computer Science, Freshman
