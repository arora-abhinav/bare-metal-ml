"""
Benchmark: GDA, KNN, KDTree, Logistic Regression, Gaussian Naive Bayes
bare_metal_ml C++ backend vs scikit-learn equivalents.
Same hyperparameters and train/test split (seed=42) throughout.
Saves one PNG per algorithm to benchmarks/results/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from bare_metal_ml._cpp import (
    GDA,
    KNN,
    KDTree,
    LogisticRegression as BML_LR,
    GaussianNaiveBayes as BML_GNB,
)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def _plot_cm(ax, y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    ax.set_title(title, fontsize=9)


def _report(name, y_true, y_pred, elapsed):
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  {name}  |  accuracy={acc:.2f}%  |  time={elapsed:.4f}s")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))


def _load_wdbc():
    path = os.path.join(ROOT, "notebooks", "gda", "wdbc.csv")
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, 2:].values.tolist()
    y = df.iloc[:, 1].tolist()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    n = int(0.8 * len(x))
    tr, te = idx[:n], idx[n:]
    return [x[i] for i in tr], [y[i] for i in tr], [x[i] for i in te], [y[i] for i in te]


def _load_iris(subdir="knn"):
    path = os.path.join(ROOT, "notebooks", subdir, "Iris.csv")
    df = pd.read_csv(path)
    x = df.iloc[:, 1:5].values.tolist()
    y = df.iloc[:, 5].tolist()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    n = int(0.8 * len(x))
    tr, te = idx[:n], idx[n:]
    return [x[i] for i in tr], [y[i] for i in tr], [x[i] for i in te], [y[i] for i in te]


# ── GDA ──────────────────────────────────────────────────────────────────────

def bench_gda():
    print("\n" + "="*60)
    print("GDA vs LinearDiscriminantAnalysis  |  WDBC Breast Cancer")
    print("="*60)
    x_tr, y_tr, x_te, y_te = _load_wdbc()
    labels = ['M', 'B']

    t0 = time.time()
    bml = GDA(positive_class="M")
    bml.fit(x_tr, y_tr)
    p_bml = bml.predict(x_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml GDA", y_te, p_bml, t_bml)

    t0 = time.time()
    sk = LinearDiscriminantAnalysis()
    sk.fit(np.array(x_tr), y_tr)
    p_sk = sk.predict(np.array(x_te)).tolist()
    t_sk = time.time() - t0
    _report("sklearn LDA", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("GDA — WDBC Breast Cancer  (M=malignant, B=benign)", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml GDA\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn LDA\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "gda_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ── KNN ───────────────────────────────────────────────────────────────────────

def bench_knn():
    print("\n" + "="*60)
    print("KNN (k=5, euclidean) vs sklearn KNeighborsClassifier  |  Iris")
    print("="*60)
    x_tr, y_tr, x_te, y_te = _load_iris()
    labels = sorted(set(y_te))

    t0 = time.time()
    bml = KNN(k=5, metric="euclidean")
    bml.fit(x_tr, y_tr)
    p_bml = bml.predict(x_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml KNN k=5", y_te, p_bml, t_bml)

    t0 = time.time()
    sk = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    sk.fit(np.array(x_tr), y_tr)
    p_sk = sk.predict(np.array(x_te)).tolist()
    t_sk = time.time() - t0
    _report("sklearn KNN k=5", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("KNN k=5 — Iris", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml KNN k=5\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn KNN k=5\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "knn_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ── KDTree ────────────────────────────────────────────────────────────────────

def bench_kdtree():
    print("\n" + "="*60)
    print("KDTree (k=9) vs sklearn kd_tree algorithm  |  Iris")
    print("="*60)
    x_tr, y_tr, x_te, y_te = _load_iris()
    labels = sorted(set(y_te))

    t0 = time.time()
    bml = KDTree()
    bml.fit(x_tr, y_tr)
    p_bml = bml.predict(x_te, k=9)
    t_bml = time.time() - t0
    _report("bare_metal_ml KDTree k=9", y_te, p_bml, t_bml)

    t0 = time.time()
    sk = KNeighborsClassifier(n_neighbors=9, algorithm='kd_tree', metric='euclidean')
    sk.fit(np.array(x_tr), y_tr)
    p_sk = sk.predict(np.array(x_te)).tolist()
    t_sk = time.time() - t0
    _report("sklearn KDTree k=9", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("KDTree k=9 — Iris", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml KDTree k=9\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn KDTree k=9\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "kdtree_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ── Logistic Regression ───────────────────────────────────────────────────────

def bench_logistic_regression():
    print("\n" + "="*60)
    print("LogisticRegression (binary: setosa vs rest)  |  Iris")
    print("lr=0.001, iterations=1000 — first 100 rows train, rest test")
    print("="*60)
    path = os.path.join(ROOT, "notebooks", "logistic_regression", "Iris.csv")
    df = pd.read_csv(path)
    feats = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    x  = df[feats].values.tolist()
    # bare_metal_ml predict returns positive_class or "" for negatives
    y  = ['Iris-setosa' if s == 'Iris-setosa' else '' for s in df['Species'].tolist()]
    x_tr, y_tr = x[:100], y[:100]
    x_te, y_te = x[100:], y[100:]
    labels = ['Iris-setosa', '']

    t0 = time.time()
    bml = BML_LR(positive_class='Iris-setosa')
    bml.fit(x_tr, y_tr, learning_rate=0.001, iterations=1000)
    p_bml = bml.predict(x_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml LR", y_te, p_bml, t_bml)

    # sklearn binary LR with lbfgs (same iteration budget)
    t0 = time.time()
    sk = SklearnLR(max_iter=1000, solver='lbfgs')
    sk.fit(np.array(x_tr), y_tr)
    p_sk = sk.predict(np.array(x_te)).tolist()
    t_sk = time.time() - t0
    _report("sklearn LR", y_te, p_sk, t_sk)

    display_labels = ['Iris-setosa', 'other']
    y_te_d  = ['Iris-setosa' if v == 'Iris-setosa' else 'other' for v in y_te]
    p_bml_d = ['Iris-setosa' if v == 'Iris-setosa' else 'other' for v in p_bml]
    p_sk_d  = ['Iris-setosa' if v == 'Iris-setosa' else 'other' for v in p_sk]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Logistic Regression — Iris binary (setosa vs rest)", fontsize=11)
    _plot_cm(axes[0], y_te_d, p_bml_d, display_labels,
             f"bare_metal_ml LR\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te_d, p_sk_d, display_labels,
             f"sklearn LR\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "logistic_regression_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ── Gaussian Naive Bayes ──────────────────────────────────────────────────────

def bench_gaussian_nb():
    print("\n" + "="*60)
    print("Gaussian Naive Bayes vs sklearn GaussianNB  |  WDBC Breast Cancer")
    print("="*60)
    x_tr, y_tr, x_te, y_te = _load_wdbc()
    labels = ['M', 'B']

    t0 = time.time()
    bml = BML_GNB()
    bml.fit(x_tr, y_tr)
    p_bml = bml.predict(x_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml GNB", y_te, p_bml, t_bml)

    t0 = time.time()
    sk = SklearnGNB()
    sk.fit(np.array(x_tr), y_tr)
    p_sk = sk.predict(np.array(x_te)).tolist()
    t_sk = time.time() - t0
    _report("sklearn GNB", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Gaussian Naive Bayes — WDBC Breast Cancer", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml GNB\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn GNB\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "gaussian_nb_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


if __name__ == "__main__":
    bench_gda()
    bench_knn()
    bench_kdtree()
    bench_logistic_regression()
    bench_gaussian_nb()
    print("\n\nAll done. Figures saved to benchmarks/results/")
