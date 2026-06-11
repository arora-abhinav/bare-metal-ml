"""
Benchmark: Bernoulli and Multinomial Naive Bayes on SMS Spam dataset.
bare_metal_ml C++ (vocab_size=1000, internal tokenisation) vs
sklearn CountVectorizer(max_features=1000) + Naive Bayes.
Same 80/20 split (seed=42).
Saves one PNG per algorithm to benchmarks/results/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB as SklearnBNB, MultinomialNB as SklearnMNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from bare_metal_ml._cpp import (
    BernoulliNaiveBayes as BML_BNB,
    MultinomialNaiveBayes as BML_MNB,
)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def _plot_cm(ax, y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    ax.set_title(title, fontsize=9)


def _report(name, y_true, y_pred, elapsed):
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  {name}  |  accuracy={acc:.2f}%  |  time={elapsed:.4f}s")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))


def _load_spam():
    path = os.path.join(ROOT, "notebooks", "naive_bayes", "spam.csv")
    df = pd.read_csv(path, encoding='latin-1', usecols=['v1', 'v2'])
    texts  = df['v2'].tolist()
    labels = df['v1'].tolist()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(texts))
    n = int(0.8 * len(texts))
    tr, te = idx[:n], idx[n:]
    return (
        [texts[i]  for i in tr], [labels[i] for i in tr],
        [texts[i]  for i in te], [labels[i] for i in te],
    )


# ── Bernoulli Naive Bayes ─────────────────────────────────────────────────────

def bench_bernoulli_nb():
    print("\n" + "="*60)
    print("Bernoulli Naive Bayes vs sklearn BernoulliNB  |  SMS Spam")
    print("vocab_size=1000, 80/20 split")
    print("="*60)
    texts_tr, y_tr, texts_te, y_te = _load_spam()
    labels = ['ham', 'spam']

    # bare_metal_ml — raw text strings, internal tokenisation
    t0 = time.time()
    bml = BML_BNB(vocab_size=1000)
    bml.fit(texts_tr, y_tr)
    p_bml = bml.predict(texts_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml BernoulliNB", y_te, p_bml, t_bml)

    # sklearn — CountVectorizer(binary=True, max_features=1000) + BernoulliNB
    t0 = time.time()
    vec = CountVectorizer(binary=True, max_features=1000)
    X_tr = vec.fit_transform(texts_tr)
    X_te = vec.transform(texts_te)
    sk   = SklearnBNB()
    sk.fit(X_tr, y_tr)
    p_sk = sk.predict(X_te).tolist()
    t_sk = time.time() - t0
    _report("sklearn BernoulliNB", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Bernoulli Naive Bayes — SMS Spam  (vocab=1000)", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml BernoulliNB\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn BernoulliNB\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "bernoulli_nb_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ── Multinomial Naive Bayes ───────────────────────────────────────────────────

def bench_multinomial_nb():
    print("\n" + "="*60)
    print("Multinomial Naive Bayes vs sklearn MultinomialNB  |  SMS Spam")
    print("vocab_size=1000, 80/20 split")
    print("="*60)
    texts_tr, y_tr, texts_te, y_te = _load_spam()
    labels = ['ham', 'spam']

    # bare_metal_ml — raw text strings, internal tokenisation
    t0 = time.time()
    bml = BML_MNB(vocab_size=1000)
    bml.fit(texts_tr, y_tr)
    p_bml = bml.predict(texts_te)
    t_bml = time.time() - t0
    _report("bare_metal_ml MultinomialNB", y_te, p_bml, t_bml)

    # sklearn — CountVectorizer(max_features=1000) + MultinomialNB
    t0 = time.time()
    vec = CountVectorizer(max_features=1000)
    X_tr = vec.fit_transform(texts_tr)
    X_te = vec.transform(texts_te)
    sk   = SklearnMNB()
    sk.fit(X_tr, y_tr)
    p_sk = sk.predict(X_te).tolist()
    t_sk = time.time() - t0
    _report("sklearn MultinomialNB", y_te, p_sk, t_sk)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Multinomial Naive Bayes — SMS Spam  (vocab=1000)", fontsize=11)
    _plot_cm(axes[0], y_te, p_bml, labels,
             f"bare_metal_ml MultinomialNB\nacc={accuracy_score(y_te,p_bml)*100:.1f}%  t={t_bml:.4f}s")
    _plot_cm(axes[1], y_te, p_sk, labels,
             f"sklearn MultinomialNB\nacc={accuracy_score(y_te,p_sk)*100:.1f}%  t={t_sk:.4f}s")
    plt.tight_layout()
    out = os.path.join(RESULTS, "multinomial_nb_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


if __name__ == "__main__":
    bench_bernoulli_nb()
    bench_multinomial_nb()
    print("\n\nAll done. Figures saved to benchmarks/results/")
