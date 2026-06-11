"""
Benchmark: MNIST handwritten digit classification.
Architecture for all models: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
Optimizer: Adam lr=0.01  |  Dropout: 0.2  |  Batch size: 64

Models:
  1. bare_metal_ml C++ (BML_EPOCHS epochs — limited by autograd overhead)
  2. PyTorch        (PT_EPOCHS  epochs)
  3. Keras/PyTorch  (PT_EPOCHS  epochs, KERAS_BACKEND=torch)
  4. TensorFlow     (PT_EPOCHS  epochs, attempted — requires Python ≤ 3.12)

Data: 60 000 training images split 80/20 → 48 000 train / 12 000 test (same for all).
Saves one PNG with a 2×2 confusion-matrix grid to benchmarks/results/.
"""

import os
import struct
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ["KERAS_BACKEND"] = "torch"

from bare_metal_ml._cpp import Network, Adam, FunctionType

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import keras
    KERAS_OK = True
except Exception:
    KERAS_OK = False
    print("WARNING: Keras import failed — Keras benchmark skipped.")

try:
    import tensorflow as tf
    TF_OK = True
except Exception:
    TF_OK = False
    print("NOTE: TensorFlow not available on Python 3.14 — TF benchmark skipped.")
    print("      Install on Python ≤ 3.12 and re-run to include TF results.\n")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

MNIST_DIR  = os.path.join(ROOT, "notebooks", "neural_network", "MNIST handwritten ")
BATCH_SIZE = 64
BML_EPOCHS = 10  # same as PyTorch and Keras
PT_EPOCHS  = 10  # PyTorch and Keras epochs


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_idx(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic == 2051:
            n, rows, cols = [int.from_bytes(f.read(4), 'big') for _ in range(3)]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        elif magic == 2049:
            n = int.from_bytes(f.read(4), 'big')
            return np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown IDX magic number: {magic}")


def load_mnist():
    # Try the standalone .idx3-ubyte files first, then the subdirectory layout
    def _img_path(name):
        standalone = os.path.join(MNIST_DIR, f"{name}.idx3-ubyte")
        nested     = os.path.join(MNIST_DIR, name, name)
        return standalone if os.path.isfile(standalone) else nested

    def _lbl_path(name):
        standalone = os.path.join(MNIST_DIR, f"{name}.idx1-ubyte")
        nested     = os.path.join(MNIST_DIR, name, name)
        return standalone if os.path.isfile(standalone) else nested

    images = _read_idx(_img_path("train-images"))
    labels = _read_idx(_lbl_path("train-labels"))

    images = images.astype(np.float64) / 255.0   # normalise to [0, 1]

    # Reproducible 80/20 split (same indices for every model)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(images))
    n_train = int(0.8 * len(images))
    tr, te  = idx[:n_train], idx[n_train:]
    return images[tr], labels[tr], images[te], labels[te]


def _one_hot(labels, n=10):
    """Return (n_classes × n_examples) column-major one-hot matrix."""
    result = [[0.0] * len(labels) for _ in range(n)]
    for i, lbl in enumerate(labels):
        result[lbl][i] = 1.0
    return result


def _to_col_major(x_np):
    """Convert row-major (n, 784) ndarray to column-major list-of-lists."""
    return x_np.T.tolist()


# ── Confusion matrix plot helper ──────────────────────────────────────────────

def _plot_cm(ax, y_true, y_pred, title):
    labels = list(range(10))
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(10)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=5,
                        color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    ax.set_title(title, fontsize=9)


def _report(name, y_true, y_pred, elapsed):
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  {name}  |  accuracy={acc:.2f}%  |  time={elapsed:.1f}s")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))


# ── bare_metal_ml ─────────────────────────────────────────────────────────────

def bench_bml(x_tr_np, y_tr, x_te_np, y_te):
    print(f"\n{'='*60}")
    print(f"bare_metal_ml C++ Network  |  {BML_EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"Architecture: 784→128(ReLU)→64(ReLU)→10  Adam lr=0.01  dropout=0.2")
    print('='*60)

    x_tr_col = _to_col_major(x_tr_np)
    y_tr_oh  = _one_hot(y_tr)
    x_te_col = _to_col_major(x_te_np)

    adam = Adam(0.01)
    net  = Network(3, [128, 64, 10], x_tr_col, adam, 0.2,
                   function_type=FunctionType.RELU)

    t0 = time.time()
    net.train_loop(BML_EPOCHS, y_tr_oh, BATCH_SIZE)
    t_train = time.time() - t0
    print(f"  Training finished in {t_train:.1f}s")

    t0 = time.time()
    preds = net.predict(x_te_col)
    t_inf = time.time() - t0
    elapsed = t_train + t_inf

    _report("bare_metal_ml", y_te.tolist(), preds, elapsed)
    return preds, elapsed


# ── PyTorch ───────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 10),
        )
        # He (Kaiming) initialisation — same as bare_metal_ml
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def bench_pytorch(x_tr_np, y_tr, x_te_np, y_te):
    print(f"\n{'='*60}")
    print(f"PyTorch MLP  |  {PT_EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"Architecture: 784→128(ReLU)→64(ReLU)→10  Adam lr=0.01  dropout=0.2")
    print('='*60)

    X_tr = torch.tensor(x_tr_np, dtype=torch.float32)
    Y_tr = torch.tensor(y_tr,    dtype=torch.long)
    X_te = torch.tensor(x_te_np, dtype=torch.float32)

    loader  = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    model   = _MLP(dropout=0.2)
    opt     = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for epoch in range(PT_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"    epoch {epoch+1}/{PT_EPOCHS}  loss={epoch_loss/len(loader):.4f}")
    t_train = time.time() - t0
    print(f"  Training finished in {t_train:.1f}s")

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(X_te)
    preds = logits.argmax(dim=1).numpy().tolist()
    t_inf = time.time() - t0
    elapsed = t_train + t_inf

    _report("PyTorch", y_te.tolist(), preds, elapsed)
    return preds, elapsed


# ── Keras (PyTorch backend) ───────────────────────────────────────────────────

def bench_keras(x_tr_np, y_tr, x_te_np, y_te):
    if not KERAS_OK:
        return None, None

    print(f"\n{'='*60}")
    print(f"Keras (backend=torch)  |  {PT_EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"Architecture: 784→128(ReLU)→64(ReLU)→10  Adam lr=0.01  dropout=0.2")
    print('='*60)

    n_classes = 10
    y_tr_oh = np.eye(n_classes)[y_tr].astype(np.float32)

    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation='relu',
                           kernel_initializer='he_uniform'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu',
                           kernel_initializer='he_uniform'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax',
                           kernel_initializer='he_uniform'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    t0 = time.time()
    model.fit(
        x_tr_np.astype(np.float32), y_tr_oh,
        epochs=PT_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    t_train = time.time() - t0
    print(f"  Training finished in {t_train:.1f}s")

    t0 = time.time()
    probs = model.predict(x_te_np.astype(np.float32), verbose=0)
    preds = np.argmax(probs, axis=1).tolist()
    t_inf = time.time() - t0
    elapsed = t_train + t_inf

    _report("Keras (torch backend)", y_te.tolist(), preds, elapsed)
    return preds, elapsed


# ── TensorFlow ────────────────────────────────────────────────────────────────

def bench_tensorflow(x_tr_np, y_tr, x_te_np, y_te):
    if not TF_OK:
        return None, None

    print(f"\n{'='*60}")
    print(f"TensorFlow  |  {PT_EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"Architecture: 784→128(ReLU)→64(ReLU)→10  Adam lr=0.01  dropout=0.2")
    print('='*60)

    n_classes = 10
    y_tr_oh = np.eye(n_classes)[y_tr].astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax',
                              kernel_initializer='he_uniform'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    t0 = time.time()
    model.fit(
        x_tr_np.astype(np.float32), y_tr_oh,
        epochs=PT_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    t_train = time.time() - t0
    print(f"  Training finished in {t_train:.1f}s")

    t0 = time.time()
    probs = model.predict(x_te_np.astype(np.float32), verbose=0)
    preds = np.argmax(probs, axis=1).tolist()
    t_inf = time.time() - t0
    elapsed = t_train + t_inf

    _report("TensorFlow", y_te.tolist(), preds, elapsed)
    return preds, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MNIST...")
    x_tr_np, y_tr, x_te_np, y_te = load_mnist()
    print(f"  train={len(y_tr)}  test={len(y_te)}")

    results = {}

    p_bml, t_bml = bench_bml(x_tr_np, y_tr, x_te_np, y_te)
    results['bare_metal_ml\n(C++ autograd)'] = (p_bml, t_bml, BML_EPOCHS)

    p_pt, t_pt = bench_pytorch(x_tr_np, y_tr, x_te_np, y_te)
    results['PyTorch'] = (p_pt, t_pt, PT_EPOCHS)

    p_k, t_k = bench_keras(x_tr_np, y_tr, x_te_np, y_te)
    if p_k is not None:
        results['Keras\n(torch backend)'] = (p_k, t_k, PT_EPOCHS)

    p_tf, t_tf = bench_tensorflow(x_tr_np, y_tr, x_te_np, y_te)
    if p_tf is not None:
        results['TensorFlow'] = (p_tf, t_tf, PT_EPOCHS)

    # ── Plot confusion matrices ───────────────────────────────────────────────
    n_models = len(results)
    cols = min(n_models, 2)
    rows = (n_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    y_te_list = y_te.tolist()
    for ax_i, (name, (preds, elapsed, epochs)) in enumerate(results.items()):
        acc = accuracy_score(y_te_list, preds) * 100
        title = (f"{name}\nacc={acc:.1f}%  |  {epochs} epochs"
                 + (f"  |  t={elapsed:.0f}s" if elapsed else ""))
        _plot_cm(axes[ax_i], y_te_list, preds, title)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        "MNIST — 784→128(ReLU)→64(ReLU)→10  |  Adam lr=0.01  dropout=0.2  batch=64",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(RESULTS, "neural_network_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Model':<28} {'Accuracy':>10} {'Epochs':>7} {'Time':>8}")
    print("-"*55)
    for name, (preds, elapsed, epochs) in results.items():
        display_name = name.replace('\n', ' ')
        acc = accuracy_score(y_te_list, preds) * 100
        t_s = f"{elapsed:.0f}s" if elapsed else "N/A"
        print(f"  {display_name:<26} {acc:>9.2f}% {epochs:>7}  {t_s:>7}")
    print("="*55)
    print("\nDone. Figures saved to benchmarks/results/")
