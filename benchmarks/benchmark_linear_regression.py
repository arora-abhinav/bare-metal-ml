"""
Benchmark: Linear Regression on Hours Studied vs Exam Score.
bare_metal_ml C++ (gradient descent, lr=0.01, 100 iterations) vs sklearn OLS.
Regression task — no confusion matrix. Reports MSE, MAE, R² and fitted-line plot.
Saves PNG to benchmarks/results/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from bare_metal_ml._cpp import LinearRegression as BML_LR

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def bench_linear_regression():
    print("\n" + "="*60)
    print("Linear Regression  |  Hours Studied vs Exam Score")
    print("bare_metal_ml: GD lr=0.01, 100 iter  |  sklearn: OLS")
    print("Note: regression task — confusion matrix not applicable.")
    print("="*60)

    path = os.path.join(ROOT, "notebooks", "linear_regression", "score.csv")
    df = pd.read_csv(path)
    x  = df['Hours'].tolist()
    y  = df['Scores'].tolist()

    # bare_metal_ml — gradient descent with same params as notebook
    t0 = time.time()
    bml = BML_LR()
    bml.fit(x, y, learning_rate=0.01, iterations=100)
    y_bml = bml.predict(x)
    t_bml = time.time() - t0

    mse_bml = mean_squared_error(y, y_bml)
    mae_bml = mean_absolute_error(y, y_bml)
    r2_bml  = r2_score(y, y_bml)
    print(f"\n  bare_metal_ml GD  |  MSE={mse_bml:.3f}  MAE={mae_bml:.3f}  R²={r2_bml:.4f}  t={t_bml:.4f}s")

    # sklearn OLS — closed-form normal equations
    t0 = time.time()
    sk = SklearnLR()
    X  = np.array(x).reshape(-1, 1)
    sk.fit(X, y)
    y_sk = sk.predict(X).tolist()
    t_sk = time.time() - t0

    mse_sk = mean_squared_error(y, y_sk)
    mae_sk = mean_absolute_error(y, y_sk)
    r2_sk  = r2_score(y, y_sk)
    print(f"  sklearn OLS       |  MSE={mse_sk:.3f}  MAE={mae_sk:.3f}  R²={r2_sk:.4f}  t={t_sk:.4f}s")

    # ── Plot ─────────────────────────────────────────────────────────────────
    x_line = np.linspace(min(x), max(x), 300).tolist()
    y_bml_line = bml.predict(x_line)
    y_sk_line  = sk.predict(np.array(x_line).reshape(-1, 1)).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Regression — Hours Studied vs Exam Score", fontsize=12)

    for ax, y_pred, y_line, label, color, mse, mae, r2 in [
        (axes[0], y_bml, y_bml_line,
         "bare_metal_ml (GD lr=0.01, 100 iter)", "crimson", mse_bml, mae_bml, r2_bml),
        (axes[1], y_sk,  y_sk_line,
         "sklearn (OLS)", "steelblue", mse_sk, mae_sk, r2_sk),
    ]:
        ax.scatter(x, y, color='grey', alpha=0.7, zorder=3, label='Data')
        ax.plot(x_line, y_line, color=color, linewidth=2, label=label)
        ax.set_xlabel("Hours Studied"); ax.set_ylabel("Exam Score")
        ax.set_title(f"{label}\nMSE={mse:.2f}  MAE={mae:.2f}  R²={r2:.4f}", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "linear_regression_benchmark.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  → {out}")


if __name__ == "__main__":
    bench_linear_regression()
    print("\nDone. Figure saved to benchmarks/results/")
