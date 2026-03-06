"""
Single Perceptron — Regression: total_bill → tip

Visualises the MSE cost function as a surface and contour map over
(weight, bias) space, with the gradient-descent trajectory overlaid.

Libraries: numpy, matplotlib, seaborn, urllib (standard lib)
"""

import csv
import io
import urllib.request

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TIPS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"


def load_data(url: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    with urllib.request.urlopen(url) as resp:
        rows = list(csv.DictReader(io.StringIO(resp.read().decode())))
    X_raw = np.array([float(r["total_bill"]) for r in rows])
    y     = np.array([float(r["tip"])        for r in rows])

    # Standardise X so the cost bowl is symmetric and GD converges cleanly
    x_mean, x_std = X_raw.mean(), X_raw.std()
    X = (X_raw - x_mean) / x_std

    print(f"Loaded {len(X)} samples  |  "
          f"total_bill ∈ [{X_raw.min():.2f}, {X_raw.max():.2f}]  |  "
          f"tip ∈ [{y.min():.2f}, {y.max():.2f}]")
    print(f"Feature standardised: mean={x_mean:.2f}  std={x_std:.2f}")
    return X, y, x_mean, x_std


# ---------------------------------------------------------------------------
# Perceptron — linear unit, MSE loss
# ---------------------------------------------------------------------------

def mse(X: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    return float(np.mean((w * X + b - y) ** 2))


def train(
    X: np.ndarray,
    y: np.ndarray,
    w0: float = 2.5,
    b0: float = 4.5,
    lr: float = 0.05,
    epochs: int = 80,
) -> tuple[float, float, list[tuple]]:
    """
    Gradient descent on MSE.
    Returns final (w, b) and the full trajectory as a list of (w, b, loss).
    """
    w, b = w0, b0
    n = len(X)
    trajectory = [(w, b, mse(X, y, w, b))]

    for epoch in range(1, epochs + 1):
        err = w * X + b - y           # residuals
        dw  = (2 / n) * (err @ X)    # ∂MSE/∂w
        db  = (2 / n) * err.sum()    # ∂MSE/∂b
        w  -= lr * dw
        b  -= lr * db
        trajectory.append((w, b, mse(X, y, w, b)))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}  w={w:.4f}  b={b:.4f}  loss={trajectory[-1][2]:.4f}")

    return w, b, trajectory


# ---------------------------------------------------------------------------
# Cost surface (vectorised over the grid)
# ---------------------------------------------------------------------------

def compute_surface(
    X: np.ndarray,
    y: np.ndarray,
    w_range: tuple[float, float],
    b_range: tuple[float, float],
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns W, B, J grids of shape (resolution, resolution).
    Broadcasting: (R, R, 1) * (n,) → (R, R, n)
    """
    ws = np.linspace(*w_range, resolution)
    bs = np.linspace(*b_range, resolution)
    W, B = np.meshgrid(ws, bs)

    # Vectorised: shape (R, R, n)
    preds = W[:, :, None] * X[None, None, :] + B[:, :, None]
    J = np.mean((preds - y[None, None, :]) ** 2, axis=2)
    return W, B, J


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    X: np.ndarray,
    y: np.ndarray,
    trajectory: list[tuple],
    w_range: tuple[float, float],
    b_range: tuple[float, float],
    x_mean: float = 0.0,
    x_std: float = 1.0,
    out_path: str = "cost_surface.png",
) -> None:
    sns.set_theme(style="white", palette="muted", font_scale=1.1)

    print("\nComputing cost surface …")
    W, B, J = compute_surface(X, y, w_range, b_range)

    traj_w = np.array([t[0] for t in trajectory])
    traj_b = np.array([t[1] for t in trajectory])
    traj_J = np.array([t[2] for t in trajectory])

    # Colour the trajectory by progress (blue → red)
    n_traj  = len(trajectory)
    colours = cm.coolwarm(np.linspace(0, 1, n_traj))

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        "Single Perceptron — Cost Surface over (Weight, Bias)\n"
        "Predicting Tip from Total Bill  |  MSE Loss",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── 1. 3D surface ────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot_surface(W, B, J, cmap="viridis", alpha=0.65, linewidth=0, antialiased=True)
    ax3d.plot(traj_w, traj_b, traj_J, color="crimson", linewidth=1.8, zorder=10)
    ax3d.scatter(*trajectory[0],  color="royalblue", s=60, zorder=11, label="Start")
    ax3d.scatter(*trajectory[-1], color="crimson",   s=60, zorder=11, label="End (min)")
    ax3d.set_xlabel("Weight  w  (normalised)", labelpad=8)
    ax3d.set_ylabel("Bias  b",                labelpad=8)
    ax3d.set_zlabel("MSE",       labelpad=8)
    ax3d.set_title("Cost Surface (3-D)")
    ax3d.legend(fontsize=9)
    ax3d.view_init(elev=30, azim=-60)

    # ── 2. Contour map + GD path ─────────────────────────────────────────────
    ax2d = fig.add_subplot(2, 2, 2)
    levels = np.percentile(J, np.linspace(0, 98, 40))   # finer near the minimum
    cf = ax2d.contourf(W, B, J, levels=levels, cmap="viridis")
    plt.colorbar(cf, ax=ax2d, label="MSE")
    ax2d.contour(W, B, J, levels=levels, colors="white", linewidths=0.3, alpha=0.4)

    # Trajectory coloured by epoch
    for i in range(n_traj - 1):
        ax2d.plot(
            traj_w[i:i+2], traj_b[i:i+2],
            color=colours[i], linewidth=1.5, alpha=0.85,
        )
    ax2d.scatter(traj_w[0],  traj_b[0],  s=90, color="royalblue", zorder=5,
                 edgecolors="white", linewidths=0.8, label="Start")
    ax2d.scatter(traj_w[-1], traj_b[-1], s=90, color="crimson",   zorder=5,
                 edgecolors="white", linewidths=0.8, label="End (min)")
    ax2d.set_xlabel("Weight  w  (normalised)")
    ax2d.set_ylabel("Bias  b")
    ax2d.set_title("Cost Contour + Gradient-Descent Path\n(colour: blue=early → red=late)")
    ax2d.legend(fontsize=9)

    # ── 3. Training loss curve ───────────────────────────────────────────────
    ax_loss = fig.add_subplot(2, 2, 3)
    sns.lineplot(x=np.arange(n_traj), y=traj_J, ax=ax_loss, color="crimson", linewidth=2)
    ax_loss.scatter([0],         [traj_J[0]],  color="royalblue", s=70, zorder=5,
                    label=f"Start  MSE={traj_J[0]:.3f}")
    ax_loss.scatter([n_traj-1],  [traj_J[-1]], color="crimson",   s=70, zorder=5,
                    label=f"Final  MSE={traj_J[-1]:.3f}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Training Loss Curve")
    ax_loss.legend(fontsize=9)

    # ── 4. Data scatter + learned line (de-normalised back to raw $ scale) ──
    ax_fit = fig.add_subplot(2, 2, 4)
    w0_n, b0_n = trajectory[0][0],  trajectory[0][1]
    wf_n, bf_n = trajectory[-1][0], trajectory[-1][1]

    # Convert normalised-space parameters to raw-bill space:
    # tip = w_n * x_norm + b_n  where x_norm = (x_raw - μ) / σ
    #     = (w_n/σ) * x_raw + (b_n - w_n*μ/σ)
    w0_r = w0_n / x_std;  b0_r = b0_n - w0_n * x_mean / x_std
    wf_r = wf_n / x_std;  bf_r = bf_n - wf_n * x_mean / x_std

    X_raw = X * x_std + x_mean
    sns.scatterplot(x=X_raw, y=y, ax=ax_fit, color="steelblue", alpha=0.6,
                    edgecolor="white", linewidth=0.4, label="Data")
    x_line = np.array([X_raw.min(), X_raw.max()])
    ax_fit.plot(x_line, w0_r * x_line + b0_r, "--", color="royalblue",
                linewidth=1.8, label=f"Initial fit  (w={w0_r:.3f}, b={b0_r:.2f})")
    ax_fit.plot(x_line, wf_r * x_line + bf_r, "-",  color="crimson",
                linewidth=2.2, label=f"Learned fit  (w={wf_r:.3f}, b={bf_r:.2f})")
    ax_fit.set_xlabel("Total Bill ($)")
    ax_fit.set_ylabel("Tip ($)")
    ax_fit.set_title("Data + Perceptron Fit  (raw scale)")
    ax_fit.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data …")
    X, y, x_mean, x_std = load_data(TIPS_URL)

    print("\nTraining (batch GD on MSE, single perceptron: tip = w·x_norm + b):")
    w_final, b_final, trajectory = train(X, y, w0=2.5, b0=4.5, lr=0.05, epochs=80)

    # De-normalise for reporting in raw-$ units
    wf_r = w_final / x_std
    bf_r = b_final - w_final * x_mean / x_std
    print(f"\nNormalised-space params :  w = {w_final:.4f}   b = {b_final:.4f}")
    print(f"Raw-scale params        :  w = {wf_r:.4f}   b = {bf_r:.4f}")
    print(f"  → tip ≈ {wf_r:.3f} × total_bill + {bf_r:.3f}")
    print(f"Final MSE  : {trajectory[-1][2]:.4f}")
    print(f"Final RMSE : {trajectory[-1][2]**0.5:.4f}")

    # Surface spans the trajectory with comfortable margins
    all_w = [t[0] for t in trajectory]
    all_b = [t[1] for t in trajectory]
    pad_w = (max(all_w) - min(all_w)) * 0.5 + 0.5
    pad_b = (max(all_b) - min(all_b)) * 0.5 + 0.5
    w_range = (min(all_w) - pad_w, max(all_w) + pad_w)
    b_range = (min(all_b) - pad_b, max(all_b) + pad_b)

    plot(X, y, trajectory, w_range=w_range, b_range=b_range,
         x_mean=x_mean, x_std=x_std)


if __name__ == "__main__":
    main()
