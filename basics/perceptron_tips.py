"""
Single Perceptron Neural Network trained on the Tips dataset.
Uses only Python standard library — no NumPy, TensorFlow, or PyTorch.

Task: Binary classification — predict whether a tip is above the median.

Training log (every epoch × step) is written to training_log.csv.
Loss curve is plotted with Seaborn.
"""

import csv
import io
import math
import random
import urllib.request

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIPS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"


def load_tips(url: str) -> list[dict]:
    print(f"Downloading Tips dataset from {url} ...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(content)))
    print(f"  Loaded {len(rows)} rows.\n")
    return rows


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

SEX_MAP    = {"Male": 0.0, "Female": 1.0}
SMOKER_MAP = {"No": 0.0,  "Yes": 1.0}
DAY_MAP    = {"Thur": 0.0, "Fri": 1.0, "Sat": 2.0, "Sun": 3.0}
TIME_MAP   = {"Lunch": 0.0, "Dinner": 1.0}


def build_features(row: dict) -> list[float]:
    return [
        float(row["total_bill"]),
        float(row["size"]),
        SEX_MAP[row["sex"]],
        SMOKER_MAP[row["smoker"]],
        DAY_MAP[row["day"]],
        TIME_MAP[row["time"]],
    ]


def preprocess(rows: list[dict]) -> tuple[list[list[float]], list[int], float]:
    tips = sorted(float(r["tip"]) for r in rows)
    median_tip = tips[len(tips) // 2]

    X = [build_features(r) for r in rows]
    y = [1 if float(r["tip"]) > median_tip else 0 for r in rows]
    return X, y, median_tip


def normalize(
    X: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Standardize each feature to zero mean and unit variance."""
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds  = [
        math.sqrt(sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) or 1.0
        for j in range(d)
    ]
    X_norm = [[(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]
    return X_norm, means, stds


def train_test_split(
    X: list, y: list, test_ratio: float = 0.2, seed: int = 42
) -> tuple:
    indices = list(range(len(X)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    tr, te = indices[:split], indices[split:]
    return [X[i] for i in tr], [y[i] for i in tr], [X[i] for i in te], [y[i] for i in te]


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def bce_loss(y_true: int, y_pred: float) -> float:
    """Binary cross-entropy loss for a single sample."""
    eps = 1e-15
    return -(y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps))


# CSV column order (written once as header, then one row per training step)
_LOG_FIELDS = [
    "epoch",        # current epoch (1-indexed)
    "step",         # sample index within the epoch (1-indexed)
    "global_step",  # absolute step counter across all epochs
    "y_true",       # ground-truth label (0 or 1)
    "y_pred",       # sigmoid output *before* the weight update
    "step_loss",    # BCE loss for this sample *before* the weight update
    "epoch_avg_loss",  # running average loss so far within this epoch
]


class Perceptron:
    """
    A single neuron: z = w·x + b,  output = sigmoid(z).
    Trained with stochastic gradient descent on binary cross-entropy loss.
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1, seed: int = 0):
        rng = random.Random(seed)
        self.w  = [rng.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.b  = 0.0
        self.lr = learning_rate

    # -- forward pass -------------------------------------------------------

    def _net(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def forward(self, x: list[float]) -> float:
        return sigmoid(self._net(x))

    def predict(self, x: list[float]) -> int:
        return 1 if self.forward(x) >= 0.5 else 0

    # -- training -----------------------------------------------------------

    def fit(
        self,
        X_train: list[list[float]],
        y_train: list[int],
        epochs: int = 200,
        log_path: str = "training_log.csv",
        verbose: bool = True,
    ) -> list[float]:
        """
        SGD with binary cross-entropy loss.

        Every (epoch, step) is appended to `log_path` as it is computed —
        before the weight update — so the CSV captures the raw prediction
        and loss the network produced at that moment in training.

        Returns a list of per-epoch average losses.
        """
        epoch_losses: list[float] = []
        n = len(X_train)
        global_step = 0

        with open(log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
            writer.writeheader()

            for epoch in range(1, epochs + 1):
                running_loss = 0.0

                for step, (x, y) in enumerate(zip(X_train, y_train), start=1):
                    global_step += 1

                    # Forward pass (before weight update)
                    y_hat = self.forward(x)
                    loss  = bce_loss(y, y_hat)
                    running_loss += loss

                    # Log this step
                    writer.writerow({
                        "epoch":          epoch,
                        "step":           step,
                        "global_step":    global_step,
                        "y_true":         y,
                        "y_pred":         round(y_hat, 8),
                        "step_loss":      round(loss, 8),
                        "epoch_avg_loss": round(running_loss / step, 8),
                    })

                    # Backward pass (gradient of BCE + sigmoid = ŷ - y)
                    err = y_hat - y
                    for j in range(len(self.w)):
                        self.w[j] -= self.lr * err * x[j]
                    self.b -= self.lr * err

                avg_loss = running_loss / n
                epoch_losses.append(avg_loss)

                if verbose and (epoch % 20 == 0 or epoch == 1):
                    acc = self._accuracy(X_train, y_train)
                    print(f"  Epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.1f}%")

        print(f"\nTraining log saved → {log_path}  "
              f"({epochs * n:,} rows, {epochs} epochs × {n} steps)")
        return epoch_losses

    def _accuracy(self, X: list[list[float]], y: list[int]) -> float:
        correct = sum(self.predict(xi) == yi for xi, yi in zip(X, y))
        return 100.0 * correct / len(y)

    # -- reporting ----------------------------------------------------------

    def evaluate(self, X: list[list[float]], y: list[int]) -> dict:
        preds = [self.predict(xi) for xi in X]
        tp = sum(p == 1 and t == 1 for p, t in zip(preds, y))
        tn = sum(p == 0 and t == 0 for p, t in zip(preds, y))
        fp = sum(p == 1 and t == 0 for p, t in zip(preds, y))
        fn = sum(p == 0 and t == 1 for p, t in zip(preds, y))
        acc       = (tp + tn) / len(y) * 100
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)
        return dict(accuracy=acc, precision=precision, recall=recall, f1=f1,
                    tp=tp, tn=tn, fp=fp, fn=fn)

    def print_weights(self, feature_names: list[str]) -> None:
        print("\nLearned parameters:")
        print(f"  {'bias':<14}: {self.b:+.4f}")
        for name, w in zip(feature_names, self.w):
            print(f"  {name:<14}: {w:+.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> tuple[list[dict], list[dict]]:
    """
    Read the training CSV and return:
      - step_rows : one dict per training step
      - epoch_rows: one dict per epoch (last step's epoch_avg_loss = final avg)
    """
    step_rows: list[dict] = []
    epoch_rows: list[dict] = []
    current_epoch = None

    with open(log_path, newline="") as fh:
        for row in csv.DictReader(fh):
            epoch      = int(row["epoch"])
            step       = int(row["step"])
            global_step = int(row["global_step"])
            step_loss  = float(row["step_loss"])
            avg_loss   = float(row["epoch_avg_loss"])

            step_rows.append({
                "epoch":       epoch,
                "step":        step,
                "global_step": global_step,
                "step_loss":   step_loss,
                "epoch_avg_loss": avg_loss,
            })

            # Keep track of final row per epoch (= epoch average)
            if epoch != current_epoch:
                current_epoch = epoch
                epoch_rows.append(None)          # placeholder
            epoch_rows[-1] = {                   # overwrite until last step
                "epoch":      epoch,
                "epoch_loss": avg_loss,
            }

    return step_rows, epoch_rows


def plot_loss(log_path: str, out_path: str = "loss_curve.png") -> None:
    """
    Two-panel Seaborn figure:
      Top   — per-step BCE loss (every individual sample, every epoch)
      Bottom — per-epoch average BCE loss (smooth trend)
    """
    step_rows, epoch_rows = load_log(log_path)

    # ---- unpack for seaborn ------------------------------------------------
    global_steps = [r["global_step"] for r in step_rows]
    step_losses  = [r["step_loss"]   for r in step_rows]

    epochs      = [r["epoch"]      for r in epoch_rows]
    epoch_losses = [r["epoch_loss"] for r in epoch_rows]

    # ---- figure setup ------------------------------------------------------
    sns.set_theme(style="darkgrid", palette="muted")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle("Perceptron Training — Tips Dataset", fontsize=14, fontweight="bold")

    # ---- top: per-step loss ------------------------------------------------
    sns.lineplot(
        x=global_steps,
        y=step_losses,
        ax=ax_top,
        color="steelblue",
        linewidth=0.4,
        alpha=0.6,
        label="Per-step BCE loss",
    )
    # Overlay a smoothed trend using epoch-boundary ticks
    epoch_boundary_steps = [
        step_rows[(e - 1) * (max(r["step"] for r in step_rows if r["epoch"] == e) - 1)
                  + (max(r["step"] for r in step_rows if r["epoch"] == e) - 1)]["global_step"]
        for e in epochs
    ]
    sns.lineplot(
        x=epoch_boundary_steps,
        y=epoch_losses,
        ax=ax_top,
        color="crimson",
        linewidth=2.0,
        label="Epoch avg loss",
    )
    ax_top.set_xlabel("Global step (epoch × sample)")
    ax_top.set_ylabel("BCE Loss")
    ax_top.set_title("Loss at every training step")
    ax_top.legend()

    # ---- bottom: per-epoch average loss ------------------------------------
    sns.lineplot(
        x=epochs,
        y=epoch_losses,
        ax=ax_bot,
        color="crimson",
        linewidth=2.5,
        marker="o",
        markersize=3,
    )
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Avg BCE Loss")
    ax_bot.set_title("Average loss per epoch")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["total_bill", "size", "sex", "smoker", "day", "time"]
LOG_PATH      = "training_log.csv"
PLOT_PATH     = "loss_curve.png"


def main() -> None:
    random.seed(42)

    # 1. Load & prepare data
    rows = load_tips(TIPS_URL)
    X, y, median_tip = preprocess(rows)
    X, means, stds   = normalize(X)

    print(f"Classification task: tip > ${median_tip:.2f} → label 1  (else label 0)")
    pos = sum(y)
    print(f"Class balance: {pos} positive ({100*pos/len(y):.1f}%),"
          f" {len(y)-pos} negative\n")

    # 2. Split
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.20, seed=42)
    print(f"Train: {len(X_tr)} samples  |  Test: {len(X_te)} samples\n")

    # 3. Train (streams every step to CSV)
    print("Training perceptron (SGD, lr=0.1, 200 epochs):")
    perceptron = Perceptron(n_inputs=len(FEATURE_NAMES), learning_rate=0.1, seed=42)
    perceptron.fit(X_tr, y_tr, epochs=200, log_path=LOG_PATH, verbose=True)

    # 4. Evaluate
    train_metrics = perceptron.evaluate(X_tr, y_tr)
    test_metrics  = perceptron.evaluate(X_te, y_te)

    print("\n--- Evaluation ---")
    for split, m in [("Train", train_metrics), ("Test ", test_metrics)]:
        print(f"  {split}: accuracy={m['accuracy']:.1f}%  "
              f"precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}")
    m = test_metrics
    print(f"\n  Confusion matrix (test):")
    print(f"                Predicted 0   Predicted 1")
    print(f"  Actual 0      {m['tn']:<12}  {m['fp']}")
    print(f"  Actual 1      {m['fn']:<12}  {m['tp']}")

    # 5. Show learned weights
    perceptron.print_weights(FEATURE_NAMES)

    # 6. A few sample predictions
    print("\nSample predictions on test set (first 10):")
    print(f"  {'Actual':<8} {'Predicted':<10} {'Confidence':>10}")
    print(f"  {'-'*30}")
    for xi, yi in list(zip(X_te, y_te))[:10]:
        conf  = perceptron.forward(xi)
        pred  = 1 if conf >= 0.5 else 0
        label = lambda v: "High" if v == 1 else "Low "
        mark  = "" if pred == yi else " ✗"
        print(f"  {label(yi):<8} {label(pred):<10} {conf:>10.3f}{mark}")

    # 7. Plot loss curve from CSV using Seaborn
    print()
    plot_loss(LOG_PATH, out_path=PLOT_PATH)


if __name__ == "__main__":
    main()
