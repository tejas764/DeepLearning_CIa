"""Paired ReLU/Tanh experiment on the scikit-learn WDBC benchmark."""
# %% setup
import argparse
import hashlib
import os
from pathlib import Path

# Configure a deterministic CPU run before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

SPLIT_SEEDS = (2026, 2027)
FIT_SEEDS = (11, 23, 37, 53, 71)
ACTIVATIONS = {"ReLU": "relu", "Tanh": "tanh"}
EXAMPLE_SEED = FIT_SEEDS[0]  # Chosen in advance for run-specific figures.
EPOCHS, PATIENCE, BATCH_SIZE = 50, 8, 16
THRESHOLD = 0.5
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LABELS = ["Benign", "Malignant"]
COLORS = {"ReLU": "#087C80", "Tanh": "#A54927"}
INK = "#193147"

# %% data
def prepare_data():
    """Return split arrays plus raw training measurements for descriptive EDA."""
    dataset = load_breast_cancer(as_frame=True)
    raw = dataset.data
    assert dataset.target_names.tolist() == ["malignant", "benign"]
    y = (dataset.target.to_numpy() == 0).astype("int32")
    ids = np.arange(len(y))
    pool, test = train_test_split(
        ids, test_size=0.20, random_state=SPLIT_SEEDS[0], stratify=y
    )
    train, valid = train_test_split(
        pool, test_size=0.20, random_state=SPLIT_SEEDS[1], stratify=y[pool]
    )
    splits = {"Train": train, "Validation": valid, "Test": test}
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), ids)
    audit = {
        "samples": len(raw), "features": raw.shape[1],
        "missing_cells_including_label": int(dataset.frame.isna().sum().sum()),
        "duplicate_feature_rows": int(raw.duplicated().sum()),
        "duplicate_labeled_rows": int(dataset.frame.duplicated().sum()),
    }
    scaler = StandardScaler().fit(raw.iloc[train])
    assert scaler.n_samples_seen_ == len(train)
    arrays = {k: (scaler.transform(raw.iloc[idx]).astype("float32"), y[idx])
              for k, idx in splits.items()}
    assert np.allclose(arrays["Train"][0].mean(axis=0), 0, atol=1e-6)
    return raw, y, splits, arrays, audit, scaler


def split_table(y, splits):
    groups = {**splits, "All": np.arange(len(y))}
    return pd.DataFrame([
        {"partition": name, "rows": len(idx), "benign": int((y[idx] == 0).sum()),
         "malignant": int(y[idx].sum()), "malignant_fraction": y[idx].mean()}
        for name, idx in groups.items()
    ])


# %% network
def make_model(label, seed):
    """Identical seeded Glorot kernels and zero biases within each paired run."""
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(name=f"MLP_{label}")
    model.add(tf.keras.Input(shape=(30,), name="measurements"))
    for i, (units, activation, name) in enumerate([
        (16, ACTIVATIONS[label], "hidden_1"), (8, ACTIVATIONS[label], "hidden_2"),
        (1, "sigmoid", "malignant_probability")
    ]):
        model.add(tf.keras.layers.Dense(
            units, activation=activation, name=name,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + i),
            bias_initializer="zeros",
        ))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                           beta_2=0.999, epsilon=1e-7),
        loss=tf.keras.losses.BinaryCrossentropy(), jit_compile=False,
    )
    assert model.count_params() == 641
    return model


def weight_digest(model):
    return hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()


# %% fit
def fit_one(label, seed, orders, train, valid):
    """Train on minibatches; score full partitions at the end of each epoch."""
    model = make_model(label, seed)
    initial_digest = weight_digest(model)
    x_train, y_train = train
    bce = tf.keras.losses.BinaryCrossentropy()

    def epoch_scores(x, y):
        probabilities = model(x, training=False)
        loss = float(bce(y[:, None], probabilities).numpy())
        accuracy = np.mean((probabilities.numpy().ravel() >= THRESHOLD) == y)
        return loss, float(accuracy)

    best_loss = float("inf")
    history = []
    for epoch, order in enumerate(orders, start=1):
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            model.train_on_batch(x_train[idx], y_train[idx, None])
        loss, accuracy = epoch_scores(*train)
        val_loss, val_accuracy = epoch_scores(*valid)
        history.append(dict(activation=label, seed=seed, epoch=epoch, loss=loss,
                            accuracy=accuracy, val_loss=val_loss, val_accuracy=val_accuracy))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = model.get_weights()
        if epoch - best_epoch >= PATIENCE:
            break
    model.set_weights(best_weights)
    assert np.isclose(epoch_scores(*valid)[0], best_loss)
    info = dict(activation=label, seed=seed, epochs_run=epoch, best_epoch=best_epoch,
                best_val_loss=best_loss, initial_weight_sha256=initial_digest)
    return model, history, info


def train_all(train, valid):
    """This stage accepts only training and validation arrays."""
    fits, histories, checkpoints = [], [], []
    for seed in FIT_SEEDS:
        rng = np.random.default_rng(seed)
        # Both activations consume the same ordered sequence, up to early stopping.
        orders = [rng.permutation(len(train[1])) for _ in range(EPOCHS)]
        pair_digests = []
        for label in ACTIVATIONS:
            model, history, info = fit_one(label, seed, orders, train, valid)
            pair_digests.append(info["initial_weight_sha256"])
            fits.append((label, seed, model))
            histories.extend(history)
            checkpoints.append(info)
            print(f"{label:4}  seed {seed:2}  epochs {info['epochs_run']:2}  "
                  f"restored {info['best_epoch']:2}  validation BCE {info['best_val_loss']:.5f}")
        assert pair_digests[0] == pair_digests[1]
    return fits, pd.DataFrame(histories), pd.DataFrame(checkpoints)


# %% evaluation
def evaluate(fits, test, test_ids):
    """Score each frozen fit on exactly the same untouched test set."""
    x_test, actual = test
    runs, predictions, reports = [], [], []
    for label, seed, model in fits:
        prob = model(x_test, training=False).numpy().ravel()
        pred = (prob >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
        runs.append(dict(activation=label, seed=seed, accuracy=accuracy_score(actual, pred),
                         precision=precision_score(actual, pred), recall=recall_score(actual, pred),
                         f1=f1_score(actual, pred), roc_auc=roc_auc_score(actual, prob),
                         tn=tn, fp=fp, fn=fn, tp=tp))
        predictions.append(pd.DataFrame(dict(activation=label, seed=seed, row_id=test_ids,
                                            target=actual, probability=prob.astype(float), predicted=pred)))
        report = classification_report(actual, pred, labels=[0, 1], target_names=LABELS,
                                       output_dict=True, zero_division=0)
        for row in [*LABELS, "macro avg", "weighted avg"]:
            reports.append(dict(activation=label, seed=seed, group=row, **report[row]))
    per_run = pd.DataFrame(runs)
    grouped = per_run.groupby("activation", sort=False)
    summary = pd.concat([grouped[m].agg(["mean", "std"]).add_prefix(m + "_")
                         for m in METRICS], axis=1).reset_index()
    return per_run, summary, pd.concat(predictions, ignore_index=True), pd.DataFrame(reports)


# %% plots
def figure_style():
    sns.set_theme(style="whitegrid", context="notebook",
                  rc={"font.family": "DejaVu Sans", "axes.labelcolor": INK,
                      "text.color": INK, "axes.edgecolor": "#A6B3B8",
                      "grid.color": "#DFE7E8", "grid.linewidth": 0.7,
                      "axes.spines.top": False, "axes.spines.right": False,
                      "legend.frameon": False, "savefig.facecolor": "white"})


def save(fig, directory, filename):
    fig.savefig(directory / filename, dpi=190, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_eda(raw, y, train, directory):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7), layout="constrained")
    for ax, feature, title in zip(axes, ["mean radius", "mean concave points"],
                                  ["Mean radius", "Mean concave points"]):
        values = raw[feature].to_numpy()[train]
        bins = np.histogram_bin_edges(values, bins=18)
        for i, color in [(0, "#517CA0"), (1, "#963D59")]:
            ax.hist(values[y[train] == i], bins=bins, histtype="step", linewidth=2,
                    color=color, label=LABELS[i])
        ax.set(xlabel=title, ylabel="Training samples")
    axes[0].legend(fontsize=9)
    save(fig, directory, "01_training_distributions.png")


def plot_architecture(directory):
    fig, ax = plt.subplots(figsize=(8.8, 1.65))
    ax.set(xlim=(0, 10), ylim=(0, 2))
    ax.axis("off")
    blocks = [(0.05, "Input", "30 features"), (2.6, "Hidden 1", "16 units"),
              (5.15, "Hidden 2", "8 units"), (7.7, "Output", "1 sigmoid")]
    for i, (x, title, subtitle) in enumerate(blocks):
        box = FancyBboxPatch((x, 0.45), 2.12, 1.1, boxstyle="round,pad=0.05,rounding_size=0.1",
                             edgecolor="#32747B", facecolor="#EFF7F6", linewidth=1.4)
        ax.add_patch(box)
        ax.text(x+1.06, 1.15, title, ha="center", weight="bold", fontsize=13)
        ax.text(x+1.06, 0.77, subtitle, ha="center", fontsize=11)
        if i < 3:
            ax.annotate("", xy=(x+2.5, 1), xytext=(x+2.18, 1),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    save(fig, directory, "02_ann_architecture.png")


def plot_curves(history, checkpoints, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.25), layout="constrained")
    for label, color in COLORS.items():
        h = history.query("seed == @EXAMPLE_SEED and activation == @label")
        best = checkpoints.query("seed == @EXAMPLE_SEED and activation == @label").iloc[0].best_epoch
        for ax, metric, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Binary cross-entropy"]):
            ax.plot(h.epoch, h[metric], color=color, lw=1.65)
            ax.plot(h.epoch, h["val_"+metric], color=color, linestyle="--", lw=1.65)
            point = h[h.epoch == best].iloc[0]
            ax.scatter(best, point["val_"+metric], s=42, color=color, edgecolors="white", zorder=4)
            ax.set(xlabel="Epoch", ylabel=title)
    axes[0].legend(handles=[Line2D([], [], color=c, label=l, lw=2) for l, c in COLORS.items()],
                   loc="lower right", fontsize=9)
    axes[1].legend(handles=[Line2D([], [], color=INK, ls=ls, label=l)
                            for l, ls in [("Train", "-"), ("Validation", "--")]],
                   loc="upper right", fontsize=9)
    save(fig, directory, "03_learning_curves.png")


def plot_confusions(predictions, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), layout="constrained")
    for ax, label in zip(axes, ACTIVATIONS):
        d = predictions.query("seed == @EXAMPLE_SEED and activation == @label")
        cm = confusion_matrix(d.target, d.predicted, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, square=True, cmap="Blues",
                    vmin=0, vmax=d.target.value_counts().max(), ax=ax, linewidths=1,
                    xticklabels=LABELS, yticklabels=LABELS, annot_kws={"fontsize": 14})
        ax.set(xlabel="Predicted class", ylabel="Actual class", title=label)
        ax.tick_params(axis="y", labelrotation=0)
    save(fig, directory, "04_confusion_matrices.png")


def plot_roc(predictions, directory):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.25), layout="constrained")
    for ax in axes:
        for label, color in COLORS.items():
            d = predictions.query("seed == @EXAMPLE_SEED and activation == @label")
            fpr, tpr, _ = roc_curve(d.target, d.probability, drop_intermediate=False)
            auc = roc_auc_score(d.target, d.probability)
            ax.step(fpr, tpr, where="post", c=color,
                    ls="-" if label == "ReLU" else "--", lw=1.8, label=f"{label}  {auc:.4f}")
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].plot([0, 1], [0, 1], ":", color="#777777", lw=1)
    axes[0].set(xlim=(-0.025, 1.02), ylim=(-0.025, 1.03), title="Full ROC")
    axes[1].set(xlim=(-0.005, .205), ylim=(.74, 1.01), title="Low false-positive region")
    axes[0].legend(loc="lower right", fontsize=9)
    save(fig, directory, "05_roc_curves.png")


def plot_comparison(summary, directory):
    fig, ax = plt.subplots(figsize=(8, 2.95), layout="constrained")
    for i, (label, color) in enumerate(COLORS.items()):
        row = summary.set_index("activation").loc[label]
        ax.errorbar(np.arange(5) + (i-.5)*.21, [row[m+"_mean"] for m in METRICS],
                    yerr=[row[m+"_std"] for m in METRICS], fmt="o" if i == 0 else "s",
                    color=color, capsize=4, ms=6, elinewidth=1.6, label=label)
    ax.set_xticks(range(5), ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set(ylabel="Test score", xlim=(-.5, 4.5))
    ax.legend(loc="best", fontsize=9)
    save(fig, directory, "06_metric_comparison.png")


# %% exports
def save_tables(directory, splits, y, history, checkpoints, runs, summary, predictions, reports):
    directory.mkdir(parents=True, exist_ok=True)
    membership = pd.concat([pd.DataFrame(dict(row_id=idx, partition=k, target=y[idx]))
                            for k, idx in splits.items()]).sort_values("row_id")
    tables = {"split_membership": membership, "training_history": history,
              "per_run_metrics": runs.merge(checkpoints, on=["activation", "seed"], validate="1:1"),
              "comparison_metrics": summary, "test_predictions": predictions,
              "classification_reports": reports}
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False, float_format="%.12g")


def main(output):
    raw, y, splits, data, audit, _ = prepare_data()
    print("Dataset audit", audit)
    print(split_table(y, splits).to_string(index=False))
    fits, history, checkpoints = train_all(data["Train"], data["Validation"])
    runs, summary, predictions, reports = evaluate(fits, data["Test"], splits["Test"])
    save_tables(output / "results", splits, y, history, checkpoints, runs, summary, predictions, reports)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_style()
    plot_eda(raw, y, splits["Train"], figures)
    plot_architecture(figures)
    plot_curves(history, checkpoints, figures)
    plot_confusions(predictions, figures)
    plot_roc(predictions, figures)
    plot_comparison(summary, figures)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().output)

