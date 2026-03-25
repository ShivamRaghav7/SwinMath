import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path="metrics.csv"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    epochs     = df["epoch"]
    train_loss = df["train_loss"]
    val_loss   = df["val_loss"]
    val_acc    = df["val_accuracy"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SwinMath Training Metrics", fontsize=13, fontweight="bold")

    ax1.plot(epochs, train_loss, label="Train", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val",   linewidth=2)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, val_acc, color="green", linewidth=2)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_metrics()