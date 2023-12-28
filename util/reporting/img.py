from typing import Dict, Tuple
import matplotlib.pyplot as plt


def history_to_img(history: Dict[str, Tuple[float]], path: str = None) -> None:
    if "val_loss" in history.keys():
        min_loss = min(history["val_loss"])
        min_idx = history["val_loss"].index(min_loss)
        plt.vlines(min_idx, 0, min_loss, "black", "dashed")
        plt.hlines(min_loss, 0, min_idx, "black", "dashed")

        plt.plot(history["val_loss"], label="Validation")

        plt.yticks(list(plt.yticks()[0]) + [min_loss])

    plt.plot(history["train_loss"], label="Training")

    plt.title("History")
    plt.grid()
    plt.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()
