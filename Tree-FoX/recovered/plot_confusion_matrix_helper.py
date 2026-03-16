import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    out_path,
    title="Confusion Matrix",
    dpi=600,
    figsize=(7, 6)
):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()