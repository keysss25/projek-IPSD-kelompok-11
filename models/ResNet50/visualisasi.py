import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Lokasi folder proyek dan hasil
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

data = np.load(MODEL_DIR / "resnet50_testing_results.npz")

y_true = data["y_true"]
y_pred = data["y_pred"]
class_names = data["class_names"]
cm = data["confusion_matrix"]


# ==========================================
# 1. CONFUSION MATRIX RESNET50 (HEATMAP)
# ==========================================

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix - ResNet50")
plt.colorbar()

ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Angka di dalam kotak heatmap
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix_resnet50.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 2. COMPARISON ACCURACY 3 MODEL
# ==========================================

models = ["SimpleCNN", "DeepCNN", "ResNet50"]
accuracies = [58.42, 72.02, 92.58]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Accuracy Comparison of CNN Models")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

# Menampilkan nilai accuracy di atas batang
for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "comparison_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Semua visualisasi berhasil disimpan!")
print("Lokasi:", MODEL_DIR)
