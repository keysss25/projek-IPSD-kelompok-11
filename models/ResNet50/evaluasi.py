import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from Utils.preprocessing import prepare_datasets
from Model.ResNet50 import get_resnet50


def main():

    PROJECT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = PROJECT_DIR / "models"

    # Ambil data test dan nama kelas
    prepared_data = prepare_datasets()
    class_names = prepared_data["class_names"]
    test_loader = prepared_data["test_ds"]

    # Cari checkpoint terbaik
    best_checkpoint = None
    best_loss = float("inf")
    best_fold = None

    for fold in range(1, 6):

        checkpoint_path = MODEL_DIR / f"SimpleCNN_fold_{fold}.pt"

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        if checkpoint["loss"] < best_loss:
            best_loss = checkpoint["loss"]
            best_checkpoint = checkpoint
            best_fold = fold

    print("Best Fold:", best_fold)
    print("Best Validation Loss:", best_loss)

    # Membuat arsitektur ResNet50 tanpa download pretrained ulang
    model = get_resnet50(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    # Prediksi data final test
    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Menghitung metrik
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    # Menampilkan hasil
    print("\n===== HASIL EVALUASI RESNET50 =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    # Simpan hasil, dengan nama terpisah dari model lain
    np.savez(
        MODEL_DIR / "resnet50_testing_results.npz",
        y_true=y_true,
        y_pred=y_pred,
        class_names=np.array(class_names),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm,
        best_fold=best_fold,
        best_loss=best_loss
    )

    print("\nHasil ResNet50 berhasil disimpan!")


if __name__ == "__main__":
    main()
