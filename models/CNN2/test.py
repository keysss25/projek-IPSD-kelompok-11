import torch
import numpy as np

from pathlib import Path
from Model.CNN2 import DeepCNN
from utils.preprocessing import prepare_datasets

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():

    # Pengaturan dasar
    PROJECT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = PROJECT_DIR / "models"

    # Mengambil hasil preprocessing
    prepared_data = prepare_datasets()

    class_names = prepared_data["class_names"]
    test_loader = prepared_data["test_ds"]

    # Mencari checkpoint DeepCNN setiap fold
    checkpoints = sorted(
        MODEL_DIR.glob("DeepCNN_fold_*.pt"),
        key=lambda x: int(x.stem.split("_")[-1])
    )

    if not checkpoints:
        print("Checkpoint DeepCNN tidak ditemukan!")
        return

    all_results = []

    # Evaluasi setiap fold
    for checkpoint_path in checkpoints:

        fold = checkpoint_path.stem.split("_")[-1]

        print(f"\n{'=' * 50}")
        print(f"TESTING DEEPCNN FOLD {fold}")
        print(f"{'=' * 50}")

        # Memuat checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        # Membuat arsitektur model
        model = DeepCNN(
            input_dim=224,
            input_c=3,
            output=len(class_names),
            hidden_dim=128,
            dropout=0.5,
            device="cpu"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        y_true = []
        y_pred = []

        # Proses testing
        with torch.no_grad():

            for src, trg in test_loader:

                pred = model(src)
                predicted = torch.argmax(pred, dim=1)

                y_true.extend(trg.numpy())
                y_pred.extend(predicted.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Menghitung metrik
        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Menampilkan hasil
        print("Total data :", len(y_true))
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-Score   : {f1:.4f}")

        # Menyimpan hasil fold
        np.savez(
            MODEL_DIR / f"testing_results_fold_{fold}.npz",
            y_true=y_true,
            y_pred=y_pred,
            class_names=np.array(class_names)
        )

        all_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    # Ringkasan seluruh fold
    print("\n")
    print("=" * 70)
    print("RINGKASAN EVALUASI DEEPCNN")
    print("=" * 70)

    print(
        f"{'Fold':<10}"
        f"{'Accuracy':<15}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1-Score':<15}"
    )

    for result in all_results:

        print(
            f"{result['fold']:<10}"
            f"{result['accuracy']:<15.4f}"
            f"{result['precision']:<15.4f}"
            f"{result['recall']:<15.4f}"
            f"{result['f1_score']:<15.4f}"
        )

    # Menghitung rata-rata
    print("-" * 70)

    print(
        f"{'Mean':<10}"
        f"{np.mean([r['accuracy'] for r in all_results]):<15.4f}"
        f"{np.mean([r['precision'] for r in all_results]):<15.4f}"
        f"{np.mean([r['recall'] for r in all_results]):<15.4f}"
        f"{np.mean([r['f1_score'] for r in all_results]):<15.4f}"
    )

    # Menghitung standar deviasi
    print(
        f"{'Std':<10}"
        f"{np.std([r['accuracy'] for r in all_results]):<15.4f}"
        f"{np.std([r['precision'] for r in all_results]):<15.4f}"
        f"{np.std([r['recall'] for r in all_results]):<15.4f}"
        f"{np.std([r['f1_score'] for r in all_results]):<15.4f}"
    )


if __name__ == "__main__":
    main()

