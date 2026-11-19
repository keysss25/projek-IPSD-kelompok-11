
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from Model.CNN1 import SimpleCNN
from Utils.preprocessing import prepare_datasets


# EVALUATION FUNCTION

def evaluate_model(model, data_loader, class_names):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for src, trg in data_loader:

            pred = model(src)
            predicted = torch.argmax(pred, dim=1)

            y_true.extend(trg.numpy())
            y_pred.extend(predicted.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }


# ==================================
# MAIN
# ==================================

def main():

    PROJECT_DIR = Path(__file__).resolve().parent

    # Folder khusus SimpleCNN
    MODEL_DIR = PROJECT_DIR / "models" / "SimpleCNN"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Mengambil hasil preprocessing
    prepared_data = prepare_datasets()

    class_names = prepared_data["class_names"]
    fold_datasets = prepared_data["fold_datasets"]
    test_loader = prepared_data["test_ds"]

    all_fold_results = []

    best_loss = float("inf")
    best_fold = None

    # EVALUASI SETIAP FOLD

    for fold_data in fold_datasets:

        fold = fold_data["fold"]
        val_loader = fold_data["val_ds"]

        print(f"\n========== FOLD {fold} ==========")

        checkpoint_path = MODEL_DIR / f"SimpleCNN_fold_{fold}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        # Membuat arsitektur SimpleCNN
        model = SimpleCNN(
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

        # Evaluasi validation dataset
        results = evaluate_model(
            model,
            val_loader,
            class_names
        )

        print("Total data:", len(results["y_true"]))
        print(f"Accuracy  : {results['accuracy']:.4f}")
        print(f"Precision : {results['precision']:.4f}")
        print(f"Recall    : {results['recall']:.4f}")
        print(f"F1-Score  : {results['f1_score']:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            results["y_true"],
            results["y_pred"],
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        ))

        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])

        # Simpan hasil evaluasi setiap fold
        np.savez(
            MODEL_DIR / f"SimpleCNN_fold_{fold}_results.npz",
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            confusion_matrix=results["confusion_matrix"],
            class_names=np.array(class_names)
        )

        all_fold_results.append({
            "Fold": fold,
            "Accuracy": results["accuracy"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "F1-Score": results["f1_score"]
        })

        # Memilih fold terbaik berdasarkan validation loss
        if checkpoint["loss"] < best_loss:

            best_loss = checkpoint["loss"]
            best_fold = fold

    # RINGKASAN EVALUASI 5-FOLD

    df_results = pd.DataFrame(all_fold_results)

    print("\n========== RINGKASAN EVALUASI 5-FOLD ==========")
    print(df_results.to_string(index=False))

    print("\nRATA-RATA DAN STANDAR DEVIASI:")
    print(
        df_results[
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        ].agg(["mean", "std"])
    )

    df_results.to_csv(
        MODEL_DIR / "SimpleCNN_5Fold_Evaluation.csv",
        index=False
    )

    print("\nBest Fold:", best_fold)
    print("Best Validation Loss:", best_loss)

    # FINAL TEST

    print("\n========== FINAL TEST ==========")

    checkpoint = torch.load(
        MODEL_DIR / f"SimpleCNN_fold_{best_fold}.pt",
        map_location="cpu"
    )

    final_model = SimpleCNN(
        input_dim=224,
        input_c=3,
        output=len(class_names),
        hidden_dim=128,
        dropout=0.5,
        device="cpu"
    )

    final_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_results = evaluate_model(
        final_model,
        test_loader,
        class_names
    )

    print("Total data:", len(final_results["y_true"]))
    print(f"Accuracy  : {final_results['accuracy']:.4f}")
    print(f"Precision : {final_results['precision']:.4f}")
    print(f"Recall    : {final_results['recall']:.4f}")
    print(f"F1-Score  : {final_results['f1_score']:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        final_results["y_true"],
        final_results["y_pred"],
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(final_results["confusion_matrix"])

    np.savez(
        MODEL_DIR / "SimpleCNN_final_test_results.npz",
        y_true=final_results["y_true"],
        y_pred=final_results["y_pred"],
        confusion_matrix=final_results["confusion_matrix"],
        class_names=np.array(class_names)
    )

    print("\nSemua hasil evaluasi SimpleCNN berhasil disimpan!")


if __name__ == "__main__":
    main()

