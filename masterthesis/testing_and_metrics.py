import os
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import label_binarize


def test_and_evaluate(
    test_loader,
    models_info,  # list of (path_to_model, model_class)
    output_dir,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_classes=3,
    class_names=None  # optional: list of class labels
):
    # Map internal names to clean display names
    name_map = {
        "dirt_road": "Dirt",
        "asphalt_road": "Asphalt",
        "cobblestone_road": "Cobblestone"
    }

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Replace with clean display names if known
    class_names = [name_map.get(n, n) for n in class_names]

    def evaluate_model(model, model_name):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # full class probabilities

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Overall metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # Per-class metrics
        per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except Exception as e:
            auroc = f"N/A (Error: {e})"

        return {
            "labels": all_labels,
            "probs": all_probs,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "auroc": auroc
        }

    def plot_confusion_matrix(cm, classes, output_path):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 17}  # annotation font size = 17 pt
        )
        plt.xlabel("Predicted", fontsize=18)
        plt.ylabel("True", fontsize=18)
        plt.title("Confusion Matrix", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_bar(metrics_dict, output_path, title):
        names = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        plt.figure(figsize=(10, 5))
        plt.bar(names, values, color="#2f6bb2")  # slightly darker blue
        plt.ylabel("Score", fontsize=18)
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_roc_curves(labels, probs, classes, output_path):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np

        # One-vs-rest ROC curves
        labels_bin = label_binarize(labels, classes=range(len(classes)))

        # Shades of blue (extend if you have more classes)
        blue_shades = ["#08306b", "#2f6bb2", "#6baed6", "#c6dbef"]

        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, lw=2,
                label=f"{class_name} (AUC = {roc_auc:.2f})",
                color=blue_shades[i % len(blue_shades)]
            )

        # Random baseline
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=18)
        plt.ylabel("True Positive Rate", fontsize=18)
        plt.title("AUROC Curves (One-vs-Rest)", fontsize=20)
        plt.legend(loc="lower right", fontsize=18, title_fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    for model_path, model_class in models_info:
        model = model_class(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        metrics = evaluate_model(model, model_name)

        metrics_txt_path = os.path.join(model_output_dir, "metrics.txt")
        with open(metrics_txt_path, "w") as f:
            f.write(f"Overall Accuracy: {metrics['acc']:.4f}\n")
            f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
            f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
            f.write(f"Overall F1-score: {metrics['f1']:.4f}\n")
            f.write(f"AUROC: {metrics['auroc']}\n\n")
            f.write("Per-Class Precision:\n")
            f.write(str(metrics["per_class_precision"]) + "\n")
            f.write("Per-Class Recall:\n")
            f.write(str(metrics["per_class_recall"]) + "\n")
            f.write("Per-Class F1-score:\n")
            f.write(str(metrics["per_class_f1"]) + "\n")

        # Plots
        plot_confusion_matrix(
            metrics["confusion_matrix"], classes=class_names,
            output_path=os.path.join(model_output_dir, "confusion_matrix.png")
        )

        plot_bar({
            "Accuracy": metrics["acc"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"]
        }, output_path=os.path.join(model_output_dir, "overall_metrics.png"), title="Overall Metrics")

        for metric_name, metric_values in [
            ("Precision", metrics["per_class_precision"]),
            ("Recall", metrics["per_class_recall"]),
            ("F1-score", metrics["per_class_f1"])
        ]:
            plot_bar(
                {class_names[i]: v for i, v in enumerate(metric_values)},
                output_path=os.path.join(model_output_dir, f"{metric_name.lower()}_per_class.png"),
                title=f"{metric_name} Per Class"
            )

        # AUROC curve plot
        try:
            plot_roc_curves(metrics["labels"], metrics["probs"], class_names,
                            output_path=os.path.join(model_output_dir, "auroc_curves.png"))
        except Exception as e:
            print(f"⚠ Could not plot AUROC curves for {model_name}: {e}")

        print(f"✔ Results for model '{model_name}' saved to: {model_output_dir}")
