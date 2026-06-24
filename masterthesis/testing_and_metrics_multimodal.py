def test_and_evaluate_multimodal(
    test_loader,
    fusion_checkpoint,           # path to .pt with FlexibleFusionClassifier state_dict
    backbone_configs,            # list for build_backbones: [{"network": ..., "path": ..., "nr_classes": 3}, ...]
    output_dir,
    device=None,
    num_classes=3,
    class_names=None,
    use_amp=False,
    debug=False,
):
    """
    Evaluate a multimodal FlexibleFusionClassifier, compute metrics/plots,
    and robustly compute multiclass AUROC (fp16-safe).
    """
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networks
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, roc_curve, auc
    )
    from sklearn.preprocessing import label_binarize

    # ---------- helpers ----------
    def compute_auroc_safe(y_true_np, y_prob_np, n_classes):
        """
        Make y_prob a valid probability matrix (fp16/NaN safe) before calling sklearn.
        Returns float AUROC or raises.
        """
        y_prob = np.asarray(y_prob_np, dtype=np.float64)

        # Replace non-finite rows with uniform distribution
        bad = ~np.isfinite(y_prob).all(axis=1)
        if bad.any():
            y_prob[bad] = 1.0 / n_classes

        # Renormalize rows to sum to 1
        row_sums = y_prob.sum(axis=1, keepdims=True)
        zero = (row_sums == 0.0)
        if zero.any():
            row_sums[zero] = 1.0
        y_prob = y_prob / row_sums

        if debug:
            print("[AUROC] shape:", y_prob.shape)
            print("[AUROC] row sums (first 5):", y_prob[:5].sum(axis=1))

        return roc_auc_score(y_true_np, y_prob, multi_class="ovr")

    def plot_confusion_matrix(cm_, classes, output_path):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 17}
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
        plt.bar(names, values, color="#2f6bb2")
        plt.ylabel("Score", fontsize=18)
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_roc_curves(labels, probs, classes, output_path):
        labels_bin = label_binarize(labels, classes=range(len(classes)))
        blue_shades = ["#08306b", "#2f6bb2", "#6baed6", "#c6dbef"]

        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(classes):
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, lw=2,
                label=f"{class_name} (AUC = {roc_auc:.2f})",
                color=blue_shades[i % len(blue_shades)]
            )
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

    # ---------- device ----------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- name mapping ----------
    name_map = {
        "asphalt": "Asphalt",
        "asphalt_road": "Asphalt",
        "asphalt_night": "Asphalt (Night)",
        "dirt": "Dirt",
        "dirt_road": "Dirt",
        "dirt_night": "Dirt (Night)",
        "cobblestone": "Cobblestone",
        "cobblestone_road": "Cobblestone",
        "cobblestone_night": "Cobblestone (Night)"
    }

    if class_names is None:
        if num_classes == 3:
            class_names = ["asphalt_road", "dirt_road", "cobblestone_road"]
        elif num_classes == 6:
            class_names = [
                "asphalt_road", "dirt_road", "cobblestone_road",
                "asphalt_night", "dirt_night", "cobblestone_night"
            ]
        else:
            class_names = [str(i) for i in range(num_classes)]
    class_names = [name_map.get(n, n) for n in class_names]

    # ---------- build model ----------
    backbones, feature_dims = networks.build_backbones(backbone_configs, device)
    model = networks.FlexibleFusionClassifier(
        backbones=backbones, feature_dims=feature_dims, nr_classes=num_classes
    ).to(device)

    # load weights
    state = torch.load(fusion_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------- output dir ----------
    model_name = os.path.splitext(os.path.basename(fusion_checkpoint))[0]
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # ---------- eval loop ----------
    all_preds, all_labels, all_probs = [], [], []
    use_cuda_amp = bool(use_amp and device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise ValueError("Expected test_loader to yield (*images, labels).")

            *modal_images, labels = batch
            if len(modal_images) != len(backbones):
                raise ValueError(
                    f"Modalities ({len(modal_images)}) do not match backbones ({len(backbones)})."
                )

            images_on_device = [img.to(device, non_blocking=True) for img in modal_images]
            labels = labels.to(device, non_blocking=True)

            if use_cuda_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(*images_on_device)  # logits (possibly fp16)
            else:
                outputs = model(*images_on_device)      # logits (fp32)

            # IMPORTANT: do softmax in float32 for numeric stability
            probs = F.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # ---------- metrics ----------
    all_labels_np = np.asarray(all_labels)
    all_preds_np  = np.asarray(all_preds)
    all_probs_np  = np.asarray(all_probs, dtype=np.float64)

    acc = accuracy_score(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)

    per_class_precision = precision_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    per_class_recall    = recall_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    per_class_f1        = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)

    cm = confusion_matrix(all_labels_np, all_preds_np)

    # robust AUROC
    try:
        auroc = compute_auroc_safe(all_labels_np, all_probs_np, num_classes)
    except Exception as e:
        auroc = f"N/A (Error: {e})"

    if debug:
        print(f"[DEBUG] acc={acc:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}, auroc={auroc}")

    # ---------- save metrics ----------
    metrics_txt_path = os.path.join(model_output_dir, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall F1-score: {f1:.4f}\n")
        f.write(f"AUROC: {auroc}\n\n")
        f.write("Per-Class Precision:\n")
        f.write(str(per_class_precision) + "\n")
        f.write("Per-Class Recall:\n")
        f.write(str(per_class_recall) + "\n")
        f.write("Per-Class F1-score:\n")
        f.write(str(per_class_f1) + "\n")

    # ---------- plots ----------
    plot_confusion_matrix(
        cm, classes=class_names,
        output_path=os.path.join(model_output_dir, "confusion_matrix.png")
    )

    plot_bar(
        {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1-score": f1},
        output_path=os.path.join(model_output_dir, "overall_metrics.png"),
        title="Overall Metrics"
    )

    for metric_name, metric_values in [
        ("Precision", per_class_precision),
        ("Recall",    per_class_recall),
        ("F1-score",  per_class_f1),
    ]:
        plot_bar(
            {class_names[i]: float(v) for i, v in enumerate(metric_values)},
            output_path=os.path.join(model_output_dir, f"{metric_name.lower()}_per_class.png"),
            title=f"{metric_name} Per Class"
        )

    try:
        # Use sanitized probs for curves too
        # (they were already made safe in compute_auroc_safe)
        # Re-sanitize here to be explicit:
        probs_for_plot = np.asarray(all_probs_np, dtype=np.float64)
        rs = probs_for_plot.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        probs_for_plot = probs_for_plot / rs

        plot_roc_curves(all_labels_np, probs_for_plot, class_names,
                        output_path=os.path.join(model_output_dir, "auroc_curves.png"))
    except Exception as e:
        print(f"⚠ Could not plot AUROC curves for {model_name}: {e}")

    print(f"✔ Results for model '{model_name}' saved to: {model_output_dir}")
