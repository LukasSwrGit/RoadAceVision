import networks, data_loaders, image_video_tools, training_functions
import cv2
import importlib
import torch
import os
import testing_and_metrics, testing_and_metrics_multimodal
import timeseries_tools
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from training_functions import kfold_train_multimodal



import torch
import networks

# === 1) Daytime + Daytime ===
exp_day_day = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs1_video",
        "/home/ws/ugoby/master_thesis/data/pvs1_mfcc",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_mfcc.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs1_video_pvs1_mfcc.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs1_video_pvs1_mfcc"
}


# === 2) Evening + Evening ===
exp_evening_evening = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs9_video",
        "/home/ws/ugoby/master_thesis/data/pvs9_mfcc",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs9_video_pvs9_mfcc_v2.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs9_video_pvs9_mfcc_v2"
}


# === 3) Daytime + Daytime Augmented ===
exp_day_aug = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs1_video",
        "/home/ws/ugoby/master_thesis/data/(warp)pvs1_mfcc++",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/(warp)pvs1_mfcc++.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs1_video_pvs1_mfcc++.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs1_video_pvs1_mfcc++"
}


# === 4) Evening + Evening Augmented ===
exp_evening_aug = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs9_video",
        "/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs9_video_pvs9_mfcc++.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs9_video_pvs9_mfcc++"
}


# === 5) Nighttime + Nighttime (Normal) ===
exp_night_normal = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs9_video+",
        "/home/ws/ugoby/master_thesis/data/pvs9_mfcc",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs9_video+_pvs9_mfcc_v2.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs9_video+_pvs9_mfcc_v2"
}


# === 6) Nighttime + Nighttime (Augmented / Extreme Adverse) ===
exp_night_aug = {
    "dataset_type": "multimodal",
    "image_folders": [
        "/home/ws/ugoby/master_thesis/data/pvs9_video+",
        "/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp",
    ],
    "backbone_configs": [
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
        {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3},
    ],
    "dataset_builder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model_class": lambda bc, dev: (
        lambda backs, dims: networks.FlexibleFusionClassifier(backs, dims, nr_classes=3)
    )(*networks.build_backbones(bc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
    "resize": (224, 224),
    "epochs": 50,
    "model_save_path": "/home/ws/ugoby/master_thesis/models/kfold/MM_pvs9_video+_pvs9_mfcc++.pt",
    "num_classes": 3,
    "early_stopping_patience": 10,
    "min_delta": 0.001,
    "log_dir_base": "tensorboard/runs/kfold/MM_pvs9_video+_pvs9_mfcc++"
}






### call the kfold training



# === 1) Daytime + Daytime ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_day_day,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs1_video_pvs1_mfcc_kfold_summary.txt"
)
print("✅ Summary TXT (Daytime + Daytime):", summary_txt)


# === 2) Evening + Evening ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_evening_evening,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs9_video_pvs9_mfcc_v2_kfold_summary.txt"
)
print("✅ Summary TXT (Evening + Evening):", summary_txt)


# === 3) Daytime + Daytime Augmented ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_day_aug,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs1_video_pvs1_mfcc++_kfold_summary.txt"
)
print("✅ Summary TXT (Daytime + Daytime Augmented):", summary_txt)


# === 4) Evening + Evening Augmented ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_evening_aug,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs9_video_pvs9_mfcc++_kfold_summary.txt"
)
print("✅ Summary TXT (Evening + Evening Augmented):", summary_txt)


# === 5) Nighttime + Nighttime (Normal) ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_night_normal,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs9_video+_pvs9_mfcc_v2_kfold_summary.txt"
)
print("✅ Summary TXT (Nighttime + Normal):", summary_txt) 


# === 6) Nighttime + Nighttime (Augmented / Extreme Adverse) ===
fold_rows, summary_txt = kfold_train_multimodal(
    exp_night_aug,
    K=10,
    seed=42,
    summary_txt_path="/home/ws/ugoby/master_thesis/test_results/kfold/MM_pvs9_video+_pvs9_mfcc++_kfold_summary.txt"
)
print("✅ Summary TXT (Nighttime + Augmented):", summary_txt)
