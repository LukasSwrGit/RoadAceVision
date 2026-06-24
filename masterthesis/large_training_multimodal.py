#Imports

#Dataset information
#Directory: cd /home/ws/ugoby/.cache/kagglehub/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets/versions/2

#Example PVS1-9:
#(master_thesis) ugoby@aifb-sydsen-sun:~/.cache/kagglehub/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets/versions/2$ ls
#'PVS 1'  'PVS 2'  'PVS 3'  'PVS 4'  'PVS 5'  'PVS 6'  'PVS 7'  'PVS 8'  'PVS 9'

#Example PVS1:
#(master_thesis) ugoby@aifb-sydsen-sun:~/.cache/kagglehub/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets/versions/2/PVS 1$ ls
#dataset_gps.csv           dataset_gps_mpu_right.csv  dataset_mpu_left.csv   dataset_settings_left.csv   map.html                video_dataset_right.mp4  video_environment_dataset_left.mp4
#dataset_gps_mpu_left.csv  dataset_labels.csv         dataset_mpu_right.csv  dataset_settings_right.csv  video_dataset_left.mp4  video_environment.mp4    video_environment_dataset_right.mp4









import networks, data_loaders, image_video_tools, training_functions
import cv2
import importlib
import torch
import os
import testing_and_metrics, testing_and_metrics_multimodal
import timeseries_tools
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("- Imports completed successfully -")
class_names = ["dirt_road", "cobblestone_road", "asphalt_road"] 


BACKBONE_DIR = r"/home/ws/ugoby/master_thesis/models_conference/single_backbones"

VIDEO_BACKBONE_NORMAL   = os.path.join(BACKBONE_DIR, "video.pt")
VIDEO_BACKBONE_NIGHT    = os.path.join(BACKBONE_DIR, "video_degraded.pt")   
SENSOR_BACKBONE_NORMAL  = os.path.join(BACKBONE_DIR, "mfcc.pt")
SENSOR_BACKBONE_DEGRADED= os.path.join(BACKBONE_DIR, "mfcc_degraded.pt")


def conference_multimodal():
    """
    Train four multimodal fusion models (video + sensor) using all PVS1–9:
      - normal_normal
      - normal_degraded
      - degraded_normal
      - degraded_degraded
    Uses data_loaders.multi_pvs_multimodal_loader() via data_type="conf_multimodal".
    """

    import torch
    import networks
    import training_functions
    from torch.utils.tensorboard import SummaryWriter

    # --- Roots (each contains PVS 1..9 subfolders) ---
    VIDEO_NORMAL_ROOT   = r"/home/ws/ugoby/master_thesis/data_conference/video"
    VIDEO_DEG_ROOT      = r"/home/ws/ugoby/master_thesis/data_conference/video_degraded"
    MFCC_NORMAL_ROOT    = r"/home/ws/ugoby/master_thesis/data_conference/mfcc"
    MFCC_DEG_ROOT       = r"/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded"

    # --- Backbones (single-backbone checkpoints) ---
    VIDEO_BK_NORMAL = r"/home/ws/ugoby/master_thesis/models_conference/single_backbones/video.pt"
    VIDEO_BK_DEG    = r"/home/ws/ugoby/master_thesis/models_conference/single_backbones/video_degraded.pt"
    MFCC_BK_NORMAL  = r"/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc.pt"
    MFCC_BK_DEG     = r"/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc_degraded.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common = dict(
        device=device,
        learning_rate=1e-4,
        batch_size=32,
        resize=(224, 224),
        epochs=50,
        num_classes=3,
        data_type="conf_multimodal",
        early_stopping_patience=10,
        min_delta=0.001,
        model_class=lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
    )

    parameters = [
        # 1) normal_normal
        dict(
            **common,
            backbone_configs=[
                {"network": networks.ConvNeXtBaseClassifier, "path": VIDEO_BK_NORMAL, "nr_classes": 3},
                {"network": networks.ConvNeXtBaseClassifier, "path": MFCC_BK_NORMAL,  "nr_classes": 3},
            ],
            image_folders=[VIDEO_NORMAL_ROOT, MFCC_NORMAL_ROOT],
            writer=SummaryWriter(log_dir=r"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/multimodal/normal_normal"),
            model_save_path=r"/home/ws/ugoby/master_thesis/models_conference/multimodal/normal_normal.pt",
        ),

        # 2) normal_degraded (normal video, degraded sensor)
        dict(
            **common,
            backbone_configs=[
                {"network": networks.ConvNeXtBaseClassifier, "path": VIDEO_BK_NORMAL, "nr_classes": 3},
                {"network": networks.ConvNeXtBaseClassifier, "path": MFCC_BK_DEG,     "nr_classes": 3},
            ],
            image_folders=[VIDEO_NORMAL_ROOT, MFCC_DEG_ROOT],
            writer=SummaryWriter(log_dir=r"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/multimodal/normal_degraded"),
            model_save_path=r"/home/ws/ugoby/master_thesis/models_conference/multimodal/normal_degraded.pt",
        ),

        # 3) degraded_normal (degraded/nighttime video, normal sensor)
        dict(
            **common,
            backbone_configs=[
                {"network": networks.ConvNeXtBaseClassifier, "path": VIDEO_BK_DEG,   "nr_classes": 3},
                {"network": networks.ConvNeXtBaseClassifier, "path": MFCC_BK_NORMAL, "nr_classes": 3},
            ],
            image_folders=[VIDEO_DEG_ROOT, MFCC_NORMAL_ROOT],
            writer=SummaryWriter(log_dir=r"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/multimodal/degraded_normal"),
            model_save_path=r"/home/ws/ugoby/master_thesis/models_conference/multimodal/degraded_normal.pt",
        ),

        # 4) degraded_degraded
        dict(
            **common,
            backbone_configs=[
                {"network": networks.ConvNeXtBaseClassifier, "path": VIDEO_BK_DEG, "nr_classes": 3},
                {"network": networks.ConvNeXtBaseClassifier, "path": MFCC_BK_DEG,  "nr_classes": 3},
            ],
            image_folders=[VIDEO_DEG_ROOT, MFCC_DEG_ROOT],
            writer=SummaryWriter(log_dir=r"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/multimodal/degraded_degraded"),
            model_save_path=r"/home/ws/ugoby/master_thesis/models_conference/multimodal/degraded_degraded.pt",
        ),
    ]

    training_functions.train_multiple_runs(parameters)


experiments_conf = [
    # 1) normal_normal
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models_conference/multimodal/normal_normal.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/video.pt",           "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc.pt",            "nr_classes": 3},
        ],
        "image_roots": [
            "/home/ws/ugoby/master_thesis/data_conference/video",
            "/home/ws/ugoby/master_thesis/data_conference/mfcc",
        ],
        # Optional:
        "batch_size": 32,
        "resize": (224, 224),
        "use_amp": True,
        "class_names": ["asphalt_road", "dirt_road", "cobblestone_road"],
    },

    # 2) normal_degraded
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models_conference/multimodal/normal_degraded.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/video.pt",           "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc_degraded.pt",   "nr_classes": 3},
        ],
        "image_roots": [
            "/home/ws/ugoby/master_thesis/data_conference/video",
            "/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
        ],
        "batch_size": 32,
        "resize": (224, 224),
        "use_amp": True,
        "class_names": ["asphalt_road", "dirt_road", "cobblestone_road"],
    },

    # 3) degraded_normal
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models_conference/multimodal/degraded_normal.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/video_degraded.pt",  "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc.pt",            "nr_classes": 3},
        ],
        "image_roots": [
            "/home/ws/ugoby/master_thesis/data_conference/video_degraded",
            "/home/ws/ugoby/master_thesis/data_conference/mfcc",
        ],
        "batch_size": 32,
        "resize": (224, 224),
        "use_amp": True,
        "class_names": ["asphalt_road", "dirt_road", "cobblestone_road"],
    },

    # 4) degraded_degraded
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models_conference/multimodal/degraded_degraded.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/video_degraded.pt",  "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc_degraded.pt",   "nr_classes": 3},
        ],
        "image_roots": [
            "/home/ws/ugoby/master_thesis/data_conference/video_degraded",
            "/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
        ],
        "batch_size": 32,
        "resize": (224, 224),
        "use_amp": True,
        "class_names": ["asphalt_road", "dirt_road", "cobblestone_road"],
    },
]


def testing_conference_multimodal(
    experiments,
    output_dir="/home/ws/ugoby/master_thesis/test_conference/multimodal",
    num_classes=3,
):
    """
    Evaluate one or more conference multimodal (conf_multimodal) fusion models.
    Each experiment dict must include:
      - "fusion_checkpoint": path to the saved fusion .pt file
      - "backbone_configs": list of backbones (for build_backbones)
      - "image_roots" or "image_folders": [image_root, sensor_root]
        (each root contains PVS1–9 subfolders)
    Optional keys:
      - "batch_size" (default: 32)
      - "resize" (default: (224, 224))
      - "use_amp" (default: True)
      - "class_names" (default: asphalt_road, dirt_road, cobblestone_road)
    """

    import testing_and_metrics_multimodal as testing_mm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for exp in experiments:
        fusion_ckpt = exp["fusion_checkpoint"]
        backbone_configs = exp["backbone_configs"]
        roots = exp.get("image_roots") or exp.get("image_folders")

        if not roots or len(roots) != 2:
            raise ValueError("conf_multimodal requires exactly two roots: [image_root, sensor_root].")

        image_root, sensor_root = roots
        batch_size = exp.get("batch_size", 32)
        resize = exp.get("resize", (224, 224))
        use_amp = exp.get("use_amp", True)
        class_names = exp.get("class_names", ["asphalt_road", "dirt_road", "cobblestone_road"])

        print(f"\n🔹 Testing conference multimodal model: {os.path.basename(fusion_ckpt)}")

        # === Build test loader using all PVS1–9 ===
        _, _, test_loader = data_loaders.multi_pvs_multimodal_loader(
            image_root=image_root,
            sensor_root=sensor_root,
            batch_size=batch_size,
            resize=resize,
            return_class_map=False,
            num_classes=num_classes,
        )

        # === Evaluate fusion model ===
        testing_mm.test_and_evaluate_multimodal(
            test_loader=test_loader,
            fusion_checkpoint=fusion_ckpt,
            backbone_configs=backbone_configs,
            output_dir=output_dir,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            use_amp=use_amp,
        )

        print(f"✔ Finished testing {os.path.basename(fusion_ckpt)}")





if __name__ == "__main__":
    #conference_multimodal()
    testing_conference_multimodal(experiments_conf)