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



train_1 = "start_experiment_10k" #necessary for original mfcc data right now
train_2 = "degraded_2_pvs1_mfcc++"
train_3 = "degraded_3_pvs1_mfcc++"
train_4 = "degraded_4_pvs1_mfcc++"
train_5 = "degraded_5_pvs1_mfcc++"
train_6 = "degraded_6_pvs1_mfcc++"


def build_degraded_label_datasets(
    in_dir: str = "/home/ws/ugoby/master_thesis/data_conference/labels",
    out_subdir: str = "degraded",
    cols=("acc_y_combined",),
    sample_rate_hz: float = 100.0,
) -> None:
    """
    Applies one degradation level to all synchronized label CSV files
    in the given directory and saves results into a 'degraded' subfolder.

    Produces files named:
        synchronized_labels_mpu_left_PVS{i}__degraded.csv
    """

    # Create output directory if not existing
    out_dir = os.path.join(in_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Process each synchronized label file
    for fname in sorted(os.listdir(in_dir)):
        if not fname.startswith("synchronized_labels_mpu_left_PVS") or not fname.endswith(".csv"):
            continue

        csv_path = os.path.join(in_dir, fname)
        pvs_id = fname.replace(".csv", "")

        print(f"Processing {fname} -> {out_dir}/{pvs_id}__degraded.csv")

        # Change working directory to prevent nested path issues
        cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            # ✅ Do NOT append "__degraded" manually, the helper adds it
            timeseries_tools.apply_six_levels_and_save(
                csv_path=csv_path,
                out_prefix=pvs_id,
                cols=cols,
                sample_rate_hz=sample_rate_hz,
            )
        finally:
            os.chdir(cwd)

    print("✅ All degraded label datasets created successfully.")


def generate_mfcc_images():

    timeseries_tools.apply_six_levels_and_save(
        csv_path="/home/ws/ugoby/master_thesis/data/pvs1_X01_label.csv",
        out_prefix="degraded",          # will create df_degraded_L1.csv ... L4.csv
        cols=("acc_y_combined",),          # which columns to degrade
        sample_rate_hz=100.0               # your sensor’s rate
    )


    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_1}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_1}", target_count=33090)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_2}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_2}", target_count=33090)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_3}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_3}", target_count=33090)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_4}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_4}", target_count=33090)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_5}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_5}", target_count=33090)

    X01_label = pd.read_csv(rf"/home/ws/ugoby/master_thesis/data/labels/{train_6}.csv")

    timeseries_tools.generate_images_fixed_count(X01_label, "acc_y_combined", method="mfcc", output_folder=rf"/home/ws/ugoby/master_thesis/data/{train_6}", target_count=33090)

#Conference pvs1-9
def generate_all_pvs_mfcc(
    base_in_synced="/home/ws/ugoby/master_thesis/data_conference/labels/synced",
    base_in_degraded="/home/ws/ugoby/master_thesis/data_conference/labels/degraded",
    base_out_root_normal="/home/ws/ugoby/master_thesis/data_conference/mfcc",
    base_out_root_degraded="/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
    video_root="/home/ws/ugoby/master_thesis/data_conference/video",
    method="mfcc",
    data_column="acc_y_combined",
    degraded=False,
    window_size=50,
):
    """
    For each PVS: count labeled frames in video folder (asphalt/cobblestone/dirt),
    set target_count to that number, and generate exactly that many spectrograms.
    Each spectrogram is named exactly like its corresponding frame (e.g. 01001_asphalt.jpg).
    """
    import os
    import pandas as pd

    base_in = base_in_degraded if degraded else base_in_synced
    base_out = base_out_root_degraded if degraded else base_out_root_normal
    os.makedirs(base_out, exist_ok=True)

    for i in range(1, 10):
        print(f"\n=== Processing PVS{i} ===")

        # Paths
        csv_name = (f"synchronized_labels_mpu_left_PVS{i}__degraded.csv"
                    if degraded else f"synchronized_labels_mpu_left_PVS{i}_synced.csv")
        csv_path = os.path.join(base_in, csv_name)
        video_folder = os.path.join(video_root, f"PVS {i}")
        out_dir = os.path.join(base_out, f"PVS{i}")
        os.makedirs(out_dir, exist_ok=True)

        # Checks
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV file: {csv_path}"); continue
        if not os.path.exists(video_folder):
            print(f"⚠️ Missing video folder: {video_folder}"); continue

        df = pd.read_csv(csv_path)
        if df.empty or 'image_filename' not in df.columns:
            print(f"⚠️ Invalid or empty CSV: {csv_path}"); continue

        # Video-side: collect valid labeled frames (names)
        all_frames = sorted([f for f in os.listdir(video_folder)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        target_filenames = [f for f in all_frames if any(k in f for k in ["asphalt", "cobblestone", "dirt"])]
        if not target_filenames:
            print(f"⚠️ No labeled frames in {video_folder}, skipping."); continue

        # CSV-side: restrict rows to those filenames (guard against leftovers)
        df_use = df[df['image_filename'].isin(set(target_filenames))].copy()
        if df_use.empty:
            print(f"⚠️ No matching CSV rows for labeled frames in PVS{i}, skipping."); continue

        target_count = len(target_filenames)
        print(f"Found {len(df_use)} labeled rows; target spectrograms: {target_count}")

        # Generate exactly target_count windows, save with exact frame filenames
        _ = timeseries_tools.generate_images_fixed_count(
            df=df_use,
            data_column=data_column,
            method=method,
            output_folder=out_dir,
            target_count=target_count,
            window_size=window_size,
            target_filenames=target_filenames,  # ← enforces unique, frame-matched names
        )

        # Verify
        generated = [f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"✅ Generated {len(generated)} MFCCs (expected {target_count}).")
        if len(generated) != target_count:
            print("⚠️ Mismatch between MFCC and frame counts!")

def single_backbones():

    #MULTI SOURCE Singlebackbone
    parameters = [
    {
        "data_type": "multi_source",
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs1_video",
            r"/home/ws/ugoby/master_thesis/data/pvs9_video++",
        ],
        "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
        "batch_size": 32,
        "resize": (224, 224),
        "num_classes": 3,
        "learning_rate": 1e-4,
        "epochs": 50,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "writer": SummaryWriter(log_dir="/home/ws/ugoby/master_thesis/tensorboard/runs/single_backbones/multi_source_pvs1_video_pvs9_video++"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/single_backbones/multi_source_pvs1_video_pvs9_video++.pt",

        "early_stopping_patience": 5,
        "min_delta": 0.001
    },
    ]
    
    #SINGLE SOURCE Singlebackbone
    parameters = [
    {
        "image_folder": fr"/home/ws/ugoby/master_thesis/data/pvs9_mfcc",
        "csv_file": r"",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": networks.ConvNeXtBaseClassifier,
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir=fr"/home/ws/ugoby/master_thesis/tensorboard/runs/single_backbones\pvs9_mfcc_v2"),
        "model_save_path": fr"/home/ws/ugoby/master_thesis/models/single_backbones/pvs9_mfcc_v2.pt",
        "data_type": "mfcc",
        "num_classes": 3,  # 3 for normal , 6 for night

        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },
    ]

    ############# Actually used right now #############
    #Original
    pvs_base_path = rf"/home/ws/ugoby/master_thesis/data_conference/mfcc/{train_1}"

    parameters = [
        # --- Model 1: Mixed samples disabled ---
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS{i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir=f"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/{train_1}"
            ),
            "model_save_path": rf"/home/ws/ugoby/master_thesis/models_conference/single_backbones/{train_1}.pt",
            "early_stopping_patience": 10,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
    ]

    #Degraded
    pvs_base_path = "/home/ws/ugoby/master_thesis/data_conference/mfcc/sensor_degraded_10k"

    parameters = [
        # --- Model 1: Degraded sensor datasets ---
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS{i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir="/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/sensor_degraded_10k"
            ),
            "model_save_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_degraded_10k.pt",
            "early_stopping_patience": 20,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS{i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir="/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/sensor_degraded_10k2"
            ),
            "model_save_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_degraded_10k2.pt",
            "early_stopping_patience": 20,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS{i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir="/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/sensor_degraded_10k3"
            ),
            "model_save_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_degraded_10k3.pt",
            "early_stopping_patience": 20,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS{i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir="/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/sensor_degraded_10k4"
            ),
            "model_save_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_degraded_10k4.pt",
            "early_stopping_patience": 20,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
    ]


    training_functions.legacy_train_multiple_runs(parameters)

    


def multimodal():

    ### Multimodal Training ###
    # The number of backbone configs determine the number of feature extraction backbones to be used
    # The number of image folders should match the number of backbone configs
    # The files inside the image folders corresponding to the same sample must have the same name 
    # The backbones that are loaded should be pretrained on the same dataset as the images in the folders

    
    parameters = [
        #Daytime + Daytime
        {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs1_mfcc.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs1_video",
            r"/home/ws/ugoby/master_thesis/data/pvs1_mfcc",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs1_video_pvs1_mfcc"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs1_video_pvs1_mfcc.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },


    #Evening + Evening
    {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs9_video",
            r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs9_video_pvs9_mfcc_v2"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs9_video_pvs9_mfcc_v2.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },

    #Adverse
    #Daytime + Daytime Augmented
        {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/(warp)pvs1_mfcc++.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs1_video",
            r"/home/ws/ugoby/master_thesis/data/(warp)pvs1_mfcc++",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs1_video_pvs1_mfcc++"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs1_video_pvs1_mfcc++.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },

    #Evening + Evening Augmented
    {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs9_video",
            r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs9_video_pvs9_mfcc++"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs9_video_pvs9_mfcc++.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },

    #Nightime + Nighttime normal
    {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs9_video+",
            r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs9_video+_pvs9_mfcc_v2"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs9_video+_pvs9_mfcc_v2.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },

    #Extreme Adverse Conditions
    #Nightime + Nighttime Augmented
    {
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": r"/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3},

        ],
        "image_folders": [
            r"/home/ws/ugoby/master_thesis/data/pvs9_video+",
            r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp",
        ],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model_class": lambda bc, dev: (
            lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
        )(*networks.build_backbones(bc, dev)),
        "resize": (224, 224),
        "epochs": 50,
        "writer": SummaryWriter(log_dir="tensorboard/runs/MM_pvs9_video+_pvs9_mfcc++"),
        "model_save_path": r"/home/ws/ugoby/master_thesis/models/MM_pvs9_video+_pvs9_mfcc++.pt",
        "data_type": "multimodal",
        "num_classes": 3,
        
        # Early stopping additions:
        "early_stopping_patience": 10,
        "min_delta": 0.001
    },
    ]

    training_functions.train_multiple_runs(parameters)


def multimodal_special():

    ## Special Multimodal Training ##

    video_b_path = r"/home/ws/ugoby/master_thesis/models/single_backbones/multi_source_pvs1_video_pvs9_video++.pt"
    mfcc_b_path  = r"/home/ws/ugoby/master_thesis/models/single_backbones/multi_source_pvs1_mfcc++_pvs9_mfcc.pt"

    parameters = [
        {
            "backbone_configs": [
                {"network": networks.ConvNeXtBaseClassifier, "path": video_b_path, "nr_classes": 3},  # video backbone
                {"network": networks.ConvNeXtBaseClassifier, "path": mfcc_b_path,  "nr_classes": 3},  # mfcc backbone
            ],

            # Explicit folder pairs instead of image_folders
            "folder_pairs": [
                (r"/home/ws/ugoby/master_thesis/data/pvs1_video",   r"/home/ws/ugoby/master_thesis/data/pvs1_mfcc++"),
                (r"/home/ws/ugoby/master_thesis/data/pvs9_video++", r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc"),
            ],

            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate": 1e-4,
            "batch_size": 32,

            "model_class": lambda bc, dev: (
                lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
            )(*networks.build_backbones(bc, dev)),

            "resize": (224, 224),
            "epochs": 50,
            "writer": SummaryWriter(log_dir="/home/ws/ugoby/master_thesis/tensorboard/runs/multimodal/MM_multi_source_2bb"),
            "model_save_path": r"/home/ws/ugoby/master_thesis/models/multimodal/MM_multi_source_2bb.pt",

            # NEW: special loader branch, no duplication
            "data_type": "multimodal_special",
            "duplicate": False,          # <- important: yields (video, mfcc, label)

            "num_classes": 3,
            "early_stopping_patience": 10,
            "min_delta": 0.001,
        },
    ]

    training_functions.train_multiple_runs(parameters)

    ## Special Multimodal Training (4 backbones, duplicated) ##

    video_b1_path = r"/home/ws/ugoby/master_thesis/models/single_backbones/pretty_good/pvs1_video.pt"
    video_b2_path = r"/home/ws/ugoby/master_thesis/models/single_backbones/pretty_good/pvs9_video++.pt"
    mfcc_b1_path  = r"/home/ws/ugoby/master_thesis/models/single_backbones/pretty_good/pvs1_mfcc++.pt"
    mfcc_b2_path  = r"/home/ws/ugoby/master_thesis/models/single_backbones/pretty_good/pvs9_mfcc.pt"

    parameters = [
        {
            "backbone_configs": [
                {"network": networks.ConvNeXtBaseClassifier, "path": video_b1_path, "nr_classes": 3},  # video backbone #1
                {"network": networks.ConvNeXtBaseClassifier, "path": video_b2_path, "nr_classes": 3},  # video backbone #2
                {"network": networks.ConvNeXtBaseClassifier, "path": mfcc_b1_path,  "nr_classes": 3},  # mfcc backbone #1
                {"network": networks.ConvNeXtBaseClassifier, "path": mfcc_b2_path,  "nr_classes": 3},  # mfcc backbone #2
            ],

            # Explicit folder pairs instead of image_folders
            "folder_pairs": [
                (r"/home/ws/ugoby/master_thesis/data/pvs1_video",   r"/home/ws/ugoby/master_thesis/data/pvs1_mfcc++"),
                (r"/home/ws/ugoby/master_thesis/data/pvs9_video++", r"/home/ws/ugoby/master_thesis/data/pvs9_mfcc"),
            ],

            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate": 1e-4,
            "batch_size": 32,

            "model_class": lambda bc, dev: (
                lambda backbones, dims: networks.FlexibleFusionClassifier(backbones, dims, nr_classes=3)
            )(*networks.build_backbones(bc, dev)),

            "resize": (224, 224),
            "epochs": 50,
            "writer": SummaryWriter(log_dir="/home/ws/ugoby/master_thesis/tensorboard/runs/multimodal/MMspecial_4b_pvs1++pvs9++"),
            "model_save_path": r"/home/ws/ugoby/master_thesis/models/multimodal/MMspecial_4b_pvs1++pvs9++.pt",

            # NEW: special loader branch, duplication (default True)
            "data_type": "multimodal_special",
            "duplicate": True,           # <- yields (video, video, mfcc, mfcc, label)

            "num_classes": 3,
            "early_stopping_patience": 10,
            "min_delta": 0.001,
        },
    ]



###### TESTING ######
# -------------------------
# Settings for each run - singlebackbone
# 
# -------------------------

#test loader
def get_loader(
    dataset_type,
    image_folder,
    batch_size=32,
    resize=(224, 224),
    return_class_map=False,
    use_mixed_samples=False,
):
    """
    General-purpose loader for both single-folder and multi-folder datasets.
    Automatically selects the right loader (single or multi).
    """
    is_multi = isinstance(image_folder, (list, tuple))

    if dataset_type == "mfcc":
        if is_multi:
            return data_loaders.mfcc_loader_multi(
                image_folders=image_folder,
                batch_size=batch_size,
                resize=resize,
                return_class_map=return_class_map,
                num_classes=3,
                use_mixed_samples=use_mixed_samples,
            )[2]
        else:
            return data_loaders.mfcc_loader(
                image_folder,
                batch_size,
                resize,
                return_class_map,
                use_mixed_samples=use_mixed_samples,
            )[2]

    elif dataset_type == "cwt":
        if is_multi:
            return data_loaders.cwt_loader_multi(
                image_folders=image_folder,
                batch_size=batch_size,
                resize=resize,
                return_class_map=return_class_map,
                num_classes=3,
                use_mixed_samples=use_mixed_samples,
            )[2]
        else:
            return data_loaders.cwt_loader(
                image_folder,
                batch_size,
                resize,
                return_class_map,
                use_mixed_samples=use_mixed_samples,
            )[2]

    elif dataset_type == "raw":
        if is_multi:
            return data_loaders.general_image_loader_multi(
                image_folders=image_folder,
                batch_size=batch_size,
                resize=resize,
                return_class_map=return_class_map,
                num_classes=3,
                use_mixed_samples=use_mixed_samples,
            )[2]
        else:
            return data_loaders.general_image_loader(
                image_folder,
                batch_size,
                resize,
                return_class_map,
                use_mixed_samples=use_mixed_samples,
            )[2]

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

#Single folder input
experiments = [
    {
        "model_path": rf"/home/ws/ugoby/master_thesis/models/thesis/(warp)pvs1_mfcc++.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",  # options: "mfcc", "cwt", "raw"
        "image_folder": rf"/home/ws/ugoby/master_thesis/data/(warp)pvs1_mfcc++",
        "use_mixed_samples": True,
    },
]

#Multi folder input
experiments = [
    {
        "model_path": r"/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_all_pvs.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",
        "image_folder": [
            os.path.join(r"/home/ws/ugoby/master_thesis/data_conference/mfcc/only_sensor_no_sync_start_experiment", f"PVS{i}")
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,  # same as training default
    },
]

#Actually used right now
#Original
experiments = [
    {
        "model_path": rf"/home/ws/ugoby/master_thesis/models_conference/single_backbones/{train_1}.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",
        "image_folder": [
            os.path.join(
                rf"/home/ws/ugoby/master_thesis/data_conference/mfcc/{train_1}",
                f"PVS{i}"
            )
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,
    },
]

#Degraded
experiments = [
    {
        "model_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/sensor_degraded_10k.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",
        "image_folder": [
            os.path.join(
                "/home/ws/ugoby/master_thesis/data_conference/mfcc/sensor_degraded_10k",
                f"PVS{i}"
            )
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,
    },
]



def testing_single_backbones(experiments, output_dir="/home/ws/ugoby/master_thesis/test_conference/single_backbones/", num_classes=3):
    for exp in experiments:
        print(f"\n🔹 Testing model {os.path.basename(exp['model_path'])} on {exp['dataset_type']} dataset")

        test_loader = get_loader(
            dataset_type=exp["dataset_type"],
            image_folder=exp["image_folder"],
            batch_size=32,
            resize=(224,224),
            return_class_map=False,
            use_mixed_samples=exp.get("use_mixed_samples", False),
        )

        models_info = [(exp["model_path"], exp["model_class"])]

        testing_and_metrics.test_and_evaluate(
            test_loader=test_loader,
            models_info=models_info,
            output_dir=output_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_classes=num_classes,
            class_names=["asphalt_road", "dirt_road", "cobblestone_road"]
        )



###### TESTING ######
# -------------------------
# Settings for each run - Multimodal
# -------------------------
experiments_multimodal = [
    #1video 1mfcc
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs1_video_pvs1_mfcc.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_mfcc.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs1_video",
            "/home/ws/ugoby/master_thesis/data/pvs1_mfcc"
        ],
        "dataset_type": "multimodal"
    },
    #1video 1mfcc++
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs1_video_pvs1_mfcc++.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs1_video_v2.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/(warp)pvs1_mfcc++.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs1_video",
            "/home/ws/ugoby/master_thesis/data/(warp)pvs1_mfcc++"
        ],
        "dataset_type": "multimodal"
    },
    #9video 9mfcc
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs9_video_pvs9_mfcc_v2.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs9_video",
            "/home/ws/ugoby/master_thesis/data/pvs9_mfcc"
        ],
        "dataset_type": "multimodal"
    },
    #9video 9mfcc++
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs9_video_pvs9_mfcc++.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs9_video",
            "/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp"
        ],
        "dataset_type": "multimodal"
    },
    #9video+ 9mfcc
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs9_video+_pvs9_mfcc_v2.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc_v2.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs9_video+",
            "/home/ws/ugoby/master_thesis/data/pvs9_mfcc"
        ],
        "dataset_type": "multimodal"
    },
    #9video+ 9mfcc++
    {
        "fusion_checkpoint": "/home/ws/ugoby/master_thesis/models/MM_pvs9_video+_pvs9_mfcc++.pt",
        "backbone_configs": [
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_video+.pt", "nr_classes": 3},
            {"network": networks.ConvNeXtBaseClassifier, "path": "/home/ws/ugoby/master_thesis/models/thesis/pvs9_mfcc++_timewarp.pt", "nr_classes": 3}
        ],
        "image_folders": [
            "/home/ws/ugoby/master_thesis/data/pvs9_video+",
            "/home/ws/ugoby/master_thesis/data/pvs9_mfcc++_timewarp"
        ],
        "dataset_type": "multimodal"
    },
]

def testing_multimodal_models(
    experiments_multimodal,
    output_dir="/home/ws/ugoby/master_thesis/test_results/multimodal",
    num_classes=3
    ):

        for exp in experiments_multimodal:
            print(f"\n🔹 Testing fusion model {os.path.basename(exp['fusion_checkpoint'])} "
                f"on {exp['dataset_type']} dataset")

            # --- Get multimodal test loader ---
            if exp["dataset_type"] == "multimodal":
                test_loader = data_loaders.multimodal_loader(
                    folders=exp["image_folders"],
                    batch_size=32,
                    resize=(224, 224),
                    return_class_map=False,
                    num_classes=exp.get("num_classes", 3)
                )[2]  # take only the test loader
            else:
                test_loader = get_loader(
                    dataset_type=exp["dataset_type"],
                    image_folder=exp["image_folder"],
                    batch_size=32,
                    resize=(224, 224),
                    return_class_map=False
    )


            # --- Run evaluation ---
            testing_and_metrics_multimodal.test_and_evaluate_multimodal(
                test_loader=test_loader,
                fusion_checkpoint=exp["fusion_checkpoint"],
                backbone_configs=exp["backbone_configs"],
                output_dir=output_dir,
                device=exp.get("device", None),
                num_classes=num_classes,
                class_names=["asphalt_road", "dirt_road", "cobblestone_road"],
                use_amp=exp.get("use_amp", False)
            )





if __name__ == "__main__":
    #generate_mfcc_images()
    #build_degraded_label_datasets()
    #generate_all_pvs_mfcc(degraded=True)
    #single_backbones()
    #testing_single_backbones(experiments)
    #multimodal()
    #testing_multimodal_models(experiments_multimodal)
    #multimodal_special()

    generate_all_pvs_mfcc(
    base_in_synced="/home/ws/ugoby/master_thesis/data_conference/labels/synced",
    base_in_degraded="/home/ws/ugoby/master_thesis/data_conference/labels/degraded",
    base_out_root_normal="/home/ws/ugoby/master_thesis/data_conference/mfcc",
    base_out_root_degraded="/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
    video_root="/home/ws/ugoby/master_thesis/data_conference/video",
    method="mfcc",
    data_column="acc_y_combined",
    degraded = False,
    )
