import networks, data_loaders, image_video_tools, training_functions
import cv2
import importlib
import torch
import os
import testing_and_metrics, testing_and_metrics_multimodal
import timeseries_tools
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import image_video_tools
import albumentations as A

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("- Imports completed successfully -")
class_names = ["dirt_road", "cobblestone_road", "asphalt_road"] 



#### GENERAL ####

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

#### SENSOR ####
## DEGRADED ##  
train_1 = "mfcc_degraded"
def build_degraded_label_datasets(
    in_dir: str = "/home/ws/ugoby/master_thesis/data_conference/labels/synced",
    out_dir: str = "/home/ws/ugoby/master_thesis/data_conference/labels/degraded",
    cols=("acc_y_combined",),
    sample_rate_hz: float = 100.0,
) -> None:
    """
    Reads synchronized CSVs like:
      /labels/synced/synchronized_labels_mpu_left_PVS{i}_synced.csv
    and writes degraded CSVs to:
      /labels/degraded/synchronized_labels_mpu_left_PVS{i}__degraded.csv
    (drops the '_synced' in the output name).
    """
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(in_dir)):
        # Expect files like synchronized_labels_mpu_left_PVS1_synced.csv
        if not (fname.startswith("synchronized_labels_mpu_left_PVS") and fname.endswith("_synced.csv")):
            continue

        csv_path = os.path.join(in_dir, fname)
        base_no_ext = os.path.splitext(fname)[0]                        # e.g. synchronized_labels_mpu_left_PVS1_synced
        out_prefix = base_no_ext.replace("_synced", "")                 # -> synchronized_labels_mpu_left_PVS1

        print(f"Processing {fname} -> {out_dir}/{out_prefix}__degraded.csv")

        # Run in out_dir so helper writes there; it appends '__degraded' itself
        cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            timeseries_tools.apply_six_levels_and_save(
                csv_path=csv_path,
                out_prefix=out_prefix,     # helper will create {out_prefix}__degraded.csv
                cols=cols,
                sample_rate_hz=sample_rate_hz,
            )
        finally:
            os.chdir(cwd)

    print("✅ All degraded label datasets created successfully.")

def generate_all_pvs_mfcc_degraded(
    base_in_degraded="/home/ws/ugoby/master_thesis/data_conference/labels/degraded",
    base_out_root_degraded="/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
    video_root="/home/ws/ugoby/master_thesis/data_conference/video",
    method="mfcc",
    data_column="acc_y_combined",
    window_size=50,
):
    """
    Generate degraded MFCC images for PVS1–PVS9 using previously degraded CSV files.
    Saves outputs to /home/ws/ugoby/master_thesis/data_conference/mfcc_degraded.

    Each MFCC image matches a labeled video frame (e.g. 01001_asphalt.jpg),
    ensuring identical counts and filenames between degraded and normal datasets.
    """
    import os
    import pandas as pd

    base_in = base_in_degraded
    base_out = base_out_root_degraded
    os.makedirs(base_out, exist_ok=True)

    for i in range(1, 10):
        print(f"\n=== Processing DEGRADATED PVS{i} ===")

        # Define paths
        csv_name = f"synchronized_labels_mpu_left_PVS{i}__degraded.csv"
        csv_path = os.path.join(base_in, csv_name)
        video_folder = os.path.join(video_root, f"PVS {i}")
        out_dir = os.path.join(base_out, f"PVS{i}")
        os.makedirs(out_dir, exist_ok=True)

        # Check input availability
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing degraded CSV file: {csv_path}")
            continue
        if not os.path.exists(video_folder):
            print(f"⚠️ Missing video folder: {video_folder}")
            continue

        # Load CSV
        df = pd.read_csv(csv_path)
        if df.empty or "image_filename" not in df.columns:
            print(f"⚠️ Invalid or empty degraded CSV: {csv_path}")
            continue

        # Collect video frame names
        all_frames = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        target_filenames = [
            f for f in all_frames
            if any(k in f for k in ["asphalt", "cobblestone", "dirt"])
        ]
        if not target_filenames:
            print(f"⚠️ No labeled frames in {video_folder}, skipping.")
            continue

        # Filter CSV rows that correspond to valid labeled frames
        df_use = df[df["image_filename"].isin(set(target_filenames))].copy()
        if df_use.empty:
            print(f"⚠️ No matching rows for degraded frames in PVS{i}, skipping.")
            continue

        target_count = len(target_filenames)
        print(f"Found {len(df_use)} labeled rows; target degraded MFCCs: {target_count}")

        # Generate exactly target_count degraded MFCCs
        _ = timeseries_tools.generate_images_fixed_count(
            df=df_use,
            data_column=data_column,
            method=method,
            output_folder=out_dir,
            target_count=target_count,
            window_size=window_size,
            target_filenames=target_filenames,
        )

        # Verification
        generated = [
            f for f in os.listdir(out_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"✅ Generated {len(generated)} degraded MFCCs (expected {target_count}).")
        if len(generated) != target_count:
            print("⚠️ Mismatch between degraded MFCC and frame counts!")

def sensor_backbones_degraded(train_1="mfcc_degraded"):
    """
    Train ConvNeXtBaseClassifier models on the degraded MFCC dataset (PVS1–PVS9).
    Saves models and TensorBoard logs under *_degraded directories.
    """
    # Base path for degraded dataset
    pvs_base_path = rf"/home/ws/ugoby/master_thesis/data_conference/{train_1}/"

    parameters = [
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS {i}") for i in range(1, 10)],
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

    print("\n🚀 Starting training on degraded MFCC dataset...")
    training_functions.legacy_train_multiple_runs(parameters)
    print("✅ Training complete for degraded MFCC dataset.\n")

experiments = [
    {
        "model_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc_degraded.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",  # same data format, just degraded version
        "image_folder": [
            os.path.join(
                "/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
                f"PVS{i}"
            )
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,
    },
]

def testing_sensor_backbones_degraded(
    experiments=None,
    output_dir="/home/ws/ugoby/master_thesis/test_conference/single_backbones/",
    num_classes=3,
):
    """
    Test the trained ConvNeXtBaseClassifier on the degraded MFCC dataset.
    """
    if experiments is None:
        experiments = [
            {
                "model_path": "/home/ws/ugoby/master_thesis/models_conference/single_backbones/mfcc_degraded.pt",
                "model_class": networks.ConvNeXtBaseClassifier,
                "dataset_type": "mfcc",
                "image_folder": [
                    os.path.join(
                        "/home/ws/ugoby/master_thesis/data_conference/mfcc_degraded",
                        f"PVS {i}"
                    )
                    for i in range(1, 10)
                ],
                "use_mixed_samples": False,
            },
        ]

    for exp in experiments:
        print(f"\n🔹 Testing degraded model {os.path.basename(exp['model_path'])} on {exp['dataset_type']} dataset")

        # Create DataLoader
        test_loader = get_loader(
            dataset_type=exp["dataset_type"],
            image_folder=exp["image_folder"],
            batch_size=32,
            resize=(224, 224),
            return_class_map=False,
            use_mixed_samples=exp.get("use_mixed_samples", False),
        )

        # Model info
        models_info = [(exp["model_path"], exp["model_class"])]

        # Run testing + evaluation
        testing_and_metrics.test_and_evaluate(
            test_loader=test_loader,
            models_info=models_info,
            output_dir=output_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_classes=num_classes,
            class_names=["asphalt_road", "dirt_road", "cobblestone_road"],
        )

    print("\n✅ Testing complete for degraded MFCC dataset.\n")

#### VIDEO ####
## DEGRADED ##  
NIGHTISH_AUG_100 = A.Compose([          #100%
    A.RandomGamma(gamma_limit=(150, 260), p=0.95),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.45, -0.10),
        contrast_limit=(-0.25, 0.40),
        p=0.95
    ),
    A.HueSaturationValue(
        hue_shift_limit=6,
        sat_shift_limit=(-40, -12),
        val_shift_limit=(-25, -8),
        p=0.90
    ),
    A.RandomToneCurve(scale=0.20, p=0.60),
    A.OneOf([
        A.ISONoise(color_shift=(0.01, 0.07), intensity=(0.12, 0.40), p=0.7),
        A.GaussNoise(var_limit=(8.0, 40.0), mean=0, per_channel=True, p=0.3),
    ], p=0.90),
    A.OneOf([
        A.MotionBlur(blur_limit=(5, 11), p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    ], p=0.50),
    A.ImageCompression(quality_lower=30, quality_upper=65, p=0.75),
    A.RandomSunFlare(
        num_flare_circles_lower=1,
        num_flare_circles_upper=2,
        src_radius=80,
        angle_lower=-15,
        angle_upper=15,
        flare_roi=(0.0, 0.0, 1.0, 0.6),
        p=0.10
    ),
])
NIGHTISH_AUG = A.Compose([          #70%
    A.RandomGamma(gamma_limit=(120, 210), p=0.85),   # was (150, 260), p=0.95
    A.RandomBrightnessContrast(
        brightness_limit=(-0.35, -0.07),             # was (-0.45, -0.10)
        contrast_limit=(-0.18, 0.28),                # was (-0.25, 0.40)
        p=0.85                                       # was 0.95
    ),
    A.HueSaturationValue(
        hue_shift_limit=4,                           # was 6
        sat_shift_limit=(-28, -8),                   # was (-40, -12)
        val_shift_limit=(-18, -6),                   # was (-25, -8)
        p=0.75                                       # was 0.90
    ),
    A.RandomToneCurve(scale=0.15, p=0.50),           # was 0.20, p=0.60
    A.OneOf([
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.08, 0.28), p=0.7), # reduced
        A.GaussNoise(var_limit=(6.0, 28.0), mean=0, per_channel=True, p=0.3),
    ], p=0.75),                                      # was 0.90
    A.OneOf([
        A.MotionBlur(blur_limit=(3, 8), p=0.6),      # was (5, 11)
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),    # was (3, 7)
    ], p=0.40),                                      # was 0.50
    A.ImageCompression(quality_lower=40, quality_upper=75, p=0.65),  # was 30–65, p=0.75
    A.RandomSunFlare(
        num_flare_circles_lower=1,
        num_flare_circles_upper=2,
        src_radius=70,                               # was 80
        angle_lower=-10,                             # was -15
        angle_upper=10,                              # was 15
        flare_roi=(0.0, 0.0, 1.0, 0.5),              # slightly reduced ROI
        p=0.07                                       # was 0.10
    ),
])
def apply_nightish_augmentation_to_pvs(
    base_dir="/home/ws/ugoby/master_thesis/data_conference",
    min_mean_v=40.0,
    target_mean_v=70.0,
    clear_output=True
):
    """
    Applies the NIGHTISH_AUG pipeline to all PVS 1–9 datasets located in the given base directory.
    Each frame from 'video/PVS *' is augmented and saved with the same name under 'video_degraded/PVS *'.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing 'video' and 'video_degraded' subfolders.
    min_mean_v : float
        Minimum mean brightness threshold for rescue correction.
    target_mean_v : float
        Target mean brightness value for dark images.
    clear_output : bool
        Whether to clear the destination folder before writing new images.
    """

    base_dir = Path(base_dir)
    video_dir = base_dir / "video"
    video_degraded_dir = base_dir / "video_degraded"
    video_degraded_dir.mkdir(parents=True, exist_ok=True)

    # Identify PVS folders
    pvs_folders = [f for f in sorted(video_dir.iterdir()) if f.is_dir() and f.name.startswith("PVS")]
    if not pvs_folders:
        print(f"No PVS folders found under {video_dir}")
        return

    print(f"Found {len(pvs_folders)} PVS datasets under {video_dir}\n")

    for pvs_folder in pvs_folders:
        dest_folder = video_degraded_dir / pvs_folder.name
        print(f"--- Processing {pvs_folder.name} ---")
        print(f"Input:  {pvs_folder}")
        print(f"Output: {dest_folder}")

        image_video_tools.transform_images_to_nighttime(
            input_folder=str(pvs_folder),
            output_folder=str(dest_folder),
            aug=NIGHTISH_AUG,
            min_mean_v=min_mean_v,
            target_mean_v=target_mean_v,
            clear_output=clear_output
        )

        print(f"✔ Completed {pvs_folder.name}\n")

    print("✅ All PVS datasets processed successfully.")





train_video = "video_degraded"  # or "raw" depending on your data loader namin

def video_backbones():
    """
    Train ConvNeXt backbones on raw video frames (PVS1 PVS9).
    """
    pvs_base_path = rf"/home/ws/ugoby/master_thesis/data_conference/{train_video}/"

    parameters = [
        {
            "data_type": "multi_source",
            "image_folders": [os.path.join(pvs_base_path, f"PVS {i}") for i in range(1, 10)],
            "model_class": lambda nc: networks.ConvNeXtBaseClassifier(nr_classes=nc),
            "batch_size": 32,
            "resize": (224, 224),
            "num_classes": 3,
            "learning_rate": 1e-4,
            "epochs": 50,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "writer": SummaryWriter(
                log_dir=f"/home/ws/ugoby/master_thesis/tensorboard_conference/runs/single_backbones/{train_video}"
            ),
            "model_save_path": rf"/home/ws/ugoby/master_thesis/models_conference/single_backbones/{train_video}.pt",
            "early_stopping_patience": 10,
            "min_delta": 0.001,
            "use_mixed_samples": False,
        },
    ]

    training_functions.legacy_train_multiple_runs(parameters)

experiments_video = [
    {
        "model_path": rf"/home/ws/ugoby/master_thesis/models_conference/single_backbones/{train_video}.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "raw",  # raw = single-frame images
        "image_folder": [
            os.path.join(
                rf"/home/ws/ugoby/master_thesis/data_conference/{train_video}",
                f"PVS {i}"
            )
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,
    },
]

def testing_video_backbones(
    experiments, output_dir="/home/ws/ugoby/master_thesis/test_conference/single_backbones/", num_classes=3
):
    """
    Evaluate trained video models on all PVS sets.
    """
    for exp in experiments:
        print(f"\n🎥 Testing video model {os.path.basename(exp['model_path'])}")

        test_loader = get_loader(
            dataset_type=exp["dataset_type"],
            image_folder=exp["image_folder"],
            batch_size=32,
            resize=(224, 224),
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
            class_names=["asphalt_road", "dirt_road", "cobblestone_road"],
        )


if __name__ == "__main__":

    #build_degraded_label_datasets()
    #generate_all_pvs_mfcc_degraded()
    #sensor_backbones_degraded()
    #testing_sensor_backbones_degraded()

    apply_nightish_augmentation_to_pvs()
    video_backbones()
    testing_video_backbones(experiments_video)  
