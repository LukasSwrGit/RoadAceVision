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





#train_2 = "degraded_2_pvs1_mfcc++"
#train_3 = "degraded_3_pvs1_mfcc++"
#train_4 = "degraded_4_pvs1_mfcc++"
#train_5 = "degraded_5_pvs1_mfcc++"
#train_6 = "degraded_6_pvs1_mfcc++"

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
## ORIGINAL ##  
train_1 = "mfcc"

def sensor_backbones():
    ############# Actually used right now #############
    #Original
    pvs_base_path = rf"/home/ws/ugoby/master_thesis/data_conference/{train_1}/"

    parameters = [
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


    training_functions.legacy_train_multiple_runs(parameters)

experiments = [
    {
        "model_path": rf"/home/ws/ugoby/master_thesis/models_conference/single_backbones/{train_1}.pt",
        "model_class": networks.ConvNeXtBaseClassifier,
        "dataset_type": "mfcc",
        "image_folder": [
            os.path.join(
                rf"/home/ws/ugoby/master_thesis/data_conference/{train_1}",
                f"PVS{i}"
            )
            for i in range(1, 10)
        ],
        "use_mixed_samples": False,
    },
]

def testing_sensor_backbones(experiments, output_dir="/home/ws/ugoby/master_thesis/test_conference/single_backbones/", num_classes=3):
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
    ## DEGRADED ##


#### VIDEO ####
## ORIGINAL ##  
train_video = "video"  # or "raw" depending on your data loader namin

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
    #generate_mfcc_images()
    #build_degraded_label_datasets()
    #generate_all_pvs_mfcc(degraded=True)

    #sensor_backbones()
    #testing_sensor_backbones(experiments)

    #multimodal()
    #testing_multimoda
    # l_models(experiments_multimodal)
    #multimodal_special()

    #video_backbones()
    testing_video_backbones(experiments_video)
    pass
