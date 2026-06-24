# RoadAceVision: Deep Multi-Modal Learning for Road Surface Classification

<p align="center">
  <img src="assets/Car%20Wallpaper.png" alt="RoadAceVision overview image" width="800">
</p>

RoadAceVision contains the implementation code for a master thesis on deep multimodal road surface classification using the Passive Vehicular Sensors (PVS) dataset. The project compares image-based, accelerometer-based, and multimodal fusion approaches for classifying road surfaces under normal and adverse input conditions.

The repository is intended as a research codebase. It contains the data preparation utilities, dataset loaders, model definitions, training routines, evaluation scripts, and experiment scripts used for the thesis. The current version still contains several hard-coded absolute paths from the original research environment. These paths must be adapted before running the code on another machine. A future version should replace these paths with a centralized configuration file or command-line interface.

---

## Quick start

The repository currently has the following expected layout:

```text
RoadAceVision/
├── environment.yml
├── .gitignore
├── masterthesis/
│   ├── data_loaders.py
│   ├── image_video_tools.py
│   ├── timeseries_tools.py
│   ├── networks.py
│   ├── training_functions.py
│   ├── testing_and_metrics.py
│   ├── testing_and_metrics_multimodal.py
│   ├── large_training.py
│   ├── large_training_single_modality.py
│   ├── large_training_single_modality_degraded.py
│   ├── large_training_multimodal.py
│   ├── k_fold.py
│   └── notebooks and legacy files
```

Clone the repository and enter the project:

```bash
git clone https://github.com/LukasSwrGit/RoadAceVision.git
cd RoadAceVision
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate IEEE_Detroit
```

Enter the code directory:

```bash
cd masterthesis
```

Verify that the basic imports work:

```bash
python - <<'PY'
import torch
import networks
import data_loaders
import training_functions

print("CUDA available:", torch.cuda.is_available())
print("Imports completed successfully.")
PY
```

Download the PVS dataset from Kaggle:

```text
https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
```

Using the Kaggle CLI, the download can be done with:

```bash
kaggle datasets download -d jefmenegazzo/pvs-passive-vehicular-sensors-datasets
unzip pvs-passive-vehicular-sensors-datasets.zip -d data/raw_pvs
```

The original development environment used a structure similar to:

```text
RoadAceVision/
├── data/
├── data_conference/
├── models/
├── models_conference/
├── tensorboard/
├── tensorboard_conference/
├── test_results/
├── test_conference/
└── masterthesis/
```

Create these folders manually if needed:

```bash
mkdir -p data data_conference models models_conference tensorboard tensorboard_conference test_results test_conference
```

Before running any experiment, update hard-coded paths in the scripts. Many scripts currently refer to paths such as:

```python
/home/ws/ugoby/master_thesis/data_conference/
/home/ws/ugoby/master_thesis/models_conference/
/home/ws/ugoby/master_thesis/data/
/home/ws/ugoby/master_thesis/models/
```

Replace these with the corresponding paths on your local machine.

A minimal workflow is:

```text
1. Download the raw PVS dataset.
2. Extract and crop video frames.
3. Synchronize frames with sensor labels.
4. Generate MFCC images from accelerometer time series.
5. Train single-modality video and sensor backbones.
6. Train multimodal fusion models using the trained backbones.
7. Evaluate models and generate metrics/plots.
```

This repository does not currently provide a single end-to-end command-line entry point. Most workflows are function-based and are started from the experiment scripts or notebooks.

---

## Project purpose

The thesis studies road surface classification (RSC) using three classes:

```text
asphalt
dirt
cobblestone
```

The central research goal is to compare how different input modalities behave under normal and adverse conditions:

```text
video-only classification
accelerometer/spectrogram-only classification
multimodal video + accelerometer fusion
```

The thesis focuses on robustness. Video models can perform well under normal illumination but are vulnerable to low-visibility scenarios. Sensor-based models are independent of illumination but can suffer under noisy or degraded sensor conditions. Multimodal fusion is used to combine complementary information from both modalities.

---

## Dataset

The project uses the Passive Vehicular Sensors dataset:

```text
PVS Passive Vehicular Sensors Dataset
https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
```

The raw dataset contains several PVS recordings. In the original thesis work, the central thesis experiments mainly used:

```text
PVS 1  -> daytime setting
PVS 9  -> evening / reduced-light setting
```

For the expanded conference-style scripts, several functions also support processing PVS1-PVS9.

A raw PVS folder contains files such as:

```text
dataset_gps.csv
dataset_gps_mpu_left.csv
dataset_gps_mpu_right.csv
dataset_labels.csv
dataset_mpu_left.csv
dataset_mpu_right.csv
dataset_settings_left.csv
dataset_settings_right.csv
video_dataset_left.mp4
video_dataset_right.mp4
video_environment.mp4
video_environment_dataset_left.mp4
video_environment_dataset_right.mp4
map.html
```

The original PVS data contains labelled sensor recordings. The video frames are not directly labelled and must be synchronized with the sensor labels.

---

## Processed data and checkpoints

Processed datasets and trained model checkpoints are not included in this repository.

This means the following folders may be missing or empty after cloning:

```text
data/
data_conference/
models/
models_conference/
tensorboard/
tensorboard_conference/
test_results/
test_conference/
```

You need to generate the processed data locally before training. You also need to train the single-modality backbones before training fusion models that depend on those checkpoints.

The multimodal training scripts expect pretrained backbone checkpoints, for example:

```text
models/thesis/pvs1_video_v2.pt
models/thesis/pvs1_mfcc.pt
models/thesis/pvs9_video.pt
models/thesis/pvs9_mfcc_v2.pt
models_conference/single_backbones/video.pt
models_conference/single_backbones/mfcc.pt
```

These files are created by the single-modality training workflows and are not part of the public repository.

---

## Main experimental pipeline

### 1. Video preparation

Video data is converted into frame images using utilities in:

```text
masterthesis/image_video_tools.py
```

The key function is:

```python
extract_frames_from_video(
    video_path,
    output_folder="extracted_frames",
    frame_rate=24,
    img_format=".jpg",
    prefix="frame",
    start_time=None,
    end_time=None,
    resize=None,
    crop_x_percent=None,
    crop_y_percent=None,
)
```

The thesis preprocessing used a centered road-surface region of interest to reduce irrelevant background information. The relevant crop was approximately:

```python
crop_x_percent = (0.25, 0.75)
crop_y_percent = (0.30, 0.66)
```

Example:

```python
from image_video_tools import extract_frames_from_video

extract_frames_from_video(
    video_path="../data/raw_pvs/PVS 1/video_dataset_left.mp4",
    output_folder="../data/pvs1_video",
    frame_rate="all",
    prefix="pvs1",
    crop_x_percent=(0.25, 0.75),
    crop_y_percent=(0.30, 0.66),
)
```

The exact source video may need to be adjusted depending on whether left, right, or environment videos are used.

---

### 2. Label synchronization

The raw video frames are synchronized with sensor labels using moments of interest such as:

```text
surface transitions
speed bumps
start/end alignment points
```

The codebase contains utilities for assigning image filenames to synchronized sensor rows, for example:

```python
synchronize_images_to_dataset(
    dataset,
    image_folder,
    output_csv_path,
)
```

This function assumes that the first image corresponds to the first relevant row and the last image corresponds to the last relevant row. If the video and sensor streams do not cover exactly the same time interval, manual trimming or alignment is required before using this mapping.

---

### 3. Sensor preprocessing

Sensor-related preprocessing is implemented mainly in:

```text
masterthesis/timeseries_tools.py
masterthesis/data_exploration.py
```

The final thesis experiments use accelerometer data, especially the longitudinal y-axis. The three left-side accelerometer locations are combined into a single signal, commonly represented as:

```text
acc_y_combined
```

The original thesis pipeline uses the left-side sensor stream to reduce cross-side delay effects.

---

### 4. Time-series to image conversion

Accelerometer time series are converted into image-like spectrogram representations. The main representation used in the final experiments is MFCC.

Relevant functions:

```python
generate_mfcc_images(...)
generate_cwt_images(...)
generate_images_fixed_count(...)
```

The most important function for aligned experiments is:

```python
generate_images_fixed_count(
    df,
    data_column,
    method,
    output_folder,
    target_count,
    window_size=50,
    target_filenames=None,
)
```

Example:

```python
import pandas as pd
from timeseries_tools import generate_images_fixed_count

df = pd.read_csv("../data/labels/pvs1_synchronized.csv")

target_filenames = sorted([
    # List of video frame filenames with labels, e.g. 00001_asphalt.jpg
])

generate_images_fixed_count(
    df=df,
    data_column="acc_y_combined",
    method="mfcc",
    output_folder="../data/pvs1_mfcc",
    target_count=len(target_filenames),
    window_size=50,
    target_filenames=target_filenames,
)
```

The purpose of `target_filenames` is to ensure that generated spectrograms and video frames share matching filenames. This is important for multimodal training.

---

### 5. Adverse-condition generation

The thesis creates synthetic adverse-condition datasets for both video and sensor inputs.

#### Video degradation / nighttime generation

Nighttime-like image augmentation is implemented in:

```text
masterthesis/image_video_tools.py
```

Main function:

```python
transform_images_to_nighttime(
    input_folder,
    output_folder,
    seed=123,
    clear_output=True,
)
```

Example:

```python
from image_video_tools import transform_images_to_nighttime

transform_images_to_nighttime(
    input_folder="../data/pvs9_video",
    output_folder="../data/pvs9_video_night",
    seed=123,
    clear_output=True,
)
```

The transformation pipeline uses operations such as gamma correction, brightness/contrast changes, HSV shifts, tone curves, noise, blur, JPEG compression, and synthetic flares.

#### Sensor degradation

Sensor degradation is implemented in:

```text
masterthesis/timeseries_tools.py
```

Main function:

```python
degrade_signal_chain(...)
```

The degradation chain can apply:

```text
time warping
aliasing through downsampling and restoration
random spikes
hard clipping
noise injection
```

Example:

```python
from timeseries_tools import degrade_signal_chain

degraded_signal = degrade_signal_chain(
    y=signal_array,
    sample_rate_hz=100.0,
    downsample_factor=4,
    time_warp_enabled=True,
    time_warp_strength=0.10,
    time_warp_knots=4,
)
```

Higher-level degraded dataset generation is also used in:

```text
large_training_single_modality_degraded.py
```

---

## Data loaders

Dataset and DataLoader utilities are implemented in:

```text
masterthesis/data_loaders.py
```

The codebase supports several dataset types:

```text
raw image/video frames
MFCC spectrogram images
CWT spectrogram images
multi-source datasets across several folders
multimodal paired video + sensor image datasets
```

Important loaders include:

```python
general_image_loader(...)
general_image_loader_multi(...)
mfcc_loader(...)
mfcc_loader_multi(...)
cwt_loader(...)
cwt_loader_multi(...)
multimodal_loader(...)
multi_pvs_multimodal_loader(...)
```

Most loaders use a 75/15/10 split:

```text
75% train
15% validation
10% test
```

The split uses a fixed random seed of 42 for reproducibility.

The expected class labels are usually parsed from filenames. Common filename examples are:

```text
00001_asphalt.jpg
00002_dirt.jpg
00003_cobblestone.jpg
00004_mixed_asphalt.jpg
```

Mixed samples can be included or excluded depending on the `use_mixed_samples` argument.

---

## Models

Model definitions are implemented in:

```text
masterthesis/networks.py
```

The code includes classifier wrappers for several pretrained vision backbones:

```text
ResNet18
ResNet34
ResNet50
ConvNeXt-Tiny
ConvNeXt-Small
ConvNeXt-Base
Vision Transformer variants
EfficientViT-related code
```

The primary backbone used in the final thesis experiments is:

```python
ConvNeXtBaseClassifier
```

Example:

```python
import networks

model = networks.ConvNeXtBaseClassifier(nr_classes=3)
```

Most model wrappers replace the original classification head with a new linear layer matching the number of road-surface classes.

---

## Multimodal fusion architecture

The multimodal model is defined in:

```text
masterthesis/networks.py
```

The central learned fusion model is:

```python
FlexibleFusionClassifier
```

It combines multiple pretrained backbones and applies MMTM-style cross-modal recalibration.

Core components:

```python
MMTM
FlexibleFusionClassifier
build_backbones
```

The MMTM module receives feature vectors from two modalities, computes cross-modal gates, and rescales the feature vectors before classification.

The typical fusion setup combines:

```text
video backbone
MFCC/sensor backbone
fusion classifier
```

A simplified usage pattern is:

```python
import torch
import networks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone_configs = [
    {
        "network": networks.ConvNeXtBaseClassifier,
        "path": "../models/thesis/pvs1_video_v2.pt",
        "nr_classes": 3,
    },
    {
        "network": networks.ConvNeXtBaseClassifier,
        "path": "../models/thesis/pvs1_mfcc.pt",
        "nr_classes": 3,
    },
]

backbones, feature_dims = networks.build_backbones(backbone_configs, device)

fusion_model = networks.FlexibleFusionClassifier(
    backbones=backbones,
    feature_dims=feature_dims,
    nr_classes=3,
)
```

The exact checkpoint paths must be adapted to your local folder structure.

---

## Training

Training logic is implemented mainly in:

```text
masterthesis/training_functions.py
```

The codebase supports:

```text
single-modality training
multi-source training
multimodal training
K-fold training
early stopping
TensorBoard logging
class-balanced loss
mixed precision training
```

Important training parameters used in the thesis include:

```text
backbone: ConvNeXt-Base
image size: 224 x 224
batch size: 32
learning rate: 1e-4
epochs: up to 50
early stopping patience: 10
minimum delta: 0.001
number of classes: 3
```

Single-modality training is orchestrated mainly through:

```text
large_training_single_modality.py
large_training_single_modality_degraded.py
```

Multimodal training is orchestrated mainly through:

```text
large_training.py
large_training_multimodal.py
k_fold.py
```

Because the current version is a research codebase, these files are not clean command-line entry points. They contain experiment definitions and functions that can be imported or executed after paths are adapted.

Example pattern for single-modality sensor training:

```python
from large_training_single_modality import sensor_backbones

sensor_backbones()
```

Example pattern for degraded sensor training:

```python
from large_training_single_modality_degraded import sensor_backbones_degraded

sensor_backbones_degraded()
```

Example pattern for multimodal experiments:

```python
from large_training import testing_multimodal_models, experiments_multimodal

testing_multimodal_models(experiments_multimodal)
```

Before running these examples, check and update all path variables inside the corresponding script.

---

## Evaluation

Evaluation logic is implemented in:

```text
masterthesis/testing_and_metrics.py
masterthesis/testing_and_metrics_multimodal.py
```

The evaluation scripts compute:

```text
accuracy
weighted precision
weighted recall
weighted F1-score
per-class precision
per-class recall
per-class F1-score
confusion matrix
multiclass AUROC
ROC curves
metric bar plots
```

Single-modality evaluation uses:

```python
test_and_evaluate(...)
```

Multimodal evaluation uses:

```python
test_and_evaluate_multimodal(...)
```

The multimodal evaluation includes additional safeguards for AUROC computation, including conversion to float64, replacement of non-finite probability rows, and row-wise probability normalization.

Example single-modality evaluation:

```python
import torch
import networks
from testing_and_metrics import test_and_evaluate

test_and_evaluate(
    test_loader=test_loader,
    models_info=[
        ("../models/thesis/pvs1_video_v2.pt", networks.ConvNeXtBaseClassifier),
    ],
    output_dir="../test_results/pvs1_video",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_classes=3,
    class_names=["dirt_road", "cobblestone_road", "asphalt_road"],
)
```

Example multimodal evaluation:

```python
from testing_and_metrics_multimodal import test_and_evaluate_multimodal
import networks

test_and_evaluate_multimodal(
    test_loader=test_loader,
    fusion_checkpoint="../models/thesis/MM_pvs1_video_pvs1_mfcc.pt",
    backbone_configs=[
        {
            "network": networks.ConvNeXtBaseClassifier,
            "path": "../models/thesis/pvs1_video_v2.pt",
            "nr_classes": 3,
        },
        {
            "network": networks.ConvNeXtBaseClassifier,
            "path": "../models/thesis/pvs1_mfcc.pt",
            "nr_classes": 3,
        },
    ],
    output_dir="../test_results/multimodal",
    num_classes=3,
    class_names=["dirt_road", "cobblestone_road", "asphalt_road"],
)
```

---

## TensorBoard

Training scripts write TensorBoard logs to folders such as:

```text
tensorboard/
tensorboard_conference/
```

Start TensorBoard from the repository root:

```bash
tensorboard --logdir tensorboard
```

or for the conference-style experiments:

```bash
tensorboard --logdir tensorboard_conference
```

Then open the displayed local URL in a browser.

---

## Important files

### `data_loaders.py`

Contains dataset classes and loader functions for image, MFCC, CWT, multi-source, and multimodal datasets. It defines the logic for reading filenames, parsing labels, applying torchvision transforms, and splitting into train/validation/test sets.

### `image_video_tools.py`

Contains video and image utilities, including frame extraction, filename synchronization, image grid visualization, and nighttime-like image augmentation.

### `timeseries_tools.py`

Contains sensor preprocessing, MFCC/CWT image generation, signal degradation, and utilities for aligning generated sensor images with video frame counts.

### `networks.py`

Contains model definitions. It includes pretrained ResNet, ConvNeXt, ViT-style classifiers, the MMTM module, and the flexible fusion classifier.

### `training_functions.py`

Contains the main training and validation routines, including early stopping, TensorBoard logging, multiple-run training, K-fold routines, and multimodal training utilities.

### `testing_and_metrics.py`

Contains evaluation routines for single-modality models.

### `testing_and_metrics_multimodal.py`

Contains evaluation routines for multimodal fusion models.

### `large_training.py`

Contains thesis-style multimodal experiment definitions and evaluation configurations.

### `large_training_single_modality.py`

Contains single-modality training and evaluation workflows for normal sensor/video inputs.

### `large_training_single_modality_degraded.py`

Contains workflows for degraded sensor data generation, training, and evaluation.

### `large_training_multimodal.py`

Contains multimodal experiment definitions, especially for broader PVS1-PVS9 conference-style experiments.

### `k_fold.py`

Contains K-fold experiment logic for more robust model evaluation.

### Notebooks

The notebooks are exploratory and workflow-oriented. They are useful for reproducing parts of the preparation and analysis process but should not be treated as the primary stable API.

Common notebooks include:

```text
complete_pvs_data_prep.ipynb
dataset_exploration.ipynb
prepare_model_and_dataloader.ipynb
thesis_graphs.ipynb
test.ipynb
```

---

## Legacy and backup files

The repository contains legacy or backup code from earlier development stages. These files are kept for transparency and reproducibility but are not the recommended starting point for new experiments.

Legacy/back-up items include:

```text
training_functions_17_10.py
code_backup.py
legacy_* functions inside training_functions.py
older experiment text files such as 15_10_MM_training.txt and 19_10_KFold_on_exp_6.txt
```

Prefer the current main files unless you specifically need to inspect previous experiment versions.

---

## Reproducibility notes

The project uses several reproducibility mechanisms:

```text
fixed random seed 42 for dataset splitting
documented train/validation/test split
saved model checkpoints
TensorBoard logs
explicit experiment dictionaries in training scripts
```

However, exact reproduction requires:

```text
the same PVS data version
the same processed frame/spectrogram generation steps
the same local paths or path configuration
the same environment from environment.yml
the same GPU/software stack where possible
the same trained checkpoints
```

Small differences in preprocessing, video frame extraction, or filename alignment can change the final data splits and therefore the results.

---

## Known limitations of the current repository

This repository is currently a research code release, not a polished Python package.

Known limitations:

```text
several scripts contain hard-coded absolute paths
processed datasets are not included
trained checkpoints are not included
some workflows require manual preprocessing before training
some scripts are function-based rather than command-line based
notebooks contain exploratory code and may require path adaptation
legacy files are still present
```

The most important first step before running the project is to update the local paths and regenerate the processed data.

---

## Recommended future improvements

The following improvements would make the repository easier to use:

```text
central config file for all paths
CLI entry points for data preparation, training, and evaluation
separate requirements file for pip users
documentation of exact processed folder structure
small sample dataset for smoke tests
automatic checkpoint download or clear checkpoint registry
removal or archiving of legacy files
unit tests for dataset pairing and label parsing
```

---

## Citation

This repository accompanies the master thesis:

```text
RoadAceVision: Deep Multi-Modal Learning for Road Surface Classification
Lukas Michael Schwemer
Master Thesis, Karlsruhe Institute of Technology (KIT), 2025
```

Dataset:

```text
PVS Passive Vehicular Sensors Dataset
https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
```
