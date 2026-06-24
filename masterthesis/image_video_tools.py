import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import math
import numpy as np
import albumentations as A
import random
import re
import shutil

from tqdm import tqdm  # for progress bar
from PIL import Image
from torchvision import transforms
from collections import Counter
from typing import List, Tuple
from pathlib import Path









# -*- coding: utf-8 -*-
"""
Two-folder night-ish augmentation pipeline.

Behavior:
- transform_images_to_nighttime(input_folder, output_folder, ...):
    * Reads all .jpg files from input_folder (flat, no subfolders).
    * Validates filenames end with one of the known classes: asphalt|cobblestone|dirt|mixed
    * Applies NIGHTISH_AUG + adaptive brightness rescue.
    * Writes augmented images to output_folder using the SAME FILENAME (no '_night').
    * By default, clears the output folder first and overwrites contents.

- augment_and_plot(input_folder, output_folder, ...):
    * Displays side-by-side comparisons (original from input_folder vs. augmented from output_folder).
    * Only shows pairs where both files exist and match the class-naming pattern.

Requirements:
    pip install albumentations opencv-python matplotlib numpy
"""

#from __future__ import annotations

import os
import re
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


# --- Night-ish augmentation pipeline (unchanged) ---
NIGHTISH_AUG = A.Compose([
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


# --- Helper: adaptive rescue for overly dark outcomes (unchanged) ---
def _rescue_if_too_dark(
    rgb: np.ndarray,
    min_mean_v: float,
    target_mean_v: float,
    max_gain: float,
    use_clahe: bool
) -> np.ndarray:
    """Lift overly dark results by scaling HSV-V and optional CLAHE."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    cur = float(v.mean())
    if cur < min_mean_v:
        gain = min(max_gain, target_mean_v / max(cur, 1.0))
        v = np.clip(v * gain, 0, 255)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v.astype(np.uint8)).astype(np.float32)
        hsv[..., 2] = v.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


# --- Function 1: Create and save nighttime versions to a separate output folder ---
def transform_images_to_nighttime(
    input_folder: str,
    output_folder: str,
    aug: A.BasicTransform | A.Compose = NIGHTISH_AUG,
    seed: Optional[int] = 123,
    min_mean_v: float = 60.0,
    target_mean_v: float = 85.0,
    max_gain: float = 1.7,
    use_clahe: bool = True,
    clear_output: bool = True,
) -> None:
    """
    For every .jpg image in input_folder, create a nighttime version using the given
    augmentation pipeline. The new image is saved in output_folder with the SAME
    filename (no '_night' suffix).

    Filenames must match: {prefix}_{class}.jpg
    Valid classes: asphalt, cobblestone, dirt, mixed

    Args:
        input_folder: Path to original (day) images.
        output_folder: Path where augmented (night-ish) images will be written.
        aug: Albumentations pipeline to apply.
        seed: Optional RNG seed for reproducibility.
        min_mean_v, target_mean_v, max_gain, use_clahe: parameters for brightness rescue.
        clear_output: If True, the output folder will be cleared before writing.

    Returns:
        None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.is_dir():
        raise ValueError(f"{in_dir} is not a valid directory")

    # Safety check: avoid nuking input if paths are the same
    if in_dir.resolve() == out_dir.resolve():
        raise ValueError("input_folder and output_folder must be different paths.")

    # Prepare output directory
    if clear_output and out_dir.exists():
        # Remove and recreate to guarantee a clean slate
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pattern: enforce known class at the end (before .jpg)
    pattern = re.compile(r"^(.*)_(asphalt|cobblestone|dirt|mixed)\.jpg$", re.IGNORECASE)

    jpg_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
    if not jpg_files:
        print(f"[info] No .jpg files found in {in_dir}")
        return

    processed = 0
    skipped = 0

    for path in jpg_files:
        fname = path.name
        if not pattern.match(fname):
            print(f"[warn] Skipping file without valid class name: {fname}")
            skipped += 1
            continue

        # Read
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[warn] Failed to read image: {fname}")
            skipped += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Augment
        out = aug(image=rgb)
        rgb_aug = out["image"]

        # Adaptive brightness rescue
        rgb_aug = _rescue_if_too_dark(
            rgb_aug,
            min_mean_v=min_mean_v,
            target_mean_v=target_mean_v,
            max_gain=max_gain,
            use_clahe=use_clahe,
        )

        # Write with SAME filename into output folder
        save_path = out_dir / fname
        bgr_aug = cv2.cvtColor(rgb_aug, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(save_path), bgr_aug)
        if not ok:
            print(f"[warn] OpenCV failed to write: {save_path.name}")
            skipped += 1
            continue

        processed += 1
        print(f"[ok] Saved: {save_path.name}")

    print(f"✅ Done! Processed: {processed} | Skipped: {skipped}")


# --- Function 2: Preview originals (Folder A) vs augmented (Folder B) ---
def augment_and_plot(
    input_folder: str,
    output_folder: str,
    n_cols: int = 5,
    show_titles: bool = True,
    figsize_per_col: float = 4.0,
    limit: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Plot original vs. augmented image pairs from two flat folders.

    - Left/top row: originals from input_folder
    - Right/bottom row: augmented counterparts from output_folder
    - Only pairs with matching filenames (and valid class naming) are shown.

    Args:
        input_folder: Folder with original images.
        output_folder: Folder with already-generated augmented images.
        n_cols: Number of columns per grid page.
        show_titles: Show filenames above originals.
        figsize_per_col: Figure width/height scale per column.
        limit: If provided, limit the number of pairs displayed.

    Returns:
        (originals_rgb_list, augmented_rgb_list)
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.is_dir():
        raise ValueError(f"{in_dir} is not a valid directory")
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a valid directory")

    pattern = re.compile(r"^(.*)_(asphalt|cobblestone|dirt|mixed)\.jpg$", re.IGNORECASE)

    # Collect candidate filenames existing in both folders
    in_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
    pairs = []
    for p in in_files:
        fname = p.name
        if not pattern.match(fname):
            continue
        q = out_dir / fname
        if q.is_file():
            pairs.append((p, q))

    if not pairs:
        raise ValueError("No matching original/augmented .jpg pairs found between the two folders.")

    if limit is not None:
        pairs = pairs[:max(0, int(limit))]

    originals, augmented = [], []
    display_names: List[str] = []

    for p_in, p_out in pairs:
        bgr_in = cv2.imread(str(p_in), cv2.IMREAD_COLOR)
        bgr_out = cv2.imread(str(p_out), cv2.IMREAD_COLOR)
        if bgr_in is None or bgr_out is None:
            print(f"[warn] Failed to read pair: {p_in.name}")
            continue
        rgb_in = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
        rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

        originals.append(rgb_in)
        augmented.append(rgb_out)
        display_names.append(p_in.name)

    if len(originals) == 0:
        raise ValueError("Failed to load any matching pairs for plotting.")

    N = len(originals)
    n_cols = max(1, n_cols)
    n_rows = 2

    # Paginate in blocks of n_cols
    for start in range(0, N, n_cols):
        end = min(start + n_cols, N)
        cols = end - start

        fig = plt.figure(figsize=(figsize_per_col * cols, figsize_per_col * n_rows))
        for i in range(cols):
            idx = start + i

            ax1 = plt.subplot(n_rows, cols, i + 1)
            ax1.imshow(originals[idx]); ax1.axis("off")
            if show_titles:
                ax1.set_title(display_names[idx])

            ax2 = plt.subplot(n_rows, cols, cols + i + 1)
            ax2.imshow(augmented[idx]); ax2.axis("off")

        plt.tight_layout()
        plt.show()

    return originals, augmented


# # --- Example usage ---
# if __name__ == "__main__":
#     input_dir = "path/to/day_images"
#     output_dir = "path/to/night_images"
#
#     # 1) Generate augmented images into a separate folder (clears output by default)
#     transform_images_to_nighttime(input_dir, output_dir, clear_output=True)
#
#     # 2) Visual check: originals vs. augmented counterparts
#     augment_and_plot(input_dir, output_dir, n_cols=5, figsize_per_col=3.5)



def synchronize_images_to_dataset(dataset: pd.DataFrame, image_folder: str, output_csv_path: str) -> pd.DataFrame:
    """
    Synchronizes image filenames with dataset rows by assuming the first image corresponds to the first row and 
    the last image to the last row. Adds a new column 'image_filename' to the dataset.
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset to which image filenames will be added.
    - image_folder (str): Path to the folder containing images.
    - output_csv_path (str): Path to save the updated CSV.
    
    Returns:
    - pd.DataFrame: The updated dataset with an 'image_filename' column.
    """
    # List image files and sort them
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    
    if len(image_files) != len(dataset):
        print(f"Warning: Number of images ({len(image_files)}) does not match number of dataset rows ({len(dataset)}). Mapping will be proportional.")
    
    # Generate image filenames for each row
    indices = (dataset.index / (len(dataset) - 1) * (len(image_files) - 1)).round().astype(int)
    dataset = dataset.copy()
    dataset['image_filename'] = [image_files[i] for i in indices]

    dataset.to_csv(output_csv_path, index=False)
    
    return dataset

## Image grid still has problems with loading enough images from each class... ###
def show_image_grid(dataloader):
    class_names = ['dirt_road', 'cobblestone_road', 'asphalt_road']
    class_images = {0: [], 1: [], 2: []}
    filenames_per_class = {0: [], 1: [], 2: []}

    # Total target per class
    target_count = 3
    found_all = lambda: all(len(class_images[i]) == target_count for i in range(3))


    # Get reference to dataset filenames and indices
    try:
        base_dataset = dataloader.dataset.dataset
        indices = dataloader.dataset.indices
        image_filenames = base_dataset.image_filenames
    except Exception as e:
        raise ValueError("Ensure the dataset is a Subset of a custom Dataset with 'image_filenames' attribute.") from e

    sample_idx = 0

    for images, labels in dataloader:
        batch_size = images.size(0)
        for i in range(batch_size):
            label = labels[i].item()
            if len(class_images[label]) < target_count:
                img = images[i].permute(1, 2, 0).numpy()
                class_images[label].append(img)
                filenames_per_class[label].append(image_filenames[indices[sample_idx]])
            sample_idx += 1

            if found_all():
                break
        if found_all():
            break

    if not found_all():
        raise ValueError("Not enough examples for each class in this DataLoader. Try using the train_loader.")

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row in range(3):
        for col in range(3):
            axes[row, col].imshow(class_images[row][col])
            axes[row, col].set_title(f"{class_names[row]}\n{filenames_per_class[row][col]}")
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def get_video_information():
    video_path = r'D:\Uni\Coding\PVS 1\video_environment.mp4'
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resolution: {width}x{height}")

    cap.release()

def extract_frames_from_video(
    video_path,
    output_folder='extracted_frames',
    frame_rate=24,
    img_format='.jpg',
    prefix='frame',
    start_time=None,
    end_time=None,
    resize=None,
    crop_x_percent=None,
    crop_y_percent=None
):
    """
    Extract frames from a video at a specified frame rate.
    
    Set frame_rate='all' to extract **all** frames.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if start_time is None:
        start_time = 0
    if end_time is None or end_time > duration:
        end_time = duration

    current_frame = 0
    saved_frames = 0

    # Determine frame interval

    if frame_rate == 'all':
        frame_interval = 1
    else:
        frame_interval = max(1, int(fps / frame_rate))

    # Prepare progress bar

    total_to_process = int((end_time - start_time) * fps)
    pbar = tqdm(total=total_to_process, desc="Extracting Frames", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = current_frame / fps

        if current_time < start_time:
            current_frame += 1
            continue
        if current_time > end_time:
            break

        if current_frame % frame_interval == 0:
            # Crop if percentages are given

            if crop_x_percent and crop_y_percent:
                h, w, _ = frame.shape
                x_start = int(w * crop_x_percent[0])
                x_end = int(w * crop_x_percent[1])
                y_start = int(h * crop_y_percent[0])
                y_end = int(h * crop_y_percent[1])
                frame = frame[y_start:y_end, x_start:x_end]

            # Resize if needed
            
            if resize:
                frame = cv2.resize(frame, resize)

            filename = os.path.join(
                output_folder, f"{prefix}_{int(current_time):05d}{img_format}"
            )
            cv2.imwrite(filename, frame)
            saved_frames += 1

        current_frame += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Saved {saved_frames} frames to '{output_folder}'.")

def extract_frames_v2(video_path, output_folder, img_format='.jpg', prefix='frame', crop_x_percent=None, crop_y_percent=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop if percentages are given
        if crop_x_percent and crop_y_percent:
            h, w, _ = frame.shape
            x_start = int(w * crop_x_percent[0])
            x_end = int(w * crop_x_percent[1])
            y_start = int(h * crop_y_percent[0])
            y_end = int(h * crop_y_percent[1])
            frame = frame[y_start:y_end, x_start:x_end]

        frame_filename = f"{prefix}_{frame_number:05d}{img_format}"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"Extracted {frame_number}/{total_frames} frames")

    cap.release()
    print(f"Done! Extracted {frame_number} frames in total.")

## This functino is used to manually generate labels for images in a folder
def generate_static_csv_from_images(image_folder_path, output_csv_path):
    # Static row 
    static_row = [1,0,0,0,1,1,0,0,0,1,0,0,1,0]
    
    # Header
    header = [
        "paved_road", "unpaved_road", "dirt_road", "cobblestone_road", "asphalt_road",
        "no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone",
        "good_road_left", "regular_road_left", "bad_road_left",
        "good_road_right", "regular_road_right", "bad_road_right", "image_filename"
    ]

    # Get all .jpg image filenames sorted
    image_filenames = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith(".jpg")])

    # Write to CSV
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for filename in image_filenames:
            writer.writerow(static_row + [filename])


def load_and_preprocess_image(image_path, device, show=True):   ##loads 1 image from path and shows it for testing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    
    if show:
        image.show(title="Loaded Image")  # shows the image in default viewer

    tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    return tensor

def count_classes_in_folder(folder_path, image_format=".jpg"):
    """
    Count the number of images per class in a folder based on filenames like 00001_asphalt.jpg.

    Parameters:
        folder_path (str): Path to the folder containing images
        image_format (str): Image file extension (default: '.jpg')
    """
    filenames = [f for f in os.listdir(folder_path) if f.endswith(image_format)]
    
    labels = []
    for f in filenames:
        try:
            label = f.split("_")[-1].replace(image_format, "")
            labels.append(label)
        except IndexError:
            print(f"Skipping unreadable filename: {f}")
    
    class_counts = Counter(labels)

    print(f"\n📂 Folder: {folder_path}")
    print(f"🔢 Total: {len(labels)} images\n")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count}")

    return class_counts

video_path = r"M:\Coding\Master\masterthesis\data\data_base\downloads\PVS 9\video_environment.mp4"
output_folder = r"M:\Coding\Master\masterthesis\data\data_base\pvs9_video"

manual_label_path = r"D:\Uni\Coding\PVS 1\manual_labelling"
output_csv_path = r"data\manual_labelling\folder_labels.csv"

#### extracted frames v2 for extracting all frames, extract frames from video is outdated, since it can maximum extract 1 frame per second ####

#generate_static_csv_from_images(manual_label_path, output_csv_path)
#extract_frames_v2(video_path, output_folder, crop_x_percent=(0.25, 0.75),crop_y_percent=(0.3, 0.66))
#extract_frames_from_video(video_path,output_folder, frame_rate="all", resize=(1280, 720),crop_x_percent=(0.25, 0.75),crop_y_percent=(0.3, 0.66))
#synchronize_images_to_dataset(dataset = pd.read_csv(r"data\time_synchronized_data\labels\X01_label.csv"), image_folder = r"data\time_synchronized_data\raw_video", output_csv_path = "X01_label_new.csv")
