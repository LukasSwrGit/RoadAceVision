import torch
import os
import pandas as pd
from torch.utils.data import ConcatDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split   
from torchvision.datasets import MNIST
from torchvision import transforms

# Dummy loader

def mnist_loader():
    transform_pipeline = transforms.Compose([
        # transforms.Resize((28, 28)),         
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor()
    ])

    dataset = MNIST("data\pytorch_datasets", train=True, transform=transform_pipeline, download=True)

    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


    test_set = MNIST("data\pytorch_datasets", train=False)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

# Dataloader function

class RoadDataset(Dataset):

    def __init__(self, csv_file, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        df = pd.read_csv(csv_file)

        # Only keep actual image files present in folder
        image_files = set(os.listdir(image_folder))
        df = df[df['image_filename'].isin(image_files)]

        # Relevant one-hot label columns
        self.label_columns = ['dirt_road', 'cobblestone_road', 'asphalt_road']
        df = df[self.label_columns + ['image_filename']]

        # Group by filename, ensure only one hot encoded class per image
        df_grouped = df.groupby('image_filename')[self.label_columns].sum().reset_index()
        self.image_filenames = df_grouped['image_filename'].values
        self.labels = df_grouped[self.label_columns].values.argmax(axis=1)

        # Define class mapping (order must match label_columns)
        self.class_to_idx = {
            'dirt_road': 0,
            'cobblestone_road': 1,
            'asphalt_road': 2
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx]).long()

        if self.transform:
            image = self.transform(image)

        return image, label

def custom_loader(image_folder, csv_file, batch_size, resize):
    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    full_dataset = RoadDataset(csv_file=csv_file, image_folder=image_folder, transform=transform_pipeline)

    # Split sizes
    total_size = len(full_dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Ensure reproducibility
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ONE FOLDER Singlebackbone: Custom Dataset for all types of data

class NoiseImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, num_classes=3, use_mixed_samples=True):
        self.image_folder = image_folder
        self.transform = transform
        self.use_mixed_samples = use_mixed_samples

        # Default: 3 classes
        if num_classes == 3:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2
            }
        # Extended: 6 classes (day + night variants)
        elif num_classes == 6:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2,
                'asphalt_night': 3,
                'dirt_night': 4,
                'cobblestone_night': 5
            }
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

        self.image_filenames = []
        self.labels = []

        # ✅ Deterministic file listing
        for filename in sorted(os.listdir(image_folder)):
            if not filename.endswith('.jpg'):
                continue
            try:
                # Keep everything after the first underscore
                label_str = "_".join(filename.split("_")[1:]).replace('.jpg', '').strip()

                # Handle 'mixed_' prefix
                is_mixed = label_str.startswith('mixed_')
                base_label = label_str.replace('mixed_', '')

                # Skip mixed samples if not requested
                if is_mixed and not self.use_mixed_samples:
                    continue

                # Assign class index
                if base_label in self.class_to_idx:
                    self.image_filenames.append(filename)
                    self.labels.append(self.class_to_idx[base_label])

            except IndexError:
                continue

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx]).long()

        if self.transform:
            image = self.transform(image)

        return image, label

def general_image_loader(image_folder, batch_size, resize, return_class_map=False, num_classes=3):
    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    dataset = NoiseImageDataset(
        image_folder=image_folder,
        transform=transform_pipeline,
        num_classes=num_classes   # ✅ forward num_classes from parameters
    )

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if return_class_map:
        return train_loader, val_loader, test_loader, dataset.class_to_idx
    else:
        return train_loader, val_loader, test_loader
  
def cwt_loader(image_folder, batch_size, resize, return_class_map, num_classes=3):
    return general_image_loader(
        image_folder=image_folder,
        batch_size=batch_size,
        resize=resize,
        return_class_map=return_class_map,
        num_classes=num_classes
    )

def mfcc_loader(image_folder, batch_size, resize, return_class_map, num_classes=3):
    return general_image_loader(
        image_folder=image_folder,
        batch_size=batch_size,
        resize=resize,
        return_class_map=return_class_map,
        num_classes=num_classes
    )


# MULTI FOLDER Singlebackbone (LOAD FROM 2 OR MORE FOLDERS FOR TRAINING) DATALOADERS WITH MIXED SAMPLES SUPPORT
def general_image_loader_multi(image_folders, batch_size, resize, return_class_map=False,
                               
                            
                               num_classes=3, use_mixed_samples=False):
    """
    Load (image, label) from multiple folders (e.g., pvs1_video + pvs9_video OR pvs1_mfcc + pvs9_mfcc)
    and concatenate them into a single dataset for single-backbone training.
    """
    if isinstance(image_folders, (str, bytes)):
        image_folders = [image_folders]
    assert isinstance(image_folders, (list, tuple)) and len(image_folders) >= 2, \
        "Pass a list/tuple of at least two folders."

    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    # Build one dataset per folder and concatenate
    per_folder_datasets = [
        NoiseImageDataset(
            image_folder=folder,
            transform=transform_pipeline,
            num_classes=num_classes,
            use_mixed_samples=use_mixed_samples
        )
        for folder in image_folders
    ]

    combined_dataset = ConcatDataset(per_folder_datasets)

    # Split into train/val/test (75/15/10)
    total_size = len(combined_dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        combined_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if return_class_map:
        return train_loader, val_loader, test_loader, per_folder_datasets[0].class_to_idx
    else:
        return train_loader, val_loader, test_loader
    

def cwt_loader_multi(image_folders, batch_size, resize, return_class_map=False,
                     num_classes=3, use_mixed_samples=True):
    return general_image_loader_multi(
        image_folders=image_folders,
        batch_size=batch_size,
        resize=resize,
        return_class_map=return_class_map,
        num_classes=num_classes,
        use_mixed_samples=use_mixed_samples
    )


def mfcc_loader_multi(image_folders, batch_size, resize, return_class_map=False,
                      num_classes=3, use_mixed_samples=True):
    return general_image_loader_multi(
        image_folders=image_folders,
        batch_size=batch_size,
        resize=resize,
        return_class_map=return_class_map,
        num_classes=num_classes,
        use_mixed_samples=use_mixed_samples
    )




# MULTI FOLDER MULTI backbones (LOAD FROM 2 OR MORE FOLDERS FOR TRAINING) DATALOADERS WITH MIXED SAMPLES SUPPORT
class MultiPVSMultimodalDataset(Dataset):
    """
    Multimodal dataset for all PVS1–9 folders.
    Each sample = (image_modality, sensor_modality, label)
    """

    def __init__(self, image_root, sensor_root, transform=None):
        self.image_root = image_root
        self.sensor_root = sensor_root
        self.transform = transform

        # Fixed class mapping (3 classes only)
        self.class_to_idx = {
            'asphalt': 0,
            'dirt': 1,
            'cobblestone': 2
        }

        # Collect all (image_path, sensor_path, label)
        self.samples = []
        for i in range(1, 10):
            image_folder = os.path.join(image_root, f"PVS {i}")
            sensor_folder = os.path.join(sensor_root, f"PVS{i}")

            if not os.path.isdir(image_folder) or not os.path.isdir(sensor_folder):
                print(f"⚠️ Skipping PVS{i}: missing folder")
                continue

            image_files = set(os.listdir(image_folder))
            sensor_files = set(os.listdir(sensor_folder))
            common_files = sorted(image_files & sensor_files)

            for fname in common_files:
                if not fname.endswith(".jpg"):
                    continue

                fname_noext = fname[:-4]
                label_str = None
                for cls_name in self.class_to_idx.keys():
                    if fname_noext.endswith(cls_name):
                        label_str = cls_name
                        break

                if label_str is not None:
                    self.samples.append((
                        os.path.join(image_folder, fname),
                        os.path.join(sensor_folder, fname),
                        self.class_to_idx[label_str]
                    ))

        if not self.samples:
            raise RuntimeError("No valid multimodal samples found across PVS1–9!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, sensor_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        sensor = Image.open(sensor_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            sensor = self.transform(sensor)

        return image, sensor, torch.tensor(label).long()

def multi_pvs_multimodal_loader(image_root, sensor_root,
                                batch_size=32, resize=(224, 224),
                                return_class_map=False,
                                num_classes=None):
    """
    Loads multimodal (image + sensor) data from PVS1–9.
    Returns train/val/test loaders with 75/15/10 split.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    dataset = MultiPVSMultimodalDataset(
        image_root=image_root,
        sensor_root=sensor_root,
        transform=transform_pipeline
    )

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if return_class_map:
        return train_loader, val_loader, test_loader, dataset.class_to_idx
    else:
        return train_loader, val_loader, test_loader






###### Multimodal below #######

class FlexibleModalRoadDataset(Dataset):
    def __init__(self, folders, transform=None, num_classes=3):
        self.folders = folders
        self.transform = transform

        # Default: 3 classes
        if num_classes == 3:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2
            }
        # Extended: 6 classes (day + night variants)
        elif num_classes == 6:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2,
                'asphalt_night': 3,
                'dirt_night': 4,
                'cobblestone_night': 5
            }
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

        # ✅ Ensure deterministic file matching
        base_filenames = set(sorted(os.listdir(folders[0])))
        for folder in folders[1:]:
            base_filenames &= set(sorted(os.listdir(folder)))

        # Filter valid labeled files
        self.pairs = []
        for fname in sorted(base_filenames):  # ✅ deterministic order
            if fname.endswith('.jpg'):
                fname_noext = fname.replace('.jpg', '')

                # Find a matching class name at the end of the filename
                label_str = None
                for cls_name in self.class_to_idx.keys():
                    if fname_noext.endswith(cls_name):
                        label_str = cls_name
                        break

                if label_str is not None:
                    self.pairs.append((fname, self.class_to_idx[label_str]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fname, label = self.pairs[idx]
        images = []

        for folder in self.folders:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return *images, torch.tensor(label).long()

def multimodal_loader(folders, batch_size, resize, return_class_map=False, num_classes=3):
    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    dataset = FlexibleModalRoadDataset(folders=folders, transform=transform_pipeline, num_classes=num_classes)

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if return_class_map:
        return train_loader, val_loader, test_loader, dataset.class_to_idx
    else:
        return train_loader, val_loader, test_loader



##Special double input Multimodal Dataset
#4 Backbones input is duplicated
class SpecialMultimodalRoadDataset(Dataset):
    def __init__(self, folder_pairs, transform=None, num_classes=3, duplicate=True):
        """
        folder_pairs: list of tuples -> [(video_folder, mfcc_folder), ...]
        duplicate: if True -> return (video, video, mfcc, mfcc, label)
                   if False -> return (video, mfcc, label)
        """
        self.folder_pairs = folder_pairs
        self.transform = transform
        self.duplicate = duplicate   # <--- store flag

        # Class mapping
        if num_classes == 3:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2
            }
        elif num_classes == 6:
            self.class_to_idx = {
                'asphalt': 0,
                'dirt': 1,
                'cobblestone': 2,
                'asphalt_night': 3,
                'dirt_night': 4,
                'cobblestone_night': 5
            }
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

        # Collect (video, mfcc, label) samples
        self.samples = []
        for video_folder, mfcc_folder in folder_pairs:
            video_files = set(os.listdir(video_folder))
            mfcc_files = set(os.listdir(mfcc_folder))
            common_files = sorted(video_files & mfcc_files)

            for fname in common_files:
                if not fname.endswith('.jpg'):
                    continue

                fname_noext = fname.replace('.jpg', '')

                # find class label
                label_str = None
                for cls_name in self.class_to_idx.keys():
                    if fname_noext.endswith(cls_name):
                        label_str = cls_name
                        break

                if label_str is not None:
                    self.samples.append((
                        os.path.join(video_folder, fname),
                        os.path.join(mfcc_folder, fname),
                        self.class_to_idx[label_str]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, mfcc_path, label = self.samples[idx]

        video = Image.open(video_path).convert("RGB")
        mfcc  = Image.open(mfcc_path).convert("RGB")

        if self.transform:
            video = self.transform(video)
            mfcc  = self.transform(mfcc)

        if self.duplicate:
            # Duplicate each modality for the two backbones
            return video, video.clone(), mfcc, mfcc.clone(), torch.tensor(label).long()
        else:
            # Regular single modality outputs
            return video, mfcc, torch.tensor(label).long()



def special_multimodal_loader(
    folder_pairs,
    batch_size,
    resize,
    return_class_map=False,
    num_classes=3,
    duplicate=True,   # <--- add this
):
    transform_pipeline = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    dataset = SpecialMultimodalRoadDataset(
        folder_pairs=folder_pairs,
        transform=transform_pipeline,
        num_classes=num_classes,
        duplicate=duplicate   # <--- pass it here
    )

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if return_class_map:
        return train_loader, val_loader, test_loader, dataset.class_to_idx
    else:
        return train_loader, val_loader, test_loader



###### Kfold cross validation below #######

from typing import Tuple, List, Dict, Optional

def get_multimodal_dataset(
    folders: List[str],
    resize: Tuple[int, int] = (224, 224),
    num_classes: int = 3,
    return_class_map: bool = True,
    transform: Optional[transforms.Compose] = None,
):
    """
    Build a single unsplit multimodal dataset to feed K-fold splits.
    Reuses FlexibleModalRoadDataset with the same transforms used in multimodal_loader.
    Returns: (dataset, labels, class_to_idx) if return_class_map else (dataset, labels, None)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    ds = FlexibleModalRoadDataset(folders=folders, transform=transform, num_classes=num_classes)

    # Extract labels directly from the dataset's internal index
    # FlexibleModalRoadDataset stores list of (filename, label_idx) in ds.pairs
    labels = [lbl for _, lbl in ds.pairs]

    if return_class_map:
        return ds, labels, ds.class_to_idx
    return ds, labels, None


def get_special_multimodal_dataset(
    folder_pairs: List[Tuple[str, str]],
    resize: Tuple[int, int] = (224, 224),
    num_classes: int = 3,
    return_class_map: bool = True,
    duplicate: bool = True,
    transform: Optional[transforms.Compose] = None,
):
    """
    Build a single unsplit 'special' multimodal dataset (video+mfcc pairs) for K-fold.
    Reuses SpecialMultimodalRoadDataset with the same transforms used in special_multimodal_loader.
    Returns: (dataset, labels, class_to_idx) if return_class_map else (dataset, labels, None)

    Note: 'duplicate' must match your training/eval configuration:
      - duplicate=True  -> each sample yields (video, video, mfcc, mfcc, label)
      - duplicate=False -> each sample yields (video, mfcc, label)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    ds = SpecialMultimodalRoadDataset(
        folder_pairs=folder_pairs,
        transform=transform,
        num_classes=num_classes,
        duplicate=duplicate
    )

    # SpecialMultimodalRoadDataset stores list of (video_path, mfcc_path, label_idx) in ds.samples
    labels = [lbl for _, _, lbl in ds.samples]

    if return_class_map:
        return ds, labels, ds.class_to_idx
    return ds, labels, None
