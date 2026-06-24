from tqdm import tqdm as std_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import torch
import copy
import torch
import data_loaders
import training_functions
import numpy as np
import itertools
import random
import os
import torch.nn as nn

## Legacy code for non multimodal training

def legacy_train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_notebook_tqdm=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    tqdm_fn = notebook_tqdm if use_notebook_tqdm else std_tqdm
    loop = tqdm_fn(dataloader, desc="Training", leave=True)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # AMP guard: only enable on CUDA when scaler is present
        use_amp = (scaler is not None) and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def legacy_validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def legacy_train_model(model, train_loader, val_loader, optimizer, num_epochs, device, save_path, writer,
                       early_stopping_patience=5, min_delta=0.001):
    # Prepare writer starting values
    if writer:
        writer.add_scalar('Loss/train', 2, 0)
        writer.add_scalar('Loss/validation', 2, 0)
        writer.add_scalar('Accuracy/train', 0, 0)
        writer.add_scalar('Accuracy/validation', 0, 0)

    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())

    class_counts = Counter(all_labels)
    num_classes = len(class_counts)
    counts = np.array([class_counts[i] for i in range(num_classes)])
    weights = 1.0 / counts
    weights = weights / weights.sum()  # optional normalization
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    model.to(device)
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0  # NEW: Track early stopping

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = legacy_train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = legacy_validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")

        # Write to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch+1)
            writer.add_scalar('Loss/validation', val_loss, epoch+1)
            writer.add_scalar('Accuracy/train', train_acc, epoch+1)
            writer.add_scalar('Accuracy/validation', val_acc, epoch+1)

        # Early stopping check
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"✔ Best model updated and saved to {save_path}")
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            print(f"⚠ No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print(f"⏹ Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.2f}%")
    model.load_state_dict(best_model_wts)


def legacy_train_multiple_runs(parameters):
    for i, param_block in enumerate(parameters):
        print(f"\n----- Starting training run {i+1} -----")

        # --- Load dataset + get class mapping ---
        if param_block["data_type"] == "classic":
            train_loader, val_loader, _, class_to_idx = data_loaders.general_image_loader(
                image_folder=param_block["image_folder"],
                csv_file=param_block["csv_file"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block["num_classes"]
            )

        elif param_block["data_type"] == "cwt":
            train_loader, val_loader, _, class_to_idx = data_loaders.cwt_loader(
                image_folder=param_block["image_folder"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block["num_classes"]
            )

        elif param_block["data_type"] == "mfcc":
            train_loader, val_loader, _, class_to_idx = data_loaders.mfcc_loader(
                image_folder=param_block["image_folder"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block["num_classes"]
            )

        elif param_block["data_type"] == "multimodal":
            train_loader, val_loader, _, class_to_idx = data_loaders.multimodal_loader(
                folders=param_block["folders"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block["num_classes"]
            )

        elif param_block["data_type"] == "multi_source":
            # NEW: single-backbone training from MULTIPLE folders (e.g., pvs1_video + pvs9_video or pvs1_mfcc + pvs9_mfcc)
            # Expect: param_block["image_folders"] = [folder1, folder2, ...]
            train_loader, val_loader, _, class_to_idx = data_loaders.general_image_loader_multi(
                image_folders=param_block["image_folders"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block["num_classes"]
            )

        else:
            raise ValueError(f"Unknown data_type: {param_block['data_type']}")

        # --- Dynamically get number of classes ---
        num_classes = len(class_to_idx)

        # --- Create model with correct output layer ---
        model_class = param_block["model_class"]
        model = model_class(num_classes)

        # --- Optimizer ---
        optimizer = torch.optim.Adam(model.parameters(), lr=param_block["learning_rate"])

        # --- Train ---
        training_functions.legacy_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=param_block["epochs"],
            device=param_block["device"],
            save_path=param_block["model_save_path"],
            writer=param_block["writer"],
            early_stopping_patience=param_block.get("early_stopping_patience", 5),
            min_delta=param_block.get("min_delta", 0.001)
        )

        print(f"----- Finished training run {i+1} -----\n")


def legacy_grid_search_train(base_params, search_space, run_name_prefix="grid"):
    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))
    run_configs = []

    for combo in combinations:
        config = base_params.copy()
        for key, value in zip(keys, combo):
            config[key] = value

        # Safe setting string
        setting_str = "__".join(
            f"{k}_{(v.__name__ if isinstance(v, type) else str(v)).replace('/', '').replace(':', '').replace(' ', '')}".replace("\\", "")
            for k, v in zip(keys, combo)
        )

        # TensorBoard writer path
        config["writer"] = SummaryWriter(
            log_dir=os.path.join("tensorboard", "runs", "hyperparameteroptimization", f"{run_name_prefix}__{setting_str}")
        )

        # Model save path
        model_dir = os.path.join("models", "hyperparameter_optimization")
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{run_name_prefix}__{setting_str}.pt"
        config["model_save_path"] = os.path.join(model_dir, model_filename)

        run_configs.append(config)

    # Print summary
    print(f"\n🔍 Starting {run_name_prefix.upper()}... {len(run_configs)} total runs will be performed.")

    print("\n📌 Fixed (base) parameters:")
    for key, value in base_params.items():
        if key not in search_space:
            val_str = value.__name__ if isinstance(value, type) else str(value)
            print(f"  - {key}: {val_str}")

    print("\n🧪 Varying parameters:")
    for key, values in search_space.items():
        value_strs = [v.__name__ if isinstance(v, type) else str(v) for v in values]
        print(f"  - {key}: {value_strs}")

    print("\n🧵 Full list of experiment configurations:")
    for i, cfg in enumerate(run_configs):
        summary = {k: (v.__name__ if isinstance(v, type) else v) for k, v in cfg.items() if k in search_space}
        print(f"  Run {i+1}: {summary}")

    legacy_train_multiple_runs(run_configs)


def legacy_random_search_train(base_params, search_space, num_samples=5, run_name_prefix="random"):
    seen = set()
    run_configs = []

    keys = list(search_space.keys())
    all_possible = 1
    for values in search_space.values():
        all_possible *= len(values)

    if num_samples > all_possible:
        raise ValueError(f"num_samples ({num_samples}) exceeds total unique combinations ({all_possible}).")

    while len(run_configs) < num_samples:
        config_values = tuple(random.choice(search_space[k]) for k in keys)
        if config_values not in seen:
            seen.add(config_values)
            config = base_params.copy()
            for k, v in zip(keys, config_values):
                config[k] = v

            # Safe setting string
            setting_str = "__".join(
                f"{k}_{(v.__name__ if isinstance(v, type) else str(v)).replace('/', '').replace(':', '').replace(' ', '')}".replace("\\", "")
                for k, v in zip(keys, config_values)
            )

            # TensorBoard writer path
            config["writer"] = SummaryWriter(
                log_dir=os.path.join("tensorboard", "runs", "hyperparameteroptimization", f"{run_name_prefix}__{setting_str}")
            )

            # Model save path
            model_dir = os.path.join("models", "hyperparameter_optimization")
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"{run_name_prefix}__{setting_str}.pt"
            config["model_save_path"] = os.path.join(model_dir, model_filename)

            run_configs.append(config)

    # Print summary
    print(f"\n🔍 Starting {run_name_prefix.upper()}... {len(run_configs)} total runs will be performed.")

    print("\n📌 Fixed (base) parameters:")
    for key, value in base_params.items():
        if key not in search_space:
            val_str = value.__name__ if isinstance(value, type) else str(value)
            print(f"  - {key}: {val_str}")

    print("\n🧪 Varying parameters:")
    for key, values in search_space.items():
        value_strs = [v.__name__ if isinstance(v, type) else str(v) for v in values]
        print(f"  - {key}: {value_strs}")

    print("\n🧵 Full list of experiment configurations:")
    for i, cfg in enumerate(run_configs):
        summary = {k: (v.__name__ if isinstance(v, type) else v) for k, v in cfg.items() if k in search_space}
        print(f"  Run {i+1}: {summary}")

    legacy_train_multiple_runs(run_configs)


## Below experimentation Multimodal

def grid_search_train(base_params, search_space, run_name_prefix="grid"):
    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))
    run_configs = []

    for combo in combinations:
        config = base_params.copy()
        for key, value in zip(keys, combo):
            config[key] = value

        # Safe setting string
        setting_str = "__".join(
            f"{k}_{(v.__name__ if isinstance(v, type) else str(v)).replace('/', '').replace(':', '').replace(' ', '')}".replace("\\", "")
            for k, v in zip(keys, combo)
        )

        # TensorBoard writer path
        config["writer"] = SummaryWriter(
            log_dir=os.path.join("tensorboard", "runs", "hyperparameteroptimization", f"{run_name_prefix}__{setting_str}")
        )

        # Model save path
        model_dir = os.path.join("models", "hyperparameter_optimization")
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{run_name_prefix}__{setting_str}.pt"
        config["model_save_path"] = os.path.join(model_dir, model_filename)

        run_configs.append(config)

    # Print summary
    print(f"\n🔍 Starting {run_name_prefix.upper()}... {len(run_configs)} total runs will be performed.")

    print("\n📌 Fixed (base) parameters:")
    for key, value in base_params.items():
        if key not in search_space:
            val_str = value.__name__ if isinstance(value, type) else str(value)
            print(f"  - {key}: {val_str}")

    print("\n🧪 Varying parameters:")
    for key, values in search_space.items():
        value_strs = [v.__name__ if isinstance(v, type) else str(v) for v in values]
        print(f"  - {key}: {value_strs}")

    print("\n🧵 Full list of experiment configurations:")
    for i, cfg in enumerate(run_configs):
        summary = {k: (v.__name__ if isinstance(v, type) else v) for k, v in cfg.items() if k in search_space}
        print(f"  Run {i+1}: {summary}")

    train_multiple_runs(run_configs)


def random_search_train(base_params, search_space, num_samples=5, run_name_prefix="random"):
    seen = set()
    run_configs = []

    keys = list(search_space.keys())
    all_possible = 1
    for values in search_space.values():
        all_possible *= len(values)

    if num_samples > all_possible:
        raise ValueError(f"num_samples ({num_samples}) exceeds total unique combinations ({all_possible}).")

    while len(run_configs) < num_samples:
        config_values = tuple(random.choice(search_space[k]) for k in keys)
        if config_values not in seen:
            seen.add(config_values)
            config = base_params.copy()
            for k, v in zip(keys, config_values):
                config[k] = v

            # Safe setting string
            setting_str = "__".join(
                f"{k}_{(v.__name__ if isinstance(v, type) else str(v)).replace('/', '').replace(':', '').replace(' ', '')}".replace("\\", "")
                for k, v in zip(keys, config_values)
            )

            # TensorBoard writer path
            config["writer"] = SummaryWriter(
                log_dir=os.path.join("tensorboard", "runs", "hyperparameteroptimization", f"{run_name_prefix}__{setting_str}")
            )

            # Model save path
            model_dir = os.path.join("models", "hyperparameter_optimization")
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"{run_name_prefix}__{setting_str}.pt"
            config["model_save_path"] = os.path.join(model_dir, model_filename)

            run_configs.append(config)

    # Print summary
    print(f"\n🔍 Starting {run_name_prefix.upper()}... {len(run_configs)} total runs will be performed.")

    print("\n📌 Fixed (base) parameters:")
    for key, value in base_params.items():
        if key not in search_space:
            val_str = value.__name__ if isinstance(value, type) else str(value)
            print(f"  - {key}: {val_str}")

    print("\n🧪 Varying parameters:")
    for key, values in search_space.items():
        value_strs = [v.__name__ if isinstance(v, type) else str(v) for v in values]
        print(f"  - {key}: {value_strs}")

    print("\n🧵 Full list of experiment configurations:")
    for i, cfg in enumerate(run_configs):
        summary = {k: (v.__name__ if isinstance(v, type) else v) for k, v in cfg.items() if k in search_space}
        print(f"  Run {i+1}: {summary}")

    train_multiple_runs(run_configs)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_notebook_tqdm=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    tqdm_fn = notebook_tqdm if use_notebook_tqdm else std_tqdm
    loop = tqdm_fn(dataloader, desc="Training", leave=True)

    for batch in loop:
        *images, labels = batch  # Unpack all inputs + label
        images = [img.to(device) for img in images]
        labels = labels.to(device)

        optimizer.zero_grad()

        # ---- AMP guard (only enable on CUDA with a scaler) ----
        use_amp = (scaler is not None) and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(*images)  # Forward with multiple inputs
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, save_path, writer,
                early_stopping_patience=5, min_delta=0.001, num_classes=None):
    # Prepare writer starting values
    if writer:
        writer.add_scalar('Loss/train', 2, 0)
        writer.add_scalar('Loss/validation', 2, 0)
        writer.add_scalar('Accuracy/train', 0, 0)
        writer.add_scalar('Accuracy/validation', 0, 0)

    # --- Compute class weights from training data ---
    all_labels = []
    for batch in train_loader:
        labels = batch[-1]  # last element is always the label
        all_labels.extend(labels.cpu().numpy())

    if num_classes is None:
        num_classes = max(all_labels) + 1  # fallback if not provided

    counts = np.zeros(num_classes, dtype=np.int64)
    for lbl in all_labels:
        counts[lbl] += 1

    # Avoid division by zero for unseen classes
    counts[counts == 0] = 1  

    weights = 1.0 / counts
    weights = weights / weights.sum()
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    # --- Loss + mixed precision ---
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    model.to(device)
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    best_snapshot = {
        "epoch": 0,
        "train_loss": None,
        "train_acc": None,
        "val_loss": None,
        "val_acc": 0.0
    }


    # --- Training loop ---
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch+1)
            writer.add_scalar('Loss/validation', val_loss, epoch+1)
            writer.add_scalar('Accuracy/train', train_acc, epoch+1)
            writer.add_scalar('Accuracy/validation', val_acc, epoch+1)

        # --- Early stopping ---
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"✔ Best model updated and saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= early_stopping_patience:
                print(f"⏹ Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.2f}%")
    model.load_state_dict(best_model_wts)

    # NEW: return summary for fold logging
    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_snapshot["epoch"]),
        "best_train_loss": None if best_snapshot["train_loss"] is None else float(best_snapshot["train_loss"]),
        "best_train_acc": None if best_snapshot["train_acc"] is None else float(best_snapshot["train_acc"]),
        "best_val_loss": None if best_snapshot["val_loss"] is None else float(best_snapshot["val_loss"]),
        "best_val_acc_at_save": float(best_snapshot["val_acc"])
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            *images, labels = batch  # Unpack all inputs + label
            images = [img.to(device) for img in images]
            labels = labels.to(device)

            outputs = model(*images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_multiple_runs(parameters):
    for i, param_block in enumerate(parameters):
        print(f"\n----- Starting training run {i+1} -----")

        # --- Load dataset + get class mapping ---
        if param_block["data_type"] == "classic":
            train_loader, val_loader, _, class_to_idx = data_loaders.general_image_loader(
                image_folder=param_block["image_folder"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block.get("num_classes", 3)
            )

        elif param_block["data_type"] == "cwt":
            train_loader, val_loader, _, class_to_idx = data_loaders.cwt_loader(
                image_folder=param_block["image_folder"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block.get("num_classes", 3)
            )

        elif param_block["data_type"] == "mfcc":
            train_loader, val_loader, _, class_to_idx = data_loaders.mfcc_loader(
                image_folder=param_block["image_folder"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block.get("num_classes", 3)
            )

        elif param_block["data_type"] == "multimodal":
            # --- Old multimodal path (unchanged) ---
            train_loader, val_loader, _, class_to_idx = data_loaders.multimodal_loader(
                folders=param_block["image_folders"],
                batch_size=param_block["batch_size"],
                resize=param_block["resize"],
                return_class_map=True,
                num_classes=param_block.get("num_classes", 3)
            )

        elif param_block["data_type"] == "multimodal_special":
            duplicate = param_block.get("duplicate", True)
            train_loader, val_loader, _, class_to_idx = data_loaders.special_multimodal_loader(
            folder_pairs=param_block.get("folder_pairs"),
            batch_size=param_block["batch_size"],
            resize=param_block["resize"],   # <- use resize
            return_class_map=True,
            num_classes=param_block.get("num_classes", 3),
            duplicate=duplicate
    )


        else:
            raise ValueError(f"Unknown data_type: {param_block['data_type']}")

        # --- Dynamically get number of classes ---
        num_classes = len(class_to_idx)

        # --- Create model (keeps old behavior) ---
        model_class = param_block["model_class"]

        if isinstance(model_class, nn.Module):
            # Already a constructed model instance (your current pattern)
            model = model_class

        elif callable(model_class):
            # For old single-backbone cases, call with num_classes
            # For multimodal and multimodal_special, allow the same BC+device signature you used before
            if param_block["data_type"] in ("multimodal", "multimodal_special"):
                model = model_class(
                    param_block.get("backbone_configs"),
                    param_block["device"]
                )
            else:
                model = model_class(num_classes)
        else:
            raise ValueError(f"Invalid model_class: {model_class}")

        # --- Optimizer ---
        optimizer = torch.optim.Adam(model.parameters(), lr=param_block["learning_rate"])

        # --- Train ---
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=param_block["epochs"],
            device=param_block["device"],
            save_path=param_block["model_save_path"],
            writer=param_block["writer"],
            early_stopping_patience=param_block.get("early_stopping_patience", 5),
            min_delta=param_block.get("min_delta", 0.001),
            num_classes=num_classes  # keep class-weighting correct
        )

        print(f"----- Finished training run {i+1} -----\n")


####### Kfold cross validation below


import os, time, copy, math, random
import numpy as np
from statistics import mean, stdev
from typing import Dict, Tuple, List, Callable
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn 

# ---- helpers for summary txt ----


def _default_summary_path(root_dir, prefix="kfold_summary"):
    os.makedirs(root_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(root_dir, f"{prefix}_{stamp}.txt")

def _fmt(x):
    """Helper for consistent string formatting."""
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)

def write_kfold_summary_txt(rows: List[Dict], out_path: str, title: str = "K-fold summary"):
    """
    Write per-fold metrics and averages to a .txt summary file.
    Automatically creates the output directory if missing.
    """
    keys_to_avg = ["best_val_acc", "best_val_acc_at_save", "best_val_loss", "best_train_loss", "best_train_acc"]
    buckets = {k: [] for k in keys_to_avg}

    # ✅ Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total folds: {len(rows)}\n\n")

        header = [
            "fold", "save_path", "best_epoch",
            "best_val_acc", "best_val_acc_at_save",
            "best_val_loss", "best_train_loss", "best_train_acc"
        ]
        f.write("\t".join(header) + "\n")

        for row in rows:
            for k in keys_to_avg:
                v = row.get(k, None)
                if isinstance(v, (int, float)):
                    buckets[k].append(float(v))

            line = [
                _fmt(row.get("fold")),
                _fmt(row.get("model_save_path")),
                _fmt(row.get("best_epoch")),
                _fmt(row.get("best_val_acc")),
                _fmt(row.get("best_val_acc_at_save")),
                _fmt(row.get("best_val_loss")),
                _fmt(row.get("best_train_loss")),
                _fmt(row.get("best_train_acc")),
            ]
            f.write("\t".join(line) + "\n")

        f.write("\nAverages (and stdev if >=2 folds):\n")
        for k in keys_to_avg:
            vals = buckets[k]
            if len(vals) == 0:
                f.write(f"{k}: NA\n")
            elif len(vals) == 1:
                f.write(f"{k}: mean={mean(vals):.6f}\n")
            else:
                f.write(f"{k}: mean={mean(vals):.6f}, stdev={stdev(vals):.6f}\n")

    print(f"✅ Summary written to: {out_path}")


# ---- optional: reuse your model-construction logic ----
def build_model_from_param_block(param_block: Dict, num_classes: int) -> nn.Module:
    model_class = param_block["model_class"]
    if isinstance(model_class, nn.Module):
        return model_class
    if callable(model_class):
        if param_block["dataset_type"] in ("multimodal", "multimodal_special"):
            return model_class(param_block.get("backbone_configs"), param_block["device"])
        else:
            return model_class(num_classes)
    raise ValueError(f"Invalid model_class: {model_class}")


# ---- fold loaders from a shared dataset ----
def make_fold_loaders(dataset, train_idx, val_idx, batch_size, num_workers=4, pin_memory=True, collate_fn=None):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )
    return train_loader, val_loader


# ---- MAIN: K-fold trainer for one multimodal experiment block ----
def kfold_train_multimodal(param_block: Dict,
                           K: int = 5,
                           seed: int = 42,
                           summary_txt_path: str = None):
    """
    param_block fields expected (typical):
      - dataset_type: "multimodal" or "multimodal_special"
      - image_folders or folder_pairs (for special)
      - resize, batch_size, device, learning_rate, epochs
      - model_class: callable or instance
      - backbone_configs (for multimodal models)
      - model_save_path: base path (we append __fold{i}of{K}.pt)
      - log_dir_base (optional): base TB dir; we append /fold_{i}_of_{K}
      - collate_fn (optional): if your dataset needs a custom collation
      - dataset_builder: callable -> (dataset, labels, class_to_idx)
          signature: dataset_builder(param_block) -> (dataset, labels, class_to_idx)
    """
    assert param_block["dataset_type"] in ("multimodal", "multimodal_special"), \
        "kfold_train_multimodal only supports multimodal datasets."

    # 1) Build the full dataset once
    if "dataset_builder" in param_block and callable(param_block["dataset_builder"]):
        dataset, labels, class_to_idx = param_block["dataset_builder"](param_block)
    else:
        # Fallback to a conventional entry point you add in data_loaders
        # e.g., data_loaders.get_multimodal_dataset(...)
        import data_loaders
        if param_block["dataset_type"] == "multimodal":
            dataset, labels, class_to_idx = data_loaders.get_multimodal_dataset(
                folders=param_block["image_folders"],
                resize=param_block["resize"],
                num_classes=param_block.get("num_classes", 3),
                return_class_map=True
            )
        else:
            dataset, labels, class_to_idx = data_loaders.get_special_multimodal_dataset(
                folder_pairs=param_block["folder_pairs"],
                resize=param_block["resize"],
                num_classes=param_block.get("num_classes", 3),
                return_class_map=True,
                duplicate=param_block.get("duplicate", True)
            )

    num_classes = len(class_to_idx)
    labels = np.asarray(labels)
    n = len(labels)
    print(f"Total samples: {n}, num_classes: {num_classes}")

    # 2) Stratified split indices
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    # 3) Prepare logging destinations
    base_model_path = param_block.get("model_save_path", "models/multimodal_model.pt")
    base_root = os.path.dirname(base_model_path) if os.path.dirname(base_model_path) else "models"
    os.makedirs(base_root, exist_ok=True)

    log_dir_base = param_block.get("log_dir_base", os.path.join("tensorboard", "runs", "kfold_multimodal"))
    os.makedirs(log_dir_base, exist_ok=True)

    if summary_txt_path is None:
        summary_txt_path = _default_summary_path(base_root, prefix="kfold_summary")

    # 4) Collate function if needed
    collate_fn = param_block.get("collate_fn", None)

    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), labels), start=1):
        print(f"\n===== Fold {fold_idx}/{K} =====")
        train_loader, val_loader = make_fold_loaders(
            dataset, train_idx, val_idx,
            batch_size=param_block["batch_size"],
            num_workers=param_block.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn
        )

        # 5) Build model and optimizer fresh each fold
        model = build_model_from_param_block(param_block, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=param_block["learning_rate"])

        # 6) Per-fold save and writer
        fold_suffix = f"__fold{fold_idx}of{K}"
        fold_save_path = os.path.splitext(base_model_path)[0] + fold_suffix + ".pt"
        os.makedirs(os.path.dirname(fold_save_path), exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join(log_dir_base, f"fold_{fold_idx}_of_{K}"))

        # 7) Train using existing loop
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=param_block["epochs"],
            device=param_block["device"],
            save_path=fold_save_path,
            writer=writer,
            early_stopping_patience=param_block.get("early_stopping_patience", 5),
            min_delta=param_block.get("min_delta", 0.001),
            num_classes=num_classes
        )

        writer.close()

        # 8) Collect fold summary for TXT
        fold_rows.append({
            "fold": fold_idx,
            "model_save_path": fold_save_path,
            "best_epoch": result.get("best_epoch"),
            "best_val_acc": result.get("best_val_acc"),
            "best_val_acc_at_save": result.get("best_val_acc_at_save"),
            "best_val_loss": result.get("best_val_loss"),
            "best_train_loss": result.get("best_train_loss"),
            "best_train_acc": result.get("best_train_acc"),
        })

    # 9) Write TXT with per-fold rows and averages
    write_kfold_summary_txt(fold_rows, summary_txt_path, title="K-fold Multimodal Summary")
    print(f"✔ K-fold summary written to: {summary_txt_path}")

    return fold_rows, summary_txt_path
