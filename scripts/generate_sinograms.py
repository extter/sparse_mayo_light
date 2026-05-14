#!/usr/bin/env python3
"""
Genera sinogrammi clean e corrupted per tutti gli split del dataset.
"""

import os
import sys
import yaml
from glob import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.sinogram import (
    get_device,
    create_operators_dict,
    save_sinograms_for_split,
)


BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PREPROCESSED_PATH = os.path.join(BASE_DATA_PATH, "preprocessed")
SINOGRAM_CLEAN_PATH = os.path.join(BASE_DATA_PATH, "sinogram_clean")
SINOGRAM_CORRUPTED_PATH = os.path.join(BASE_DATA_PATH, "sinogram_corrupted")

SPLITS = ["train", "validation", "test"]


def load_config():
    data_config_path = os.path.join(PROJECT_ROOT, "configs", "data_config.yaml")

    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Config file not found: {data_config_path}")

    with open(data_config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = ["image_size", "angle_configs", "noise_level"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in {data_config_path}")

    return config


def check_directories(angle_configs):
    for split in SPLITS:
        split_dir = os.path.join(PREPROCESSED_PATH, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Missing preprocessed directory: {split_dir}")

        for n_angles in angle_configs:
            clean_dir = os.path.join(SINOGRAM_CLEAN_PATH, split, f"angles_{n_angles}")
            corrupted_dir = os.path.join(SINOGRAM_CORRUPTED_PATH, split, f"angles_{n_angles}")

            os.makedirs(clean_dir, exist_ok=True)
            os.makedirs(corrupted_dir, exist_ok=True)


def main():
    print("Starting sinogram generation...")

    config = load_config()
    image_size = config["image_size"]
    angle_configs = config["angle_configs"]
    noise_level = config["noise_level"]

    device = get_device()

    print(f"Using device: {device}")
    print(
        f"image_size={image_size}, angle_configs={angle_configs}, noise_level={noise_level}"
    )

    check_directories(angle_configs)

    operators_dict = create_operators_dict(
        angle_configs=angle_configs,
        image_size=image_size,
    )

    for split in SPLITS:
        image_paths = sorted(glob(os.path.join(PREPROCESSED_PATH, split, "*.npy")))

        if not image_paths:
            print(f"No preprocessed images found for split: {split}")
            continue

        save_sinograms_for_split(
            image_paths=image_paths,
            split_name=split,
            preprocessed_path=PREPROCESSED_PATH,
            sinogram_clean_path=SINOGRAM_CLEAN_PATH,
            sinogram_corrupted_path=SINOGRAM_CORRUPTED_PATH,
            operators_dict=operators_dict,
            angle_configs=angle_configs,
            noise_level=noise_level,
            device=device,
        )

    print("Sinogram generation completed!")


if __name__ == "__main__":
    main()