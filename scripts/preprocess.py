# Run the full preprocessing pipeline
import os
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import MayoDataset
from data.preprocessing import get_transform

DATASET_PATH = "data/raw"

all_images = glob(
    os.path.join(DATASET_PATH, "**", "*.png"),
    recursive=True
)

print("Found images:", len(all_images))

train_images, temp_images = train_test_split(
    all_images,
    test_size=0.3,
    random_state=42
)

val_images, test_images = train_test_split(
    temp_images,
    test_size=0.5,
    random_state=42
)

transform = get_transform()

train_dataset = MayoDataset(train_images, transform)
val_dataset = MayoDataset(val_images, transform)
test_dataset = MayoDataset(test_images, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))