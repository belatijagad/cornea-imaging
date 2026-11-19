from PIL import Image, ImageOps

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CorneaDataset(Dataset):
    def __init__(self, frame, transform, label_col):
        self.label_col = label_col
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row["image_path"]) as image:
            image = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        targets = torch.tensor(row[self.label_col].to_numpy(dtype=np.float32))
        return image, targets


def make_transforms(config, training=False):
    operations = [transforms.Resize((config.image_size, config.image_size))]
    if training:
        operations += [transforms.RandomHorizontalFlip(config.horizontal_flip_probability),
                       transforms.RandomRotation(config.rotation_degrees)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])


def load_manifest(run, config):
    frame = pd.read_csv(run / "manifest.csv", dtype={"patient_id": "string"})
    frame["image_path"] = frame["image_path"].map(lambda path: str((config.data.data_root / path).resolve()))
    return frame
