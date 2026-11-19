import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from cornea.preprocessing import CORNEA_BOX_COLUMNS


class CorneaDataset(Dataset):
    def __init__(self, frame, transform, label_col):
        self.label_col = label_col
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.input_column = "preprocessed_path" if "preprocessed_path" in self.frame else "image_path"
        self.cornea_boxes = (
            self.frame[CORNEA_BOX_COLUMNS].to_numpy(dtype=np.float64)
            if self.input_column == "image_path" and all(column in self.frame for column in CORNEA_BOX_COLUMNS)
            else None
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row[self.input_column]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        if self.cornea_boxes is not None:
            box = tuple(self.cornea_boxes[idx])
            left, top, right, bottom = box
            width, height = image.size
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(f"Cornea box {box} must be an ordered rectangle within image size {image.size}")
            mask = Image.new("L", image.size)
            ImageDraw.Draw(mask).ellipse(box, fill=255)
            image = Image.composite(image, Image.new("RGB", image.size), mask)
        image = self.transform(image)
        targets = torch.tensor(row[self.label_col].to_numpy(dtype=np.float32))
        return image, targets


def make_transforms(config, training=False):
    operations = [transforms.Resize((config.image_size, config.image_size))]
    if training:
        operations += [transforms.RandomHorizontalFlip(config.augment.horizontal_flip_probability),
                       transforms.RandomRotation(config.augment.rotation_degrees)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])


def load_manifest(run, config):
    frame = pd.read_csv(run / "manifest.csv", dtype={"patient_id": "string"})
    frame["image_path"] = frame["image_path"].map(lambda path: str((config.data.data_root / path).resolve()))
    if "preprocessed_path" in frame:
        frame["preprocessed_path"] = frame["preprocessed_path"].map(lambda path: str((run / path).resolve()))
    return frame
