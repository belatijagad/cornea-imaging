from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from transformers import SamModel, SamProcessor

MODEL_ID = "wanglab/medsam-vit-base"


def largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (sizes.argmax() + 1)


@dataclass(frozen=True)
class SegmentationResult:
    image: Image.Image
    raw_mask: Image.Image
    ellipse: Image.Image
    used_box_fallback: bool


class MedSAMPreprocessor:
    def __init__(self, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(MODEL_ID)
        self.model = SamModel.from_pretrained(MODEL_ID, use_safetensors=False).to(device).eval()

    @torch.inference_mode()
    def predict_masks(self, image, boxes):
        inputs = self.processor(images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt")
        outputs = self.model(
            pixel_values=inputs["pixel_values"].to(self.model.device),
            input_boxes=inputs["input_boxes"].to(self.model.device),
            multimask_output=False,
        )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        return [Image.fromarray(mask[0].numpy()).convert("L") for mask in masks[0]]

    def __call__(self, image):
        width, height = image.size
        thumb = image.convert("L")
        thumb.thumbnail((256, 256))
        pixels = np.asarray(thumb)
        field = largest_component(pixels > max(18, 0.2 * np.percentile(pixels, 95)))
        rows, cols = np.nonzero(field)
        left, top, right, bottom = 0.0, 0.0, float(width), float(height)
        if rows.size >= 0.03 * field.size:
            field_left, field_right = np.percentile(cols, (2, 98)) * width / thumb.width
            field_top, field_bottom = np.percentile(rows, (2, 98)) * height / thumb.height
            if field_right - field_left >= 0.1 * width and field_bottom - field_top >= 0.1 * height:
                left, top, right, bottom = map(float, (field_left, field_top, field_right, field_bottom))

        field_width, field_height = right - left, bottom - top
        proposals = [(0.5, 0.45), (0.5, 0.6), (0.5, 0.75), (0.34, 0.55), (0.66, 0.55)]
        boxes = [
            (left + (center - scale / 2) * field_width, top + (0.5 - scale / 2) * field_height,
             left + (center + scale / 2) * field_width, top + (0.5 + scale / 2) * field_height)
            for center, scale in proposals
        ]
        masks = self.predict_masks(image, boxes)
        candidates = []
        for index, mask in enumerate(masks):
            foreground = ndimage.binary_fill_holes(largest_component(np.asarray(mask) > 0))
            y, x = np.nonzero(foreground)
            if x.size < 5:
                continue
            covariance = np.cov(x, y, bias=True)
            minor, major = np.linalg.eigvalsh(covariance)
            if minor < 0.25:
                continue
            axis_ratio = np.sqrt(minor / major)
            fraction = x.size / (field_width * field_height)
            if not 0.04 <= fraction <= 0.55 or axis_ratio < 0.35:
                continue
            center_x, center_y = x.mean(), y.mean()
            offset = np.hypot((center_x - left) / field_width - 0.5, (center_y - top) / field_height - 0.5)
            score = abs(np.log(fraction / 0.18)) + 3 * offset + (1 - axis_ratio)
            candidates.append((score, index, center_x, center_y, covariance))

        if candidates:
            _, index, center_x, center_y, covariance = min(candidates, key=lambda candidate: candidate[0])
            inverse = np.linalg.inv(covariance)
            rows, cols = np.ogrid[:height, :width]
            dx, dy = cols - center_x, rows - center_y
            radius = inverse[0, 0] * dx**2 + 2 * inverse[0, 1] * dx * dy + inverse[1, 1] * dy**2
            ellipse = Image.fromarray((radius <= 4).astype(np.uint8) * 255)
        else:
            index = 1
            ellipse = Image.new("L", image.size)
            ImageDraw.Draw(ellipse).ellipse(boxes[index], fill=255)

        processed = Image.composite(image, Image.new("RGB", image.size), ellipse)
        return SegmentationResult(processed, masks[index], ellipse, not candidates)
