from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

BOX_EDGES = ("left", "top", "right", "bottom")
CORNEA_BOX_COLUMNS = [f"cornea_{edge}" for edge in BOX_EDGES]


def attach_cornea_boxes(frame, path, data_root):
    boxes = pd.read_csv(path, keep_default_na=False)
    required = {"source", *BOX_EDGES}
    if not required.issubset(boxes.columns):
        raise ValueError(f"Cornea box CSV must contain columns {', '.join(sorted(required))}")
    boxes["source"] = boxes["source"].map(lambda source: str((data_root / Path(source)).resolve()))
    if boxes["source"].duplicated().any():
        raise ValueError("Cornea box CSV must contain at most one box per image")
    coordinates = boxes.set_index("source")[list(BOX_EDGES)].apply(pd.to_numeric)
    missing = ~frame["image_path"].isin(coordinates.index)
    if missing.any():
        raise ValueError(f"Cornea boxes are missing for {missing.sum()} images in the classifier cohort")
    frame = frame.copy()
    frame[CORNEA_BOX_COLUMNS] = coordinates.loc[frame["image_path"]].to_numpy()
    return frame


def attach_preprocessed_images(frame, manifest_path, data_root):
    images = pd.read_csv(manifest_path, keep_default_na=False)
    if not {"source", "preprocessed_path"}.issubset(images.columns):
        raise ValueError("Preprocessed image CSV must contain source and preprocessed_path columns")
    sources = images["source"].map(lambda source: str((data_root / source).resolve()))
    if sources.duplicated().any():
        raise ValueError("Preprocessed image CSV must contain at most one entry per source")
    outputs = images["preprocessed_path"].map(
        lambda path: str((manifest_path.parent / path).resolve()) if path else None
    )
    paths = frame["image_path"].map(pd.Series(outputs.to_numpy(), index=sources))
    missing = paths.isna()
    if missing.any():
        raise ValueError(f"Preprocessed paths are missing for {missing.sum()} images in the classifier cohort")
    for path in paths.unique():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Preprocessed image is missing: {path}")
    return frame.assign(preprocessed_path=paths)


def generate_preprocessed_images(frame, output):
    from cornea.medsam import MedSAMPreprocessor

    preprocess = MedSAMPreprocessor()
    image_dir = output / "preprocessed"
    image_dir.mkdir()
    rows = []
    paths = frame["image_path"].unique()
    for index, source in enumerate(paths, 1):
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        result = preprocess(image)
        target = image_dir / f"{index:05d}.png"
        result.image.save(target)
        rows.append((source, str(target.relative_to(output)), result.used_box_fallback))
        print(f"Preprocessed {index}/{len(paths)}", flush=True)

    manifest = output / "preprocessed_images.csv"
    pd.DataFrame(rows, columns=["source", "preprocessed_path", "used_box_fallback"]).to_csv(
        manifest, index=False,
    )
    return manifest
