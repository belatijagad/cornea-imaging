import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare one slit-lamp image for the classifier.")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New directory for the processed image and its masks")
    parser.add_argument("--precomputed", action="store_true", help="Use an already-processed input image")
    parser.add_argument("--device", help="MedSAM device, such as cuda:0 or cpu")
    args = parser.parse_args(argv)
    with Image.open(args.image) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    if not args.precomputed:
        from cornea.medsam import MedSAMPreprocessor

        result = MedSAMPreprocessor(args.device)(image)
    args.output.mkdir(parents=True)
    if not args.precomputed:
        result.raw_mask.save(args.output / "medsam_mask.png")
        result.ellipse.save(args.output / "ellipse_mask.png")
        image = result.image
        if result.used_box_fallback:
            print("No MedSAM mask passed the ellipse checks; used the central prompt box.")
    target = args.output / "processed.png"
    image.save(target)
    print(target)
    return target


if __name__ == "__main__":
    main()
