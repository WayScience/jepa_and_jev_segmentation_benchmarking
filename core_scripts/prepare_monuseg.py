"""Convert raw MoNuSeg annotations into the dsb2018-style layout.

Input is the extracted "MoNuSeg Training Data" folder from the official
Zenodo release (DOI 10.5281/zenodo.1473148), containing:

    Annotations/*.xml      per-image polygon regions (one <Region> per nucleus)
    Tissue images/*.tif    1000x1000 H&E RGB tiles

This script rasterizes each image's nucleus polygons into an instance-label
mask and writes grayscale image + uint16 mask pairs into:

    <out>/train/images/*.tif
    <out>/train/masks/*.tif
    <out>/test/images/*.tif
    <out>/test/masks/*.tif

matching the layout core_scripts/common.py.load_split expects.

The official MoNuSeg challenge test set (14 WSIs) is gated behind separate
registration on grand-challenge.org and is not included in the Zenodo
"Training Data" release. Until that's fetched separately, this script holds
out a deterministic subset of the 30 released images as a local test split
so the pipeline has something to evaluate on. This is NOT the official
held-out test set -- scores from it aren't comparable to published MoNuSeg
leaderboard numbers.
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from PIL import Image
from skimage.draw import polygon as sk_polygon


def parse_regions(xml_path):
    root = ET.parse(xml_path).getroot()
    regions = []
    for region in root.findall(".//Region"):
        verts = region.findall(".//Vertex")
        if len(verts) < 3:
            continue
        xs = [float(v.get("X")) for v in verts]
        ys = [float(v.get("Y")) for v in verts]
        regions.append((xs, ys))
    return regions


def rasterize(regions, shape):
    mask = np.zeros(shape, dtype=np.uint16)
    for i, (xs, ys) in enumerate(regions, start=1):
        rr, cc = sk_polygon(np.array(ys), np.array(xs), shape=shape)
        mask[rr, cc] = i
    return mask


def convert_one(xml_path, tif_path, img_out, mask_out):
    img = np.array(Image.open(tif_path).convert("L"), dtype=np.uint8)
    regions = parse_regions(xml_path)
    mask = rasterize(regions, img.shape)
    tifffile.imwrite(img_out, img)
    tifffile.imwrite(mask_out, mask)
    return mask.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "raw", help='Path to the extracted "MoNuSeg Training Data" directory'
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "monuseg"),
    )
    ap.add_argument(
        "--test-count",
        type=int,
        default=6,
        help="Images held out as a local test split",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann_dir = os.path.join(args.raw, "Annotations")
    img_dir = os.path.join(args.raw, "Tissue images")
    xmls = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    if not xmls:
        raise FileNotFoundError(f"No .xml annotations found in {ann_dir}")

    stems = [os.path.splitext(os.path.basename(x))[0] for x in xmls]
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(stems))
    test_idx = set(order[: args.test_count].tolist())

    for split in ("train", "test"):
        os.makedirs(os.path.join(args.out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, "masks"), exist_ok=True)

    counts = []
    for i, stem in enumerate(stems):
        split = "test" if i in test_idx else "train"
        xml_path = os.path.join(ann_dir, stem + ".xml")
        tif_path = os.path.join(img_dir, stem + ".tif")
        img_out = os.path.join(args.out, split, "images", stem + ".tif")
        mask_out = os.path.join(args.out, split, "masks", stem + ".tif")
        n = convert_one(xml_path, tif_path, img_out, mask_out)
        counts.append(n)
        print(f"[{split}] {stem}: {n} nuclei")

    print(
        f"Done. {len(stems) - len(test_idx)} train / {len(test_idx)} test images. "
        f"Total nuclei: {sum(counts)}."
    )


if __name__ == "__main__":
    main()
