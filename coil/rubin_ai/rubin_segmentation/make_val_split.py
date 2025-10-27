
import os, random, shutil
from pathlib import Path

def pair_exists(img_path: Path, labels_dir: Path):
    stem = img_path.stem
    cand = labels_dir / f"{stem}.txt"
    return cand.exists()

def gather_pairs(images_dir: Path, labels_dir: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    pairs = [(p, labels_dir / f"{p.stem}.txt") for p in imgs if (labels_dir / f"{p.stem}.txt").exists()]
    return pairs

def main(root, val_frac=0.1, seed=42, move=False, valid_name="valid"):
    random.seed(seed)
    root = Path(root)
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    valid_images = root / valid_name / "images"
    valid_labels = root / valid_name / "labels"

    assert train_images.exists(), f"Missing: {train_images}"
    assert train_labels.exists(), f"Missing: {train_labels}"

    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    pairs = gather_pairs(train_images, train_labels)
    if not pairs:
        raise SystemExit("No (image,label) pairs found in train/. Ensure YOLO-Seg labels exist.")

    n_total = len(pairs)
    n_val = max(1, int(n_total * float(val_frac)))
    random.shuffle(pairs)
    val_pairs = pairs[:n_val]

    op = shutil.move if move else shutil.copy2
    for img, lbl in val_pairs:
        dst_img = valid_images / img.name
        dst_lbl = valid_labels / lbl.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        op(str(img), str(dst_img))
        op(str(lbl), str(dst_lbl))

    print(f"Done. Created {len(val_pairs)} samples in '{valid_images.parent}'.")
    print(f"Mode: {'MOVE' if move else 'COPY'}")
    print(f"Train images remain at: {train_images}")
    print(f"Validation images at:   {valid_images}")
    print("Next: point data.yaml 'val:' to this folder (e.g., ../valid/images).")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root that contains train/ and (optionally) test/")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train to become VALID set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--valid-name", default="valid", help="Folder name for the validation set (default: valid)")
    args = ap.parse_args()
    main(args.root, args.val_frac, args.seed, args.move, args.valid_name)
