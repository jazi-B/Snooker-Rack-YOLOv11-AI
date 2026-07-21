import os
import shutil
import random
from pathlib import Path

def setup_dataset_directories(base_path="dataset"):
    """
    Creates the required YOLO directory structure:
    dataset/
      ├── images/
      │   ├── train/
      │   ├── val/
      │   └── test/
      └── labels/
          ├── train/
          ├── val/
          └── test/
    """
    subdirs = ["images/train", "images/val", "images/test", 
               "labels/train", "labels/val", "labels/test"]
    
    for subdir in subdirs:
        dir_path = Path(base_path) / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Created YOLO dataset folder structure at: {os.path.abspath(base_path)}")

def split_raw_dataset(raw_img_dir, raw_lbl_dir, output_dataset_dir="dataset", train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    Splits raw images and .txt label files into train/val sets.
    Useful for both Pilot phase (20-30 images) and Full phase (150-300 images).
    """
    random.seed(seed)
    setup_dataset_directories(output_dataset_dir)
    
    raw_img_path = Path(raw_img_dir)
    raw_lbl_path = Path(raw_lbl_dir)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in raw_img_path.glob("*") if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"[!] No images found in {raw_img_dir}")
        return

    random.shuffle(image_files)
    
    num_total = len(image_files)
    num_train = int(num_total * train_ratio)
    
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]
    
    splits = [("train", train_files), ("val", val_files)]
    
    copied_count = 0
    for split_name, files in splits:
        img_out = Path(output_dataset_dir) / "images" / split_name
        lbl_out = Path(output_dataset_dir) / "labels" / split_name
        
        for img_file in files:
            # Copy Image
            shutil.copy(img_file, img_out / img_file.name)
            
            # Copy matching label file (.txt)
            lbl_file = raw_lbl_path / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy(lbl_file, lbl_out / lbl_file.name)
            else:
                # Create empty file if background image without rack
                (lbl_out / f"{img_file.stem}.txt").touch()
            
            copied_count += 1

    print(f"[✓] Split completed! Total images: {num_total} -> Train: {len(train_files)}, Val: {len(val_files)}")

def validate_annotations(dataset_dir="dataset"):
    """
    Checks if label files are formatted correctly (class_id x_center y_center width height normalized 0..1).
    """
    labels_dir = Path(dataset_dir) / "labels"
    invalid_files = []
    total_labels = 0
    
    for split in ["train", "val"]:
        split_lbl_dir = labels_dir / split
        if not split_lbl_dir.exists():
            continue
            
        for txt_file in split_lbl_dir.glob("*.txt"):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        invalid_files.append((txt_file.name, f"Line {line_idx}: expected 5 elements, got {len(parts)}"))
                        continue
                    
                    class_id, x, y, w, h = parts
                    try:
                        cls = int(class_id)
                        vals = [float(v) for v in (x, y, w, h)]
                        if cls != 0:
                            invalid_files.append((txt_file.name, f"Line {line_idx}: Unexpected class ID {cls} (only class 0 allowed)"))
                        for v in vals:
                            if not (0.0 <= v <= 1.0):
                                invalid_files.append((txt_file.name, f"Line {line_idx}: Coordinate out of range [0, 1]: {v}"))
                        total_labels += 1
                    except ValueError:
                        invalid_files.append((txt_file.name, f"Line {line_idx}: Non-numeric values found"))
                        
    if invalid_files:
        print(f"[!] Found {len(invalid_files)} annotation issues:")
        for fname, err in invalid_files:
            print(f"    - {fname}: {err}")
    else:
        print(f"[✓] All {total_labels} annotations validated successfully!")

if __name__ == "__main__":
    setup_dataset_directories()
    print("Dataset utility script ready. Usage in workflow documented in README.md.")
