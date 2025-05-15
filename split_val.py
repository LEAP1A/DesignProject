import os
import random
import shutil
import math
from collections import defaultdict

IMG_TRAIN_DIR = r'C:\Users\TUTU\Desktop\DP\images/train'
LBL_TRAIN_DIR = r'C:\Users\TUTU\Desktop\DP\labels/train'
IMG_VAL_DIR   = r'C:\Users\TUTU\Desktop\DP\images/val'
LBL_VAL_DIR   = r'C:\Users\TUTU\Desktop\DP\labels/val'
VAL_RATIO     = 0.2  # 20%

os.makedirs(IMG_VAL_DIR, exist_ok=True)
os.makedirs(LBL_VAL_DIR, exist_ok=True)

class_to_files = defaultdict(set)
for fname in os.listdir(LBL_TRAIN_DIR):
    if not fname.endswith('.txt'):
        continue
    path = os.path.join(LBL_TRAIN_DIR, fname)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls_id = parts[0]
                class_to_files[cls_id].add(fname)

for cls_id, files in sorted(class_to_files.items(), key=lambda x: int(x[0])):
    files_list = list(files)
    total = len(files_list)
    num_to_move = max(1, math.ceil(total * VAL_RATIO))
    selected = random.sample(files_list, min(num_to_move, total))

    print(f"Class {cls_id}: total images = {total}, moving {len(selected)} to validation set")

    for lbl_fname in selected:
        img_fname = lbl_fname.replace('.txt', '.jpg')

        src_img = os.path.join(IMG_TRAIN_DIR, img_fname)
        dst_img = os.path.join(IMG_VAL_DIR, img_fname)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            print(f"  Moved image: {src_img} -> {dst_img}")
        else:
            print(f"  WARNING: image file not found: {src_img}")

        src_lbl = os.path.join(LBL_TRAIN_DIR, lbl_fname)
        dst_lbl = os.path.join(LBL_VAL_DIR, lbl_fname)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
            print(f"  Moved label: {src_lbl} -> {dst_lbl}")
        else:
            print(f"  WARNING: label file not found: {src_lbl}")

    print()  
