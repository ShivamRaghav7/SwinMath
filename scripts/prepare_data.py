import random
import os
import csv

MASTER_TXT = r"data\train.txt"
OUTPUT_DIR = "data"

# read all lines 
with open(MASTER_TXT, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# shuffle and split (90/5/5)
random.seed(42)
random.shuffle(lines)
total = len(lines)
train_end = int(total * 0.90)
val_end = int(total * 0.95)
splits = {
    "train.csv": lines[:train_end],
    "val.csv": lines[train_end: val_end],
    "test.csv": lines[val_end:]
}

# write to csv
for filename, split_lines in splits.items():
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'label'])
        for line in split_lines:
            left_stripped = line.removeprefix("images/")
            parts = left_stripped.strip().split('\t')
            if len(parts) == 2:
                writer.writerow([parts[0], parts[1]])