import os
from collections import defaultdict
import matplotlib.pyplot as plt

label_dir = r'C:\Users\TUTU\Desktop\DP\labels/train'

class_image_count = defaultdict(set)

for fname in os.listdir(label_dir):
    if not fname.endswith('.txt'):
        continue

    path = os.path.join(label_dir, fname)
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                cls_id = line.strip().split()[0]
                class_image_count[cls_id].add(fname)

total = 0
print("Class-wise image count:")
for cls_id in sorted(class_image_count.keys(), key=lambda x: int(x)):
    count = len(class_image_count[cls_id])
    total += count
    print(f"Class {cls_id}: {count} images")

print(f"\nTotal class-image associations: {total}")

classes = sorted(class_image_count.keys(), key=lambda x: int(x))
counts = [len(class_image_count[cls_id]) for cls_id in classes]

plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.title('FullIJCNN2013.zip Image Count Distribution')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
