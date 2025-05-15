import os

input_label_file = r'C:\Users\TUTU\Desktop\DP\FullIJCNN2013\gt.txt'

output_label_dir = r'C:\Users\TUTU\Desktop\DP\labels'
os.makedirs(output_label_dir, exist_ok=True)

image_width = 1360
image_height = 800

labels_per_image = {}

with open(input_label_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split(';')
    if len(parts) != 6:
        continue

    filename, x_min, y_min, x_max, y_max, class_id = parts
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)
    class_id = int(class_id)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2

    # normalization
    x_center /= image_width
    y_center /= image_height
    bbox_width /= image_width
    bbox_height /= image_height

    # generate YOLO-style label
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

    base_name = os.path.splitext(filename)[0]

    if base_name not in labels_per_image:
        labels_per_image[base_name] = []
    labels_per_image[base_name].append(label_line)

for base_name, label_lines in labels_per_image.items():
    output_path = os.path.join(output_label_dir, f"{base_name}.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(label_lines))
