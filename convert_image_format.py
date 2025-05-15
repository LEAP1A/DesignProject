import os
import cv2

input_dir = r'C:\Users\TUTU\Desktop\DP\TestIJCNN2013'
output_dir = r'C:\Users\TUTU\Desktop\DP\testset'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.ppm'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

    
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(output_dir, new_filename)
        cv2.imwrite(output_path, img)

print('All PPM images converted to JPG!')
