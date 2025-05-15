import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"C:\Users\TUTU\Desktop\DP\trained_model\weights\best.pt")

image_folder = r'C:\Users\TUTU\Desktop\DP\testset'

image_files = sorted(
    [str(p) for p in Path(image_folder).glob('*.jpg')] +
    [str(p) for p in Path(image_folder).glob('*.png')] +
    [str(p) for p in Path(image_folder).glob('*.jpeg')],
    key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x
)

current_index = 0
total = len(image_files)

while 0 <= current_index < total:
    image_path = image_files[current_index]
    print(f"Processing [{current_index+1}/{total}]: {image_path}")
    image = cv2.imread(image_path)

    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()

    if len(detections) == 0:
        cv2.putText(image, "No Detections Found", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        class_name = model.names[int(cls)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    filename = Path(image_path).name
    cv2.putText(image, filename,
                (image.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO Detection", image)

    key = cv2.waitKey(0)
    if key == ord('p'):      
        current_index = max(0, current_index - 1)
    elif key == ord('q'):      
        break
    else:
        # any key for proceeding
        current_index += 1

cv2.destroyAllWindows()
