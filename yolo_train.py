from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt')  
    model.train(
        data = r'C:\Users\TUTU\Desktop\DP\config.yaml',  # YAML path
        epochs=50,        # round
        patience=10,
        imgsz=640,         # image size
        batch=32,          # batch
        device=0,
        name = r'C:\Users\TUTU\Desktop\DP\trained_model'  
    )

if __name__ == "__main__":
    main()
