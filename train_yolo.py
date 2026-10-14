"""
Fine-tune the pretrained YOLOv8s barcode detector on YOUR packets (after prepare_dataset.py).

  python train_yolo.py                # 60 epochs, imgsz 1280
  python train_yolo.py --epochs 100

Result: runs/barcode_ft/weights/best.pt   ->  use it by setting  YOLO_FILE / path in barcode_pipeline.py
(or copy best.pt to models/YOLOV8s_Barcode_Detection.pt).
"""
import os, sys
import barcode_pipeline as B

def main():
    from ultralytics import YOLO
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 60
    if B.load_yolo() is None:
        raise SystemExit("could not load the pretrained weights (check internet / pip install ultralytics huggingface_hub)")
    weights = os.path.join(B.YOLO_DIR, B.YOLO_FILE)
    data = os.path.abspath("dataset")
    yaml_path = os.path.join(data, "data_abs.yaml")                     # absolute paths avoid ultralytics path surprises
    open(yaml_path, "w").write(f"path: {data}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: barcode\n")
    model = YOLO(weights)
    model.train(data=yaml_path, epochs=epochs, imgsz=1280, batch=4, patience=20,
                project="runs", name="barcode_ft", degrees=180, flipud=0.5, fliplr=0.5, mosaic=0.5)
    print("done -> runs/barcode_ft/weights/best.pt")

if __name__ == "__main__":
    main()
