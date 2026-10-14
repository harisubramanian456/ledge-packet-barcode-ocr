"""
Arrange images + labels into the folder layout YOLO training expects.

  python prepare_dataset.py "barcode OCR test"

creates
  dataset/images/{train,valid,test}/*.png      (copied from your folder - only images that have a label)
  dataset/labels/{train,valid,test}/*.txt      (from annotations/labels)
"""
import os, sys, glob, shutil

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    root = sys.argv[1]; n = 0
    for lab in glob.glob(os.path.join("annotations", "labels", "*", "*.txt")):
        split = os.path.basename(os.path.dirname(lab)); stem = os.path.splitext(os.path.basename(lab))[0]
        img = None
        for ext in ("png", "jpg", "jpeg", "bmp"):
            hit = glob.glob(os.path.join(root, "**", split, stem + "." + ext), recursive=True)
            if hit: img = hit[0]; break
        if img is None:
            print("image not found for", split, stem); continue
        for kind, src in (("images", img), ("labels", lab)):
            d = os.path.join("dataset", kind, split); os.makedirs(d, exist_ok=True)
            shutil.copy2(src, d)
        n += 1
    print(f"prepared {n} labelled images in ./dataset")

if __name__ == "__main__":
    main()
