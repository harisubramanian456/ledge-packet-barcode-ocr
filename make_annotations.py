"""
Build an annotation dataset from your images.

For every image the pipeline finds + decodes the barcode. When the decode succeeds (Code 128 has a checksum, so the
number is trustworthy) we know exactly where the barcode is, so we write:

  annotations/labels/<split>/<image>.txt   YOLO format:  class x_center y_center width height   (all 0-1)
  annotations/annotations.csv              split,file,truth,type,x1,y1,x2,y2,...  (truth = decoded Code128 number)
  annotations/data.yaml                    ready for ultralytics training
  annotations/needs_manual_annotation.txt  images where nothing could be decoded (label them yourself)

IMPORTANT: these are AUTO-generated labels. Open a few in Roboflow / LabelImg / CVAT and check them before training.

Usage
  python make_annotations.py "barcode OCR test"            # folder that contains train/ valid/ test/
  python make_annotations.py "barcode OCR test" --no-yolo   # skip the YOLO detector (classical detector only)
  python make_annotations.py "barcode OCR test" --max-seconds 240   # resumable: run again to continue
"""
import os, sys, glob, csv, json, time
import cv2
import numpy as np
import barcode_pipeline as B

OUT = "annotations"
CACHE = os.path.join(OUT, "_cache.jsonl")


def bars_box(img, rect):
    """Tight axis-aligned box around the BARS ONLY (not the printed digits), in original image pixels."""
    (cx, cy), (w, h), a = rect
    if w < h:
        w, h, a = h, w, a + 90
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
    rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC)
    x0, x1 = int(max(0, cx - w * 1.25 / 2)), int(cx + w * 1.25 / 2)
    y0, y1 = int(max(0, cy - h * 1.5 / 2)), int(cy + h * 1.5 / 2)
    crop = rot[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g3 = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    band = B._bar_band(g3)
    if not band:
        return None
    r0, r1 = band[0] / 3.0, band[1] / 3.0
    rows = g[int(r0):int(r1) + 1]
    if rows.shape[0] < 3:
        return None
    gx = np.abs(cv2.Sobel(rows.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    gx = cv2.GaussianBlur(gx.reshape(1, -1), (0, 0), sigmaX=2).ravel()
    xs = np.where(gx > 0.3 * gx.max())[0]
    if len(xs) < 10:
        return None
    c0, c1 = xs[0], xs[-1]
    pts = np.array([[x0 + c0, y0 + r0], [x0 + c1, y0 + r0], [x0 + c1, y0 + r1], [x0 + c0, y0 + r1]], np.float32)
    inv = cv2.invertAffineTransform(M)
    pts = cv2.transform(pts[None], inv)[0]
    x1_, y1_ = pts.min(axis=0)
    x2_, y2_ = pts.max(axis=0)
    x1_, y1_ = max(0, x1_), max(0, y1_)
    x2_, y2_ = min(W - 1, x2_), min(H - 1, y2_)
    if x2_ - x1_ < 8 or y2_ - y1_ < 4:
        return None
    return [float(x1_), float(y1_), float(x2_), float(y2_)]


def find_images(root):
    items = []
    for split in ("train", "valid", "val", "test"):
        for ext in ("png", "jpg", "jpeg", "bmp"):
            for p in glob.glob(os.path.join(root, "**", split, f"*.{ext}"), recursive=True):
                items.append(("valid" if split == "val" else split, p))
    return sorted(set(items), key=lambda t: t[1])


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    use_yolo = "--no-yolo" not in args
    max_sec = None
    if "--max-seconds" in args:
        max_sec = float(args[args.index("--max-seconds") + 1])
    root = args[0]
    items = find_images(root)
    if not items:
        print("No images found under", root); return
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line); done[r["path"]] = r
    t0 = time.time()
    for split, path in items:
        if path in done:
            continue
        if max_sec and time.time() - t0 > max_sec:
            print(f"time limit reached - run the same command again to continue ({len(done)}/{len(items)} done)")
            return
        res = B.process(path, want_ocr=False, use_yolo=use_yolo)
        rec = dict(path=path, split=split, file=os.path.basename(path), decoded=res["decoded"], type=res.get("type"),
                   detector=res.get("detector"), rect=res.get("rect"), box=None, size=None)
        img = cv2.imread(path)
        rec["size"] = [img.shape[1], img.shape[0]]
        if res["decoded"] and res.get("rect"):
            rc = res["rect"]
            rec["box"] = bars_box(img, (tuple(rc[0]), tuple(rc[1]), rc[2]))
        done[path] = rec
        open(CACHE, "a").write(json.dumps(rec) + "\n")
        print(split, rec["file"][:30], rec["decoded"], "box" if rec["box"] else "-", flush=True)

    # ------------------------------------------------------------ write outputs
    rows, missing = [], []
    for split, path in items:
        r = done[path]
        stem = os.path.splitext(r["file"])[0]
        if r["box"]:
            W, H = r["size"]; x1, y1, x2, y2 = r["box"]
            ld = os.path.join(OUT, "labels", split); os.makedirs(ld, exist_ok=True)
            with open(os.path.join(ld, stem + ".txt"), "w") as f:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
            rows.append([split, r["file"], r["decoded"], r["type"], *[round(v, 1) for v in r["box"]], W, H, r["detector"], "auto"])
        else:
            missing.append(f"{split}/{r['file']}")
    with open(os.path.join(OUT, "annotations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "truth", "type", "x1", "y1", "x2", "y2", "img_w", "img_h", "detector", "label_source"])
        w.writerows(rows)
    open(os.path.join(OUT, "needs_manual_annotation.txt"), "w").write("\n".join(missing) + "\n")
    open(os.path.join(OUT, "classes.txt"), "w").write("barcode\n")
    open(os.path.join(OUT, "data.yaml"), "w").write(
        "# ultralytics dataset file - run prepare_dataset.py first to create dataset/images + dataset/labels\n"
        "path: dataset\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: barcode\n")
    print(f"\nlabelled {len(rows)} images, {len(missing)} need manual annotation -> {OUT}/")


if __name__ == "__main__":
    main()
