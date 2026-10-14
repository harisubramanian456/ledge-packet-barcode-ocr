"""
Evaluate the full pipeline (YOLO detect -> decode Code128 -> OCR -> compare) against annotations/annotations.csv.

What is better than the old evaluate script
  * EXACT comparison. The old script accepted "close" strings (similarity > 0.9): 4233285 vs 41233285 scores 0.93, so a
    missing digit was counted as correct. Packet numbers must match exactly.
  * Reports the three things separately:  decode accuracy, OCR accuracy, and the final verdict (decode == OCR).
  * Finds the dangerous case: verdict says MATCHED but the number is wrong (should never happen).
  * Optional detector evaluation (--det-eval): YOLO box vs annotated box, IoU >= 0.5.
  * Resumable (--max-seconds) and writes evaluation_report.csv

Usage
  python evaluate.py --images "barcode OCR test"
  python evaluate.py --images "barcode OCR test" --no-yolo --max-seconds 240      # run again to continue
  python evaluate.py --images "barcode OCR test" --det-eval                          # needs ultralytics + weights
"""
import os, sys, csv, glob, json, time
import cv2
import barcode_pipeline as B

ANN = os.path.join("annotations", "annotations.csv")
CACHE = "evaluation_cache.jsonl"


def load_truth(path=ANN):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_image(root, split, name):
    hits = glob.glob(os.path.join(root, "**", split, name), recursive=True)
    return hits[0] if hits else None


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    args = sys.argv[1:]
    if "--images" not in args:
        print(__doc__); return
    root = args[args.index("--images") + 1]
    use_yolo = "--no-yolo" not in args
    det_eval = "--det-eval" in args
    max_sec = float(args[args.index("--max-seconds") + 1]) if "--max-seconds" in args else None

    truth_rows = load_truth()
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line); done[(r["split"], r["file"])] = r
    t0 = time.time()
    for row in truth_rows:
        key = (row["split"], row["file"])
        if key in done:
            continue
        if max_sec and time.time() - t0 > max_sec:
            print(f"time limit reached - run again to continue ({len(done)}/{len(truth_rows)})"); return
        path = find_image(root, row["split"], row["file"])
        rec = dict(split=row["split"], file=row["file"], truth=row["truth"], decoded=None, ocr=None, detector=None,
                   det_iou=None)
        if path:
            res = B.process(path, use_yolo=use_yolo)
            rec.update(decoded=res["decoded"], ocr=res["ocr"], detector=res.get("detector"))
            if det_eval:
                img = cv2.imread(path); boxes = B.yolo_boxes(img)
                gt = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
                rec["det_iou"] = max([iou(b, gt) for _, b in boxes], default=0.0)
        else:
            rec["error"] = "image not found"
        done[key] = rec
        open(CACHE, "a").write(json.dumps(rec) + "\n")
        print(row["split"], row["file"][:28], "truth", row["truth"], "| decoded", rec["decoded"], "| ocr", rec["ocr"], flush=True)

    # ------------------------------------------------------------------ report
    per = {}
    out_rows, bad_match = [], []
    for row in truth_rows:
        r = done[(row["split"], row["file"])]
        s = per.setdefault(r["split"], dict(n=0, dec_ok=0, ocr_ok=0, matched=0, review=0, notfound=0, det_ok=0, det_n=0))
        s["n"] += 1
        dec_ok = r["decoded"] == r["truth"]
        ocr_ok = r["ocr"] == r["truth"]
        if r["decoded"] is None:
            verdict = "NOT FOUND"; s["notfound"] += 1
        elif r["ocr"] == r["decoded"]:
            verdict = "MATCHED"; s["matched"] += 1
            if not dec_ok:
                bad_match.append(r)
        else:
            verdict = "MISMATCH - CHECK"; s["review"] += 1
        s["dec_ok"] += dec_ok; s["ocr_ok"] += ocr_ok
        if r.get("det_iou") is not None:
            s["det_n"] += 1; s["det_ok"] += r["det_iou"] >= 0.5
        out_rows.append([r["split"], r["file"], r["truth"], r["decoded"], r["ocr"], verdict, r["detector"], r.get("det_iou")])

    print("\n================ RESULTS (exact match) ================")
    tot = dict(n=0, dec_ok=0, ocr_ok=0, matched=0, review=0, notfound=0)
    for split, s in per.items():
        print(f"{split.upper():6s} n={s['n']:3d} | decode {s['dec_ok']}/{s['n']} | OCR {s['ocr_ok']}/{s['n']} | "
              f"MATCHED {s['matched']} | CHECK {s['review']} | NOT FOUND {s['notfound']}"
              + (f" | YOLO det IoU>=0.5: {s['det_ok']}/{s['det_n']}" if s["det_n"] else ""))
        for k in tot: tot[k] += s[k]
    print(f"ALL    n={tot['n']:3d} | decode {tot['dec_ok']}/{tot['n']} ({100*tot['dec_ok']/max(1,tot['n']):.1f}%) | "
          f"OCR {tot['ocr_ok']}/{tot['n']} ({100*tot['ocr_ok']/max(1,tot['n']):.1f}%) | "
          f"MATCHED {tot['matched']} | CHECK {tot['review']} | NOT FOUND {tot['notfound']}")
    print("WRONG-BUT-MATCHED (must be 0):", len(bad_match))
    print("\n--- failures ---")
    for r in out_rows:
        if r[5] != "MATCHED":
            print(f"{r[0]}/{r[1][:30]} | truth={r[2]} decoded={r[3]} ocr={r[4]} -> {r[5]}")
    with open("evaluation_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "truth", "decoded", "ocr", "verdict", "detector", "det_iou"])
        w.writerows(out_rows)
    print("\nsaved evaluation_report.csv")


if __name__ == "__main__":
    main()
