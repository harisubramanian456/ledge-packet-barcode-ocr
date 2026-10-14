"""
Barcode  ->  Detect  ->  Decode  ->  OCR  ->  Compare   (pledge packet tare-capture project)

Usage
  python barcode_ocr_pipeline.py run   <image.png|folder> [--out outdir]   # full pipeline + annotated panel
  python barcode_ocr_pipeline.py demo  <number>                            # generate a Code128 barcode + show the maths
Requires: opencv-python, numpy, pyzbar (+ libzbar0), pytesseract (+ tesseract-ocr), python-barcode, pillow
"""
import cv2, numpy as np, glob, os, re, json, sys, time
from collections import Counter
from pyzbar import pyzbar
import pytesseract
import shutil, platform
if platform.system() == "Windows" and not shutil.which("tesseract"):
    for _p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

SYMS=[pyzbar.ZBarSymbol.CODE128, pyzbar.ZBarSymbol.CODE39, pyzbar.ZBarSymbol.EAN13, pyzbar.ZBarSymbol.I25]

# ============ STEP 1 : DETECT (find candidate barcode regions) ============
# Several threshold settings are combined, so the result does not depend on one global setting
# (e.g. a white side panel in the picture no longer hides the barcode).
def _mask_cands(img, e_pct, coh_t, cancel_t, win=15, min_area=2000, min_lo=20, min_hi=70):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0); gy = cv2.Scharr(g, cv2.CV_32F, 0, 1); k = (win, win)
    Jxx = cv2.boxFilter(gx*gx, -1, k); Jyy = cv2.boxFilter(gy*gy, -1, k); Jxy = cv2.boxFilter(gx*gy, -1, k)
    en = Jxx + Jyy; coh = np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2)/(en+1e-6)
    th = 0.5*np.arctan2(2*Jxy, Jxx-Jyy); gd = gx*np.cos(th) + gy*np.sin(th)
    can = np.abs(cv2.boxFilter(gd, -1, k))/(cv2.boxFilter(np.abs(gd), -1, k)+1e-6)
    m = ((coh > coh_t) & (can < cancel_t) & (en > np.percentile(en, e_pct))).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area or a > 60000: continue
        r = cv2.minAreaRect(c); w, h = r[1]; lo, hi = min(w, h), max(w, h)
        if lo < min_lo or lo > 170 or hi < min_hi or hi > 650: continue
        if hi/lo < 1.3 or hi/lo > 8: continue
        out.append((a*a/(w*h), r))
    return out

def detect_candidates(img, max_c=24):
    allc = []
    for e_pct, coh_t, can_t in ((75, .62, .42), (60, .62, .42), (88, .70, .35), (50, .55, .50)):
        allc += _mask_cands(img, e_pct, coh_t, can_t)
    allc.sort(key=lambda x: -x[0]); out = []
    for s, r in allc:
        if all(np.hypot(r[0][0]-o[0][0], r[0][1]-o[0][1]) > 25 for o in out): out.append(r)
        if len(out) >= max_c: break
    return out

def straighten(img, rect, pad_w=1.25, pad_h=1.5):
    (cx, cy), (w, h), a = rect
    if w < h: w, h, a = h, w, a+90
    M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    x0, x1 = int(max(0, cx-w*pad_w/2)), int(cx+w*pad_w/2); y0, y1 = int(max(0, cy-h*pad_h/2)), int(cy+h*pad_h/2)
    return rot[y0:y1, x0:x1]

# ---------- YOLO barcode detector (pretrained, Hugging Face: Piero2411/YOLOV8s-Barcode-Detection) ----------
# Used as the FIRST choice. If ultralytics / the weights are not available, the classical detector above is used.
YOLO_REPO = "Piero2411/YOLOV8s-Barcode-Detection"
YOLO_FILE = "YOLOV8s_Barcode_Detection.pt"
YOLO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "models")
_YOLO = {"model": None, "failed": False}

def load_yolo():
    """Load (and on first use download, ~22 MB) the pretrained YOLOv8s barcode model. Returns None if unavailable."""
    if _YOLO["model"] is not None: return _YOLO["model"]
    if _YOLO["failed"]: return None
    try:
        from ultralytics import YOLO
        os.makedirs(YOLO_DIR, exist_ok=True)
        path = os.path.join(YOLO_DIR, YOLO_FILE)
        if not os.path.exists(path):
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILE, local_dir=YOLO_DIR)
            except ImportError:
                import urllib.request
                urllib.request.urlretrieve(f"https://huggingface.co/{YOLO_REPO}/resolve/main/{YOLO_FILE}", path)
        _YOLO["model"] = YOLO(path)
    except Exception as e:                                   # no internet / no ultralytics -> classical fallback
        print(f"[YOLO unavailable -> using classical detector] {type(e).__name__}: {str(e)[:120]}")
        _YOLO["failed"] = True
        return None
    return _YOLO["model"]

def yolo_boxes(img, conf=0.10, imgsz=1280):
    """returns [(confidence, (x1, y1, x2, y2))] for barcode detections (QR codes are skipped), best first"""
    model = load_yolo()
    if model is None: return []
    r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for xyxy, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        if "qr" in str(r.names.get(int(k), "")).lower(): continue
        out.append((float(c), tuple(float(v) for v in xyxy)))
    return sorted(out, key=lambda t: -t[0])

def rect_from_box(img, box, pad=0.12):
    """YOLO gives an axis-aligned box. Rebuild the tilted rectangle of the bars inside it (needed for deskew + OCR)."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = box; bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad)+6, int(bh*pad)+6
    X1, Y1, X2, Y2 = max(0, int(x1)-px), max(0, int(y1)-py), min(W, int(x2)+px), min(H, int(y2)+py)
    crop = img[Y1:Y2, X1:X2]; rects = []
    cx, cy = (x1+x2)/2, (y1+y2)/2
    if crop.size:
        found = []
        for e_pct, coh_t, can_t in ((75, .62, .42), (60, .62, .42), (50, .55, .50)):
            found += _mask_cands(crop, e_pct, coh_t, can_t, min_area=500, min_lo=8, min_hi=40)
        found.sort(key=lambda t: -t[0])
        for _, r in found[:2]:
            rects.append(((r[0][0]+X1, r[0][1]+Y1), r[1], r[2]))
    L, Sd = max(bw, bh), min(bw, bh)                          # fallbacks: the box itself, upright or sideways
    rects.append(((cx, cy), (bw, bh), 0.0)); rects.append(((cx, cy), (bh, bw), 0.0))
    return rects

def detect_candidates_yolo(img, max_boxes=4):
    rects = []
    for conf, box in yolo_boxes(img)[:max_boxes]:
        rects += rect_from_box(img, box)
    return rects

# ============ STEP 2 : DECODE ============
# decode_fast : binarise + upscale (works for clean barcodes)
# decode_deep : fine-deskew to 0.5 deg, bar-band only, sharpen, several scales (blurred / steep barcodes)
def _bin(g):
    yield cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    yield cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 6)
    yield cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

def decode_fast(crop):
    g0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for up in (2, 3):
        g = cv2.resize(g0, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        for bw in _bin(g):
            for ang in (0, 3, -3):
                m = bw if ang == 0 else cv2.warpAffine(bw, cv2.getRotationMatrix2D((bw.shape[1]/2, bw.shape[0]/2), ang, 1), (bw.shape[1], bw.shape[0]), borderValue=255)
                r = [x for x in pyzbar.decode(m, symbols=SYMS) if x.data]
                if r: return r[0].data.decode(), r[0].type
    return None, None

def _bar_band(g):
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=1)
    e = cv2.GaussianBlur(e.reshape(-1, 1), (1, 0), 1.5).ravel(); idx = np.where(e > 0.5*e.max())[0]
    if len(idx) < 4: return None
    run = max(np.split(idx, np.where(np.diff(idx) > 2)[0]+1), key=len); return run[0], run[-1]

def _rot(g, a):
    h, w = g.shape
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w/2, h/2), a, 1), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def decode_deep(crop):
    """fine deskew + bar-band only + several scales / sharpness levels"""
    g0 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for flip in (False, True):
        g = cv2.rotate(g0, cv2.ROTATE_180) if flip else g0
        b = _bar_band(g)
        if not b or b[1]-b[0] < 8: continue
        r0, r1 = b[0]+2, b[1]-2
        best = (-1, 0)
        for a in np.arange(-10, 10.1, 0.5):
            s = np.abs(np.diff(_rot(g, a)[r0:r1].mean(axis=0))).sum()
            if s > best[0]: best = (s, a)
        gr = _rot(g, best[1])[max(0, r0-4):r1+4]
        # keep quiet zone: pad with local background
        gr = cv2.copyMakeBorder(gr, 20, 20, 60, 60, cv2.BORDER_REPLICATE)
        for sig in (0, 1.0, 2.0):
            gs = gr if sig == 0 else cv2.addWeighted(gr, 1+sig, cv2.GaussianBlur(gr, (0, 0), 3), -sig, 0)
            for scale in (1.0, 0.66, 1.5):
                gg = cv2.resize(gs, None, fx=scale, fy=1.0, interpolation=cv2.INTER_CUBIC) if scale != 1 else gs
                for bw in _bin(gg):
                    for rr in (None, 'row'):
                        m = bw if rr is None else np.tile(bw[bw.shape[0]//2], (bw.shape[0], 1))
                        r = [x for x in pyzbar.decode(m, symbols=SYMS) if x.data]
                        if r: return r[0].data.decode(), r[0].type
    return None, None

# ============ STEP 3 : OCR the printed number (independent of the decoder) ============
# (legacy readers below are only a fallback; the v3 digit reader is defined right after this block)
# Two readers vote:
#   (W) whole straightened crop      - default; works when the box already contains the printed digits
#   (S) strips above/below the bars  - used when it gets >= 2 agreeing readings (box covers only the bars)
# The decoded barcode value is never looked at here.
def _read_digits(img_bin, psm):
    if img_bin.shape[0] < 20 or img_bin.shape[1] < 20:        # too small for Tesseract
        return []
    try:
        t = pytesseract.image_to_string(img_bin, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789 ")
    except Exception:                                         # a bad strip must never crash the pipeline
        return []
    out = []
    for line in t.splitlines():                               # '361 7056' -> '3617056'
        out += re.findall(r"\d{5,13}", re.sub(r"\s+", "", line))
    return out

def _read_digits_conf(img_bin, psm, min_conf=10):
    """like _read_digits, but drops readings Tesseract itself is unsure about (upside-down text gives ~0 confidence)"""
    if img_bin.shape[0] < 20 or img_bin.shape[1] < 20: return []
    try:
        d = pytesseract.image_to_data(img_bin, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789 ",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    lines = {}
    for i, t in enumerate(d["text"]):
        t = str(t).strip()
        try: c = float(d["conf"][i])
        except Exception: c = -1
        if t.isdigit() and c >= 0:
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((t, c))
    out = []
    for ws in lines.values():
        txt = "".join(w for w, _ in ws); conf = sum(c for _, c in ws) / len(ws)
        if 5 <= len(txt) <= 13 and conf >= min_conf: out.append(txt)
    return out

def _top(votes):
    return max(votes.items(), key=lambda kv: (kv[1], len(kv[0])))[0] if votes else None

def ocr_whole(img, rect):
    crop = straighten(img, rect)
    votes = {}
    if crop.size == 0: return votes
    g0 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for flip in (False, True):                                # printed text may be upside-down
        g = cv2.rotate(g0, cv2.ROTATE_180) if flip else g0
        g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        for pre in (g, cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            for psm in (6, 11):
                for v in _read_digits(pre, psm): votes[v] = votes.get(v, 0) + 1
    return votes

def ocr_strips(img, rect):
    """returns (votes_confident, votes_raw) for the text strips above/below the bars"""
    (cx, cy), (w, h), a = rect
    lw, lh = max(w, h), max(1.0, min(w, h))
    roi = straighten(img, rect, pad_w=1.35, pad_h=max(3.6, 0.75 * lw / lh))   # tall enough to reach the digits
    conf_votes, raw_votes = {}, {}
    if roi.size == 0: return conf_votes, raw_votes
    g0 = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for flip in (False, True):
        g = cv2.rotate(g0, cv2.ROTATE_180) if flip else g0
        band = _bar_band(g); strips = []
        if band:
            bar_h = max(10, band[1] - band[0]); reach = int(2.6 * bar_h)      # only the area right next to the bars
            if band[0] > 40: strips.append(g[max(0, band[0]-3-reach):band[0]-3])
            if g.shape[0] - band[1] > 40: strips.append(g[band[1]+4:band[1]+4+reach])
        for st in strips:
            if st.shape[0] < 30: continue
            for sc in (1.0, 1.4):
                s_ = cv2.resize(st, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc != 1 else st
                s_ = cv2.copyMakeBorder(s_, 25, 25, 25, 25, cv2.BORDER_REPLICATE)
                otsu = cv2.threshold(s_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                for pre in (s_, otsu, cv2.erode(otsu, np.ones((2, 2), np.uint8))):   # erode thickens the thin digit "1"
                    for v in _read_digits_conf(pre, 7): conf_votes[v] = conf_votes.get(v, 0) + 1
                    for v in _read_digits(pre, 7): raw_votes[v] = raw_votes.get(v, 0) + 1
    return conf_votes, raw_votes

# ============ STEP 3 (v3) : OCR the printed number  -  segment digits, then read each digit ============
# Why the old OCR failed: Tesseract reads a whole line and silently DROPS the thin digit "1" (41233285 -> 4233285).
# New method:  1) find the digit line next to the bars (fine deskew, upright or upside-down)
#              2) cut it into single digits (connected components)  -> the digit COUNT can no longer be wrong
#              3) read every digit with a template matcher (+ Tesseract on the cleaned line as second opinion)
# The decoded barcode value is never looked at here, so the comparison stays an independent check.
import base64, io, zlib

_TEMPLATES = {"X": None, "Y": None}
TEMPLATE_B64 = "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"
TPL_H = TPL_W = 16

def load_templates():
    if _TEMPLATES["X"] is None and TEMPLATE_B64 and not TEMPLATE_B64.startswith("@@"):
        d = np.load(io.BytesIO(zlib.decompress(base64.b64decode(TEMPLATE_B64))))
        n = int(d["n"]); bits = np.unpackbits(d["X"])[: n * TPL_H * TPL_W]
        _TEMPLATES["X"] = bits.reshape(n, TPL_H * TPL_W).astype(np.float32)
        _TEMPLATES["Y"] = np.array(list(d["Y"].tobytes().decode()))
    return _TEMPLATES["X"], _TEMPLATES["Y"]

def _rot_small(g, a):
    h, w = g.shape
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w/2, h/2), a, 1), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _fine_deskew(g, span=14, step=1.0):
    """the decoder's rectangle can be a few degrees off: find the angle where the bars are exactly vertical"""
    best = (-1, 0.0)
    for a in np.arange(-span, span + 0.01, step):
        r = _rot_small(g, a); b = _bar_band(r)
        if not b or b[1] - b[0] < 8: continue
        sc = np.abs(np.diff(r[b[0] + 2:b[1] - 2].mean(axis=0))).sum()
        if sc > best[0]: best = (sc, float(a))
    return best[1]

def _segment_digits(strip, bar_h):
    s = cv2.GaussianBlur(cv2.copyMakeBorder(strip, 6, 6, 6, 6, cv2.BORDER_REPLICATE), (0, 0), 1.0)
    bw = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8); boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        # thin "1" is only ~0.1 x height wide -> must not be filtered out
        if h < 0.35*bar_h or h > 1.5*bar_h or w > 1.6*h or w < 0.05*h or a < 0.04*h*h: continue
        boxes.append([int(x), int(y), int(w), int(h)])
    if len(boxes) < 3: return [], bw
    H = np.median([b[3] for b in boxes]); keep = [b for b in boxes if 0.7*H <= b[3] <= 1.3*H]
    if not keep: return [], bw
    cy = np.median([b[1] + b[3]/2 for b in keep]); keep = [b for b in keep if abs(b[1] + b[3]/2 - cy) < 0.4*H]
    if not keep: return [], bw
    keep.sort(key=lambda b: b[0]); runs = [[keep[0]]]
    for b in keep[1:]:
        if b[0] - (runs[-1][-1][0] + runs[-1][-1][2]) > 1.6*H: runs.append([b])
        else: runs[-1].append(b)
    keep = max(runs, key=len)
    wn = np.median([b[2] for b in keep if b[2] > 0.45*b[3]] or [0.6*H]); out = []
    for x, y, w, h in keep:                                   # two digits stuck together -> split
        k = int(round(w/wn)) if w > 1.45*wn else 1
        if k > 1: out += [[x + int(j*w/k), y, int(w/k), h] for j in range(k)]
        else: out.append([x, y, w, h])
    return out, bw

def _digit_feature(bw, box):
    x, y, w, h = box; c = bw[y:y+h, x:x+w]; s = TPL_H / h
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TPL_H, TPL_W), np.float32); ww = min(TPL_W, c.shape[1]); x0 = (TPL_W - ww)//2
    canvas[:, x0:x0+ww] = c[:, :ww] / 255.0
    return (canvas > 0.5).astype(np.float32).ravel()

def _line_image(bw, boxes):
    x0 = min(b[0] for b in boxes); x1 = max(b[0]+b[2] for b in boxes); y0 = min(b[1] for b in boxes); y1 = max(b[1]+b[3] for b in boxes)
    pad = int(0.4*(y1-y0)); c = 255 - bw[max(0, y0-pad):y1+pad, max(0, x0-pad):x1+pad]
    c = cv2.GaussianBlur(cv2.resize(c, None, fx=70/(y1-y0), fy=70/(y1-y0), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    return cv2.copyMakeBorder(c, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

def digit_lines(roi):
    """candidate digit lines of a straightened ROI: [(boxes, bw, upside_down)]"""
    g0 = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC); out = []
    for flip in (False, True):                                # printed number can be upside-down
        g = cv2.rotate(g0, cv2.ROTATE_180) if flip else g0
        g = _rot_small(g, _fine_deskew(g)); band = _bar_band(g)
        if not band: continue
        bar_h = max(10, band[1] - band[0]); strip = g[band[1]+3: band[1]+3+int(2.2*bar_h)]
        if strip.shape[0] < 30: continue
        boxes, bw = _segment_digits(strip, bar_h)
        if len(boxes) >= 4: out.append((boxes, bw, flip))
    return out

def read_digit_line(boxes, bw, TX=None, TY=None, dist_ok=20.0):
    """returns (text, info).  A = template reader, B = Tesseract on the cleaned line (second opinion)."""
    n = len(boxes); feats = [_digit_feature(bw, b) for b in boxes]
    A = ""; dists = []
    if TX is not None:
        for f in feats:
            d = ((TX - f) ** 2).sum(1); k = int(d.argmin()); A += TY[k]; dists.append(float(d[k]))
    B = ""
    try:
        B = pytesseract.image_to_string(_line_image(bw, boxes), config="--psm 7 -c tessedit_char_whitelist=0123456789").strip().replace(" ", "")
    except Exception:
        pass
    B_ok = len(B) == n and B.isdigit()                        # Tesseract only counts if it kept every digit
    if A:
        txt = "".join(A[i] if (dists[i] <= dist_ok or not B_ok) else B[i] for i in range(n))   # unsure digit -> take Tesseract's
    else:
        txt = B if B_ok else ""
    info = dict(n=n, templates=A or None, tesseract=B or None, dist_max=max(dists) if dists else None,
                agree=bool(A and B_ok and A == B), confident=bool(dists and max(dists) <= dist_ok))
    return txt, info

def ocr_digits(img, rect, TX=None, TY=None):
    (cx, cy), (w, h), a = rect; lw, lh = max(w, h), max(1.0, min(w, h))
    roi = straighten(img, rect, pad_w=1.35, pad_h=max(3.6, 0.75*lw/lh))
    if roi.size == 0: return None, {}
    if TX is None: TX, TY = load_templates()
    best = None
    for boxes, bw, flip in digit_lines(roi):
        feats = [_digit_feature(bw, b) for b in boxes]
        score = (np.mean([((TX - f) ** 2).sum(1).min() for f in feats]) if TX is not None else -len(boxes))   # right-way-up digits match best
        if best is None or score < best[0]: best = (score, boxes, bw)
    if best is None: return None, {}
    txt, info = read_digit_line(best[1], best[2], TX, TY)
    return (txt or None), info

def ocr_number_legacy(img, rect):
    """returns (best_text_or_None, {candidate: votes})"""
    Sc, Sr = ocr_strips(img, rect); W = ocr_whole(img, rect)
    sc, sr, wv = _top(Sc), _top(Sr), _top(W)
    if sc and Sc[sc] >= 2:       pick = sc          # confident strip reading
    elif sr and Sr[sr] >= 5:     pick = sr          # strong unfiltered strip reading (e.g. thin "1" cases)
    else:                        pick = wv or sr or sc
    allv = dict(W)
    for d in (Sc, Sr):
        for k, v in d.items(): allv[k] = max(allv.get(k, 0), v) if d is Sr else allv.get(k, 0) + v
    return pick, dict(sorted(allv.items(), key=lambda kv: -kv[1])[:3])


def ocr_number(img, rect, return_info=False):
    """returns (best_text_or_None, {candidate: votes}).  v3 digit reader first; the old whole-line / strip readers only
    run as a fallback when no digit line could be cut out (e.g. barcode without printed digits)."""
    txt, info = ocr_digits(img, rect)
    if txt:
        out = (txt, {txt: 1})
        info = dict(info, method="digits")
    else:
        out = ocr_number_legacy(img, rect); info = dict(method="legacy")
    return (out + (info,)) if return_info else out

def process(path, want_ocr=True, use_yolo=True):
    img = cv2.imread(path)
    res = dict(file=os.path.basename(path), detected=0, decoded=None, ocr=None, match=None, detector=None)
    if img is None:
        res["error"] = "cannot read image"; return res
    groups = []
    if use_yolo:
        groups.append(("yolo", detect_candidates_yolo))
    groups.append(("classical", detect_candidates))
    found = None
    for name, det in groups:
        cands = det(img)
        res["n_candidates"] = len(cands)
        for tier in (decode_fast, decode_deep):             # cheap pass on all candidates, then the deep pass
            for rect in cands:
                crop = straighten(img, rect)
                if crop.size == 0: continue
                val, typ = tier(crop)
                if val and len(val) >= 5:
                    found = (rect, crop, val, typ, name); break
            if found: break
        if found: break
    if found:
        rect, crop, val, typ, name = found
        res.update(detected=1, decoded=val, type=typ, detector=name,
                   rect=[list(map(float, rect[0])), list(map(float, rect[1])), float(rect[2])])
        if want_ocr:
            res["ocr"], res["ocr_votes"], res["ocr_info"] = ocr_number(img, rect, return_info=True)
            if res["ocr"]:
                res["match"] = res["ocr"] == res["decoded"]
    return res

# ============ BARCODE GENERATION (Code 128) : the maths, shown explicitly ============
def code128_explain(digits: str):
    """Code Set C packs 2 digits per symbol. Checksum = (start + sum(position*value)) mod 103."""
    assert digits.isdigit() and len(digits) % 2 == 0, "pure Code-C demo needs an even number of digits"
    values = [int(digits[i:i+2]) for i in range(0, len(digits), 2)]
    START_C, STOP = 105, 106
    total = START_C + sum((i+1)*v for i, v in enumerate(values))
    check = total % 103
    seq = [START_C] + values + [check, STOP]
    modules = 11*(len(seq)-1) + 13                       # every symbol = 11 modules, STOP = 13
    print(f"digits        : {digits}")
    print(f"pairs -> value: {values}")
    print(f"weighted sum  : {START_C} + " + " + ".join(f"{i+1}x{v}" for i, v in enumerate(values)) + f" = {total}")
    print(f"checksum      : {total} mod 103 = {check}")
    print(f"symbol stream : {seq}   ({modules} modules + 10-module quiet zone each side)")
    return seq

def generate_barcode(number: str, path="barcode.png"):
    import barcode
    from barcode.writer import ImageWriter
    code = barcode.get("code128", number, writer=ImageWriter())
    fn = code.save(path.replace(".png", ""), options=dict(module_width=0.4, module_height=18, quiet_zone=6.5, font_size=10, dpi=300))
    return fn

# ============ ANNOTATED OUTPUT PANEL (like packet.png) ============
def annotate(img, res, out_path):
    from datetime import datetime
    vis = img.copy()
    if res.get('rect'):
        r = res['rect']; box = cv2.boxPoints((tuple(r[0]), tuple(r[1]), r[2])).astype(int)
        cv2.polylines(vis, [box], True, (200, 255, 80), 6)
    H = vis.shape[0]; s = 1600 / H
    vis = cv2.resize(vis, None, fx=s, fy=s)
    panel = np.full((vis.shape[0], 700, 3), (243, 240, 236), np.uint8)
    ok = res.get('match')
    status = "NOT FOUND" if not res['decoded'] else ("BARCODE MATCHED" if ok else "MISMATCH - CHECK")
    col = (90, 140, 30) if ok else (30, 30, 200)
    cv2.rectangle(panel, (0, 0), (700, 150), (60, 30, 20), -1)
    cv2.putText(panel, "PLEDGE TARE CAPTURE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
    cv2.putText(panel, "Packet Summary", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    items = [("PACKET NUMBER (OCR)", res.get('ocr') or "-", (40, 30, 20)),
             ("BARCODE CHECK", status, col),
             ("BARCODE DIGITS (DECODED)", res.get('decoded') or "-", (40, 30, 20)),
             ("CAPTURE DATE", datetime.now().strftime("%d-%m-%Y"), (40, 30, 20)),
             ("CAPTURE TIME", datetime.now().strftime("%I:%M:%S %p"), (40, 30, 20))]
    y = 200
    for lab, val, c in items:
        cv2.rectangle(panel, (30, y), (670, y+150), (255, 255, 255), -1)
        cv2.rectangle(panel, (30, y), (40, y+150), (60, 160, 210), -1)
        cv2.putText(panel, lab, (65, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 95, 85), 2)
        cv2.putText(panel, str(val), (65, y+110), cv2.FONT_HERSHEY_SIMPLEX, 1.6 if len(str(val)) < 12 else 1.3, c, 3)
        y += 175
    cv2.imwrite(out_path, np.hstack([vis, panel]))

def run_on(paths, outdir, use_yolo=True):
    import csv
    os.makedirs(outdir, exist_ok=True); rows = []
    for p in paths:
        r = process(p, use_yolo=use_yolo); img = cv2.imread(p)
        annotate(img, r, os.path.join(outdir, "out_" + os.path.basename(p)[:40] + ".png"))
        rows.append({k: r.get(k) for k in ("file", "decoded", "type", "ocr", "match", "detector")}); print(rows[-1])
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

USAGE = """
Usage:
  python barcode_ocr_pipeline.py <image.png>                      check ONE image
  python barcode_ocr_pipeline.py <folder>                         check every .png/.jpg in a folder
  python barcode_ocr_pipeline.py run <image|folder> [--out DIR] [--no-yolo]   choose output folder / skip YOLO
  python barcode_ocr_pipeline.py demo <number>                    generate a Code128 barcode
Example:
  python barcode_ocr_pipeline.py "barcode OCR test/test" --out output
"""

def collect(tgt):
    if os.path.isdir(tgt):
        ps = []
        for ext in ("png", "jpg", "jpeg", "bmp"):
            ps += glob.glob(os.path.join(tgt, "**", "*." + ext), recursive=True)
        return sorted(ps)
    return [tgt]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    out = "output"
    use_yolo = "--no-yolo" not in args
    if "--no-yolo" in args: args.remove("--no-yolo")
    if "--out" in args:
        k = args.index("--out"); out = args[k+1]; del args[k:k+2]
    if not args:
        print(USAGE); sys.exit(0)
    if args[0] == "demo":
        if len(args) < 2: print("give a number, e.g.  demo 41233285"); sys.exit(1)
        num = args[1]
        if len(num) % 2 == 0: code128_explain(num)
        print("saved", generate_barcode(num)); sys.exit(0)
    if args[0] == "run": args = args[1:]
    if not args: print(USAGE); sys.exit(1)
    paths = collect(args[0])
    if not paths or not os.path.exists(paths[0]):
        print("Path not found / no images:", args[0]); sys.exit(1)
    run_on(paths, out, use_yolo=use_yolo)
