import os
import numpy as np
from PyPDF2 import PdfReader
from PIL import Image

def extract_file_features(path, feature_dim=32):
    ext = os.path.splitext(path.lower())[1]
    is_pdf = 1 if ext == ".pdf" else 0
    is_img = 1 if ext in [".png", ".jpg", ".jpeg"] else 0
    size_bytes = 0
    try:
        size_bytes = os.path.getsize(path)
    except Exception:
        size_bytes = 0
    size_kb = float(size_bytes) / 1024.0
    feats = [is_pdf, is_img, size_kb]
    if is_pdf:
        pages = 0
        img_count = 0
        js_flag = 0
        try:
            reader = PdfReader(path)
            pages = len(reader.pages)
            root = reader.trailer.get("/Root", {})
            js_flag = 1 if bool(root.get("/Names", {})) else 0
            for p in reader.pages:
                res = p.get("/Resources")
                if res and res.get("/XObject"):
                    xobj = res["/XObject"].get_object()
                    for o in xobj.values():
                        try:
                            if o.get("/Subtype") == "/Image":
                                img_count += 1
                        except Exception:
                            pass
        except Exception:
            pages = 0
            img_count = 0
            js_flag = 0
        feats.extend([pages, img_count, js_flag])
        feats.extend([0.0, 0.0, 0.0])
    elif is_img:
        w = h = ch = 0
        try:
            with Image.open(path) as im:
                w, h = im.size
                ch = len(im.getbands())
        except Exception:
            w = h = ch = 0
        feats.extend([0.0, 0.0, 0.0])
        feats.extend([w, h, ch])
    else:
        feats.extend([0.0, 0.0, 0.0])
        feats.extend([0.0, 0.0, 0.0])
    arr = np.asarray(feats, dtype=np.float32)
    arr = arr / (arr + 1.0)
    out = np.zeros(feature_dim, dtype=np.float32)
    n = min(feature_dim, arr.shape[0])
    out[:n] = arr[:n]
    return out
