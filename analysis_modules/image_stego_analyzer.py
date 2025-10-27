 import numpy as np
 from PIL import Image
 from .base_analyzer import pad_out

 def analyze(path, dim=16):
     try:
         if path.lower().endswith((".png", ".jpg", ".jpeg")):
             with Image.open(path) as im:
                 w, h = im.size
                 ch = len(im.getbands())
             arr = np.asarray([w, h, ch], dtype=np.float32)
             arr = arr / (arr + 1.0)
             return pad_out(arr, dim)
         return np.zeros(dim, dtype=np.float32)
     except Exception:
         return np.zeros(dim, dtype=np.float32)

