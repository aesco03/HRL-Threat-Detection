 import numpy as np
 from PyPDF2 import PdfReader
 from .base_analyzer import pad_out

 def analyze(path, dim=16):
     try:
         reader = PdfReader(path)
         pages = len(reader.pages)
         xobj_count = 0
         js_flag = 1 if bool(reader.trailer.get("/Root", {}).get("/Names", {})) else 0
         obj_counts = [pages, xobj_count, js_flag]
         arr = np.asarray(obj_counts, dtype=np.float32)
         arr = arr / (arr + 1.0)
         return pad_out(arr, dim)
     except Exception:
         return np.zeros(dim, dtype=np.float32)

