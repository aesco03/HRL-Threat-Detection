 import numpy as np
 from PyPDF2 import PdfReader
 from .base_analyzer import pad_out

 def analyze(path, dim=16):
     try:
         if path.lower().endswith(".pdf"):
             reader = PdfReader(path)
             total_len = 0
             for p in reader.pages[:min(len(reader.pages), 5)]:
                 t = p.extract_text() or ""
                 total_len += len(t)
             arr = np.asarray([total_len], dtype=np.float32)
             arr = arr / (arr + 1.0)
             return pad_out(arr, dim)
         return np.zeros(dim, dtype=np.float32)
     except Exception:
         return np.zeros(dim, dtype=np.float32)

