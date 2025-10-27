 from .pdf_structure_analyzer import analyze as pdf_analyze
 from .text_bytecode_analyzer import analyze as text_analyze
 from .image_stego_analyzer import analyze as image_analyze

 ACTIONS = {
     "metadata_scan": ("meta", None),
     "pdf_structure": ("pdf", pdf_analyze),
     "text_extract": ("text", text_analyze),
     "image_stego": ("image", image_analyze),
 }

 def run(action_name, path, dim):
     seg, fn = ACTIONS.get(action_name, (None, None))
     if fn is None:
         import numpy as np
         return np.zeros(dim, dtype=np.float32), seg
     return fn(path, dim), seg

