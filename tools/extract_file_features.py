 import os
 import numpy as np
 from PyPDF2 import PdfReader
 from PIL import Image

 def extract_file_features(path, feature_dim=16):
     ext = os.path.splitext(path.lower())[1]
     is_pdf = 1 if ext == ".pdf" else 0
     is_img = 1 if ext in [".png", ".jpg", ".jpeg"] else 0
     feats = []
     return out

