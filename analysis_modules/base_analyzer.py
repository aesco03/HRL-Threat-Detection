 import numpy as np

 def pad_out(vec, dim):
     out = np.zeros(dim, dtype=np.float32)
     n = min(dim, len(vec))
     out[:n] = np.asarray(vec, dtype=np.float32)[:n]
     return out

