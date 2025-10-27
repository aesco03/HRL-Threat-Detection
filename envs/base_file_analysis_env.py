 import os
 import random
 import csv
 import numpy as np
 from gymnasium import Env, spaces
 from tools.extract_file_features import extract_file_features
 from analysis_modules.module_registry import run as run_module

 A_METADATA = 0
 A_PDF = 1
 A_TEXT = 2
 A_IMAGE_STEGO = 3
 A_CLASS0 = 4
 A_CLASS1 = 5

class FileAnalysisEnv(Env):
    def __init__(self, env_config=None):
        cfg = env_config or {}
        self.dataset_path = cfg.get("dataset_path", "./data/files/")
        self.labels_csv = cfg.get("labels_csv", None)
        self.feature_dim = int(cfg.get("feature_dim", 64))
        self.max_steps = int(cfg.get("max_steps", 20))
        self.step_cost = float(cfg.get("step_cost", 0.01))
        self.pos_fraction = cfg.get("pos_fraction", None)
        self.metadata_dim = int(cfg.get("metadata_dim", 16))
        self.seg_sizes = {"meta": self.metadata_dim, "pdf": 16, "text": 16, "image": 16}
        self.seg_offsets = {"meta": 0, "pdf": 16, "text": 32, "image": 48}
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.feature_dim,), dtype=np.float32)
        self.file_paths, self.labels = self._load_dataset()
        self.pos_indices = [i for i, y in enumerate(self.labels) if int(y) == 1]
        self.neg_indices = [i for i, y in enumerate(self.labels) if int(y) == 0]
        self._idx = 0
        self._label = 0
        self._steps = 0
        self._obs = np.zeros(self.feature_dim, dtype=np.float32)

     def _load_dataset(self):
         paths, labels = [], []
         for root, _, files in os.walk(self.dataset_path):
             for f in files:
                 fl = f.lower()
                 if fl.endswith((".pdf", ".png", ".jpg", ".jpeg")):
                     paths.append(os.path.join(root, f))
         if self.labels_csv:
             lm = {}
             with open(self.labels_csv, "r") as fh:
                 for name, y in csv.reader(fh):
                     lm[name] = int(y)
             labels = [lm.get(os.path.basename(p), 0) for p in paths]
         else:
             labels = [0] * len(paths)
         if not paths:
             raise ValueError(f"No files found in {self.dataset_path}")
         return paths, labels

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        if self.pos_fraction is None or not self.pos_indices or not self.neg_indices:
            self._idx = random.randrange(len(self.file_paths))
        else:
            choose_pos = random.random() < float(self.pos_fraction)
            pool = self.pos_indices if choose_pos else self.neg_indices
            self._idx = random.choice(pool)
        self._label = self.labels[self._idx]
        self._obs = np.zeros(self.feature_dim, dtype=np.float32)
        meta = extract_file_features(self.file_paths[self._idx], self.seg_sizes["meta"])
        off = self.seg_offsets["meta"]
        size = self.seg_sizes["meta"]
        self._obs[off:off+size] = meta
        return self._obs, {}

     def step(self, action):
         self._steps += 1
         reward = -self.step_cost
         terminated = False
         truncated = self._steps >= self.max_steps
         fp = self.file_paths[self._idx]
         if action in (A_CLASS0, A_CLASS1):
             y = 0 if action == A_CLASS0 else 1
             reward = 1.0 if y == int(self._label) else -1.0
             terminated = True
         elif action == A_METADATA:
             vec, seg = run_module("metadata_scan", fp, self.seg_sizes["meta"])
             off = self.seg_offsets["meta"]
             size = self.seg_sizes["meta"]
             self._obs[off:off+size] = vec
         elif action == A_PDF:
             vec, seg = run_module("pdf_structure", fp, self.seg_sizes["pdf"])
             off = self.seg_offsets["pdf"]
             size = self.seg_sizes["pdf"]
             self._obs[off:off+size] = vec
         elif action == A_TEXT:
             vec, seg = run_module("text_extract", fp, self.seg_sizes["text"])
             off = self.seg_offsets["text"]
             size = self.seg_sizes["text"]
             self._obs[off:off+size] = vec
         elif action == A_IMAGE_STEGO:
             vec, seg = run_module("image_stego", fp, self.seg_sizes["image"])
             off = self.seg_offsets["image"]
             size = self.seg_sizes["image"]
             self._obs[off:off+size] = vec
         info = {"true_label": int(self._label), "file_path": fp}
         return self._obs, reward, terminated, truncated, info
