import random
import numpy as np
import pandas as pd
from gymnasium import Env, spaces

class PrecomputedFeaturesEnv(Env):
    def __init__(self, env_config=None):
        cfg = env_config or {}
        self.parquet_path = cfg["parquet_path"]
        self.label_col = cfg.get("label_col", "label")
        self.label_positive_values = set(cfg.get("label_positive_values", [1, "1", True, "Malicious", "malicious"]))
        self.label_negative_values = set(cfg.get("label_negative_values", [0, "0", False, "Benign", "benign"]))
        self.meta_cols = cfg.get("meta_cols", [])
        self.pdf_cols = cfg.get("pdf_cols", [])
        self.text_cols = cfg.get("text_cols", [])
        self.image_cols = cfg.get("image_cols", [])
        self.group_dim = int(cfg.get("group_dim", 16))
        self.max_steps = int(cfg.get("max_steps", 20))
        self.step_cost = float(cfg.get("step_cost", 0.01))
        self.pos_fraction = cfg.get("pos_fraction", None)
        self.seg_sizes = {"meta": self.group_dim, "pdf": self.group_dim, "text": self.group_dim, "image": self.group_dim}
        self.seg_offsets = {"meta": 0, "pdf": self.group_dim, "text": 2*self.group_dim, "image": 3*self.group_dim}
        self.feature_dim = 4 * self.group_dim
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.feature_dim,), dtype=np.float32)
        self.df = pd.read_parquet(self.parquet_path)
        raw_labels = self.df[self.label_col].tolist()
        self.labels = [1 if v in self.label_positive_values else 0 if v in self.label_negative_values else int(v) for v in raw_labels]
        self.indices = list(range(len(self.df)))
        self.pos_indices = [i for i, y in enumerate(self.labels) if int(y) == 1]
        self.neg_indices = [i for i, y in enumerate(self.labels) if int(y) == 0]
        self._idx = 0
        self._label = 0
        self._steps = 0
        self._obs = np.zeros(self.feature_dim, dtype=np.float32)

    def _vec(self, cols):
        if not cols:
            return np.zeros(self.group_dim, dtype=np.float32)
        x = self.df.iloc[self._idx][cols].to_numpy(dtype=np.float32)
        x = x / (x + 1.0)
        out = np.zeros(self.group_dim, dtype=np.float32)
        n = min(self.group_dim, x.shape[0])
        out[:n] = x[:n]
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        if self.pos_fraction is None or not self.pos_indices or not self.neg_indices:
            self._idx = random.randrange(len(self.indices))
        else:
            choose_pos = random.random() < float(self.pos_fraction)
            pool = self.pos_indices if choose_pos else self.neg_indices
            self._idx = random.choice(pool)
        self._label = int(self.labels[self._idx])
        self._obs = np.zeros(self.feature_dim, dtype=np.float32)
        vec = self._vec(self.meta_cols)
        off = self.seg_offsets["meta"]
        size = self.seg_sizes["meta"]
        self._obs[off:off+size] = vec
        return self._obs, {}

    def step(self, action):
        self._steps += 1
        reward = -self.step_cost
        terminated = False
        truncated = self._steps >= self.max_steps
        if action == 4:
            y = 0
            reward = 1.0 if y == self._label else -1.0
            terminated = True
        elif action == 5:
            y = 1
            reward = 1.0 if y == self._label else -1.0
            terminated = True
        elif action == 0:
            vec = self._vec(self.meta_cols)
            off = self.seg_offsets["meta"]
            size = self.seg_sizes["meta"]
            self._obs[off:off+size] = vec
        elif action == 1:
            vec = self._vec(self.pdf_cols)
            off = self.seg_offsets["pdf"]
            size = self.seg_sizes["pdf"]
            self._obs[off:off+size] = vec
        elif action == 2:
            vec = self._vec(self.text_cols)
            off = self.seg_offsets["text"]
            size = self.seg_sizes["text"]
            self._obs[off:off+size] = vec
        elif action == 3:
            vec = self._vec(self.image_cols)
            off = self.seg_offsets["image"]
            size = self.seg_sizes["image"]
            self._obs[off:off+size] = vec
        info = {"true_label": self._label, "row_index": self._idx}
        return self._obs, reward, terminated, truncated, info
