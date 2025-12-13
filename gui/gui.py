
from tkinterdnd2 import TkinterDnD, DND_FILES
import tkinter as tk
from tkinter import messagebox, scrolledtext
import shutil
import os
import json
import pandas as pd
import numpy as np
import joblib
from scripts.unified_extractor import process_file, ALL_FEATURE_KEYS
import torch
import torch.nn as nn
import torch.nn.functional as F

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORACLE_PTH = os.path.join(BASE_DIR, "oracle_p02.pth")
SCALER_PATH =os.path.join(BASE_DIR, "scaler.joblib")
CFG_PATH = os.path.join(BASE_DIR, "feature_cfg.json")

# Compatibility placeholders (not used for prediction)
POS_LABELS = {"malicious_pdf", "stego_images"}
NEG_LABELS = {"benign_pdf", "clean_images"}

# Load artifacts
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not (os.path.exists(ORACLE_PTH) and os.path.exists(SCALER_PATH) and os.path.exists(CFG_PATH)):
    raise FileNotFoundError(
        "Missing one or more required files: oracle_p02.pth, scaler.joblib, feature_cfg.json"
    )

with open(CFG_PATH, "r") as f:
    cfg = json.load(f)

scaler = joblib.load(SCALER_PATH)

# Must exist in cfg for correct scaling
continuous_cols_to_scale = cfg.get("continuous_cols_to_scale")
if continuous_cols_to_scale is None:
    raise ValueError("feature_cfg.json missing 'continuous_cols_to_scale'.")

# Use feature_cols if present, else build from groups
feature_cols = cfg.get("feature_cols")
if feature_cols is None:
    feature_cols = cfg["common_cols"] + cfg["pdf_cols"] + cfg["image_cols"] + cfg["pdf_img_agg_cols"]

THRESH = float(cfg.get("thresh", 0.5))
MC_T = int(cfg.get("mc_T", 20))

# ----------------------------
# Model defs (must match training)
# ----------------------------
class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)

class MultiHeadClassifier(nn.Module):
    def __init__(self, dims, p_drop=0.2):
        super().__init__()
        self.common_tower = nn.Sequential(nn.Linear(dims['common'], 64), nn.ReLU(), MCDropout(p=p_drop))
        self.pdf_tower    = nn.Sequential(nn.Linear(dims['pdf'], 64), nn.ReLU(), MCDropout(p=p_drop))
        self.image_tower  = nn.Sequential(nn.Linear(dims['image'], 64), nn.ReLU(), MCDropout(p=p_drop))
        self.agg_tower    = nn.Sequential(nn.Linear(dims['agg'], 32), nn.ReLU(), MCDropout(p=p_drop))
        self.fusion       = nn.Sequential(nn.Linear(224, 128), nn.ReLU(), MCDropout(p=p_drop), nn.Linear(128, 1))

    def forward(self, c, p, i, a):
        concat = torch.cat([self.common_tower(c), self.pdf_tower(p), self.image_tower(i), self.agg_tower(a)], dim=1)
        return self.fusion(concat)

def mc_predict_proba(model, c, p, i, a, T=20):
    probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(c, p, i, a)       # (1,1)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
    return float(np.mean(probs)), float(np.std(probs))

oracle = MultiHeadClassifier(cfg["input_dims"], p_drop=cfg["p_drop"]).to(device)
oracle.load_state_dict(torch.load(ORACLE_PTH, map_location=device))
oracle.eval()
for param in oracle.parameters():
    param.requires_grad = False

print("Loaded oracle. Dropout p =", oracle.common_tower[2].p)


# Normalization helper
def normalize_features(features):
    return {k: (0 if v is None else v) for k, v in features.items()}
    
# Feature helpers
# ---------------------
def vectorize_and_scale(feature_dict: dict) -> pd.DataFrame:
    # 1-row dataframe with exact columns in correct order
    row = {col: 0.0 for col in feature_cols}
    for k, v in feature_dict.items():
        if k in row:
            row[k] = v

    df1 = pd.DataFrame([row], columns=feature_cols)
    df1 = df1.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # scale only continuous columns (must match training)
    df1.loc[:, continuous_cols_to_scale] = scaler.transform(df1[continuous_cols_to_scale])
    return df1

def split_to_towers(df1: pd.DataFrame):
    c = torch.tensor(df1[cfg["common_cols"]].values, dtype=torch.float32, device=device)
    p = torch.tensor(df1[cfg["pdf_cols"]].values, dtype=torch.float32, device=device)
    i = torch.tensor(df1[cfg["image_cols"]].values, dtype=torch.float32, device=device)
    a = torch.tensor(df1[cfg["pdf_img_agg_cols"]].values, dtype=torch.float32, device=device)
    return c, p, i, a

def predict_with_oracle(feature_dict: dict):
    df1 = vectorize_and_scale(feature_dict)
    c, p, i, a = split_to_towers(df1)
    mean_p, std_p = mc_predict_proba(oracle, c, p, i, a, T=MC_T)
    pred = 1 if mean_p > THRESH else 0
    return mean_p, std_p, pred



# GUI setup
root = TkinterDnD.Tk()
root.title("Malware Detection")

label = tk.Label(root, text="Drag and Drop PDF/Image Here:")
label.pack(pady=10)

file_list = tk.Listbox(root, width=50, height=5)
file_list.pack(pady=10)

feature_display = scrolledtext.ScrolledText(root, width=80, height=20)
feature_display.pack(pady=10)

extracted_data = []

def on_drop(event):
    paths = root.tk.splitlist(event.data)
    for p in paths:
        p = p.strip()
        if p:
            file_list.insert(tk.END, p)


root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

def analyze_files():
    if file_list.size() == 0:
        messagebox.showwarning("No File", "Please drop a file first.")
        return
    global extracted_data
    extracted_data = []


    feature_display.delete(1.0, tk.END)

    for i in range(file_list.size()):
        file_path = file_list.get(i)
        try:
            dest_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)

            # Extract features
            features_list = process_file(dest_path, POS_LABELS, NEG_LABELS)
            if not features_list:
                feature_display.insert(tk.END, f"Could not extract features from {file_path}\n")
                continue

            features = features_list[0] if isinstance(features_list, list) else features_list
            normalized_features = normalize_features(features)

            # Display normalized features
            feature_display.insert(tk.END, f"\n=== Features for {os.path.basename(file_path)} ===\n")
            for key, value in normalized_features.items():
                feature_display.insert(tk.END, f"{key}: {value}\n")

            # Add to export list
            row = {"filename": os.path.basename(file_path)}
            row.update(normalized_features)
            extracted_data.append(row)

            # Predict if model exists
            mean_p, std_p, pred = predict_with_oracle(normalized_features)
            result = "MALICIOUS" if pred == 1 else "SAFE"
            feature_display.insert(tk.END, f"\nOracle Prediction: {result}\n")
            feature_display.insert(tk.END, f"prob_mean={mean_p:.4f}  uncertainty={std_p:.4f}\n")


        except Exception as e:
            feature_display.insert(tk.END, f"Error processing {file_path}: {str(e)}\n")

def export_to_csv():
    if not extracted_data:
        messagebox.showwarning("No Data", "No features to export yet.")
        return
    df = pd.DataFrame(extracted_data)
    output_file = "features.csv"
    df.to_csv(output_file, index=False)
    messagebox.showinfo("Exported", f"Features saved to {output_file}")

btn_extract = tk.Button(root, text="Extract Features", command=analyze_files)
btn_extract.pack(pady=10)

btn_export = tk.Button(root, text="Export to CSV", command=export_to_csv)
btn_export.pack(pady=10)

root.mainloop()
