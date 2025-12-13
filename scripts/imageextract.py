import os
import cv2
import numpy as np
from PIL import Image
import piexif
import pytesseract
from scipy.stats import entropy
import pandas as pd
#Features being extracted:
# Features being extracted:
'''
EXIF Metadata: Extract Metadata from image Files
LSB Steganalysis: Analyze Least Significant Bit Patterns
Appended Payload Detection: Check for extra data appended to image files
OCR Text Extraction: Extract text using Optical Character Recognition
Patch-Based Feature Extraction: Analyze image patches for anomalies
Color-Based Features: Compute color histograms and moments
Feature-Based Anomaly Scoring: Generate anomaly scores based on extracted features

ADDED:
DCT Analysis: Detect frequency-domain anomalies using Discrete Cosine Transform
Global Entropy: Measure overall randomness in pixel distribution
Alpha Channel Detection: Check for presence of transparency layer
Image Format & Dimensions: Log image type, width, height, and channel count

Other features to add:
Adaptive Steganography Detection - Edge maps + LSB variation
Hexadecimal Pattern Analysis - Read as binary, look for hexa patterns
Author/Camera information Already kinda covered in EXIF
Date/time stamps Already kinda covered in EXIF extractible via piexif
Comments field
File format and compression
Entropy analysis, done a bit through patch entropy, not global
'''

# 1 EXIF Metadata
def extract_exif_metadata(image_path):
    try:
        img = Image.open(image_path)
        if "exif" not in img.info:
            return {"info": "No EXIF metadata found"}
        exif_data = piexif.load(img.info["exif"])
        metadata = {}
        for ifd in exif_data:
            for tag in exif_data[ifd]:
                metadata[str(tag)] = exif_data[ifd][tag]
        return metadata
    except Exception as e:
        return {"error": str(e)}

# 2. LSB Steganalysis
def lsb_analysis(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not readable"}
    lsb_stats = {}
    channels = ('B', 'G', 'R')
    for i, ch in enumerate(channels):
        channel = img[:, :, i]
        lsb = channel & 1
        lsb_stats[f"{ch}_LSB_mean"] = float(np.mean(lsb))
        lsb_stats[f"{ch}_LSB_std"] = float(np.std(lsb))
    return lsb_stats
def extract_lsb_entropy(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not readable"}
    lsb_bits = []
    for i in range(3):  # BGR channels
        channel = img[:, :, i]
        lsb = channel & 1
        lsb_bits.extend(lsb.flatten())
    hist, _ = np.histogram(lsb_bits, bins=2, range=(0, 2), density=True)
    return {"LSB_Entropy": float(entropy(hist + 1e-6))}
# 3. Appended Payload Detection
def check_appended_payload(image_path):
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            return not data.endswith(b'\xff\xd9')  # JPEG EOF marker
        elif image_path.lower().endswith('.png'):
            return not data.endswith(b'IEND\xaeB`\x82')  # PNG EOF chunk
        else:
            return False  # Unsupported format
    except Exception:
        return False


# 4. OCR Text Extraction
def extract_text(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"OCR error: {str(e)}"

# 5. Global Entropy
def extract_global_entropy(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Image not readable"}
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
    return {"Global_Entropy": float(entropy(hist + 1e-6))}

# 6. DCT Features
def extract_dct_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Image not readable"}
    img = cv2.resize(img, (128, 128))
    dct = cv2.dct(np.float32(img))
    return {
        "DCT_Mean": float(np.mean(dct)),
        "DCT_Std": float(np.std(dct))
    }

# 7. Alpha Channel Detection
def detect_alpha_channel(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"Alpha_Channel": None}
    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
    return {"Alpha_Channel": has_alpha}

# 8. Image Format Info
def extract_image_format_info(image_path):
    try:
        img = Image.open(image_path)
        return {
            "Image_Format": img.format,
            "Image_Size": img.size,
            "Image_Mode": img.mode
        }
    except Exception as e:
        return {"error": str(e)}

# 9. Color Histograms
def extract_color_histograms(image_path, bins=32):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not readable"}
    hist_features = {}
    channels = ('b', 'g', 'r')
    for i, col in enumerate(channels):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features[f"{col.upper()}_hist"] = hist.tolist()
    return hist_features
# 10. Color Moments
def extract_color_moments(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not readable"}
    moments = {}
    channels = ('b', 'g', 'r')
    for i, col in enumerate(channels):
        channel = img[:, :, i].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-6)
        moments[f"{col.upper()}_mean"] = float(mean)
        moments[f"{col.upper()}_std"] = float(std)
        moments[f"{col.upper()}_skewness"] = float(skewness)
    return moments
# Added Additional Features:
def edge_aware_lsb_variation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Image not readable"}
    edges = cv2.Canny(img, 100, 200)
    lsb = img & 1
    edge_lsb = lsb[edges > 0]
    non_edge_lsb = lsb[edges == 0]
    return {
        "Edge_LSB_Mean": float(np.mean(edge_lsb)) if edge_lsb.size else 0,
        "NonEdge_LSB_Mean": float(np.mean(non_edge_lsb)) if non_edge_lsb.size else 0
    }

def scan_hex_patterns(image_path):
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        hex_str = data.hex()
        suspicious_patterns = ['3c7363726970743e', '706f7765727368656c6c', '68747470', '657468657265756d']
        """
        added

    '3c7363726970743e',        # <script>
    '706f7765727368656c6c',    # powershell
    '68747470',                # http
    '657468657265756d',        # ethereum
    '6576616c',                # eval
    '626173653634',            # base64
    '636d642e657865',          # cmd.exe
    '2e657865',                # .exe
    '2f7368656c6c2f',          # /shell/

        """
        hits = sum(1 for pattern in suspicious_patterns if pattern in hex_str)
        return {"Hex_Pattern_Hits": hits}
    except Exception as e:
        return {"error": str(e)}

def compression_artifact_analysis(image_path):
    try:
        size_bytes = os.path.getsize(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Image not readable"}
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
        ent = entropy(hist + 1e-6)
        return {
            "File_Size_Bytes": size_bytes,
            "Size_to_Entropy_Ratio": size_bytes / (ent + 1e-6)
        }
    except Exception as e:
        return {"error": str(e)}

def detect_keywords(image_path):
    keywords = ["<script>", "powershell", "http", "ethereum", "function", "eval"]
    score = 0
    try:
        text = pytesseract.image_to_string(Image.open(image_path)).lower()
        score += sum(1 for kw in keywords if kw in text)
        with open(image_path, 'rb') as f:
            raw = f.read().decode(errors='ignore').lower()
            score += sum(1 for kw in keywords if kw in raw)
    except Exception:
        pass
    return {"Keyword_Score": score}

def estimate_image_noise(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Image not readable"}
    noise = img - cv2.GaussianBlur(img, (3, 3), 0)
    return {"Noise_Std": float(np.std(noise))}

def color_channel_correlation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not readable"}
    b, g, r = cv2.split(img)
    corr_bg = np.corrcoef(b.flatten(), g.flatten())[0, 1]
    corr_br = np.corrcoef(b.flatten(), r.flatten())[0, 1]
    corr_gr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
    return {
        "Corr_BG": float(corr_bg),
        "Corr_BR": float(corr_br),
        "Corr_GR": float(corr_gr)
    }


# 11. Anomaly Scoring
def compute_lsb_anomaly(lsb_stats):
    means = [lsb_stats.get(f"{ch}_LSB_mean", 0.5) for ch in ['B', 'G', 'R']]
    stds = [lsb_stats.get(f"{ch}_LSB_std", 0.25) for ch in ['B', 'G', 'R']]
    mean_deviation = np.mean([abs(m - 0.5) for m in means])
    std_uniformity = np.mean([1 - s for s in stds])
    return {
        "LSB_Mean_Deviation": mean_deviation,
        "LSB_Uniformity_Score": std_uniformity
    }

def compute_ocr_suspicion(text):
    suspicious_keywords = ["<script>", "powershell", "http", "ethereum", "function", "eval"]
    score = sum(1 for word in suspicious_keywords if word.lower() in text.lower())
    return {"OCR_Suspicion_Score": score}

def compute_payload_flag(flag):
    return {"Appended_Payload_Score": int(flag)}



def compute_color_anomaly(features):
    try:
        skewness_keys = [k for k in features if "skewness" in k]
        skewness_vals = [features[k] for k in skewness_keys if features[k] is not None]
        if not skewness_vals:
            return {"Color_Skewness_Score": None}
        skewness_score = np.mean([abs(s) for s in skewness_vals])
        return {"Color_Skewness_Score": skewness_score}
    except Exception:
        return {"Color_Skewness_Score": None}



# Main Feature Extraction


def extract_all_features(image_path):
    print(f"\n Processing: {image_path}")
    features = {}

    try:
        features.update(extract_exif_metadata(image_path))
        features.update(lsb_analysis(image_path))
        features.update(extract_lsb_entropy(image_path))
        features.update({"Appended_Payload": check_appended_payload(image_path)})
        features.update({"OCR_Text": extract_text(image_path)})
        features.update(extract_global_entropy(image_path))
        features.update(extract_dct_features(image_path))
        features.update(detect_alpha_channel(image_path))
        features.update(extract_image_format_info(image_path))
        features.update(extract_color_histograms(image_path))
        features.update(extract_color_moments(image_path))
        features.update(edge_aware_lsb_variation(image_path))
        features.update(scan_hex_patterns(image_path))
        features.update(detect_keywords(image_path))
        features.update(compression_artifact_analysis(image_path))
        features.update(estimate_image_noise(image_path))
        features.update(color_channel_correlation(image_path))

        # Anomaly scoring
        scores = {}
        scores.update(compute_lsb_anomaly(features))
        scores.update(compute_ocr_suspicion(features.get("OCR_Text", "")))
        scores.update(compute_payload_flag(features.get("Appended_Payload", False)))
        scores.update(compute_color_anomaly(features.get("Color_Moments", {})))
        scores.update(compute_color_anomaly(features))
        features.update(scores)

    except Exception as e:
        print(f" Error extracting features from {image_path}: {e}")

    return features


# Batch Processing
def batch_extract_to_csv(image_folder, output_csv="features.csv"):
    rows = [] 
    for dirpath, _, filenames in os.walk(image_folder):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dirpath, filename)
                features = extract_all_features(image_path)
            row = {
                "filename": filename,
                "file_type": "image",
                "LSB_Mean_Deviation": features["Anomaly_Scores"].get("LSB_Mean_Deviation"),
                "LSB_Uniformity_Score": features["Anomaly_Scores"].get("LSB_Uniformity_Score"),
                "OCR_Suspicion_Score": features["Anomaly_Scores"].get("OCR_Suspicion_Score"),
                "Appended_Payload_Score": features["Anomaly_Scores"].get("Appended_Payload_Score"),
                "Color_Skewness_Score": features["Anomaly_Scores"].get("Color_Skewness_Score"),
                "DCT_Mean": features.get("DCT_Features", {}).get("DCT_Mean"),
                "DCT_Std": features.get("DCT_Features", {}).get("DCT_Std"),
                "Global_Entropy": features.get("Global_Entropy", {}).get("Global_Entropy"),
                "Alpha_Channel": features.get("Alpha_Channel", {}).get("Alpha_Channel_Present"),
                "Image_Format": features.get("Image_Info", {}).get("Image_Format"),
                "Image_Size": features.get("Image_Info", {}).get("Image_Size"),
                "Image_Mode": features.get("Image_Info", {}).get("Image_Mode"),
                "LSB_Entropy": features.get("LSB_Entropy"),
                "Edge_LSB_Mean": features.get("Edge_LSB_Mean"),
                "NonEdge_LSB_Mean": features.get("NonEdge_LSB_Mean"),
                "Hex_Pattern_Hits": features.get("Hex_Pattern_Hits"),
                "Keyword_Score": features.get("Keyword_Score"),
                "File_Size_Bytes": features.get("File_Size_Bytes"),
                "Size_to_Entropy_Ratio": features.get("Size_to_Entropy_Ratio"),
                "Noise_Std": features.get("Noise_Std"),
                "Corr_BG": features.get("Corr_BG"),
                "Corr_BR": features.get("Corr_BR"),
                "Corr_GR": features.get("Corr_GR"),

            }
            rows.append(row)
    df = pd.DataFrame(rows)
    #df.to_csv(output_csv, index=False)
    #print(f"\n Saved {len(rows)} feature rows to {output_csv}")

# CLI Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract image features and save to CSV.")
    parser.add_argument("input", help="Path to image file or folder")
    parser.add_argument("--csv", help="Output CSV file (for folder mode)", default="features.csv")
    args = parser.parse_args()
    if os.path.isdir(args.input):
        batch_extract_to_csv(args.input, args.csv)
    elif os.path.isfile(args.input):
        all_features = extract_all_features(args.input)
        print("\n Feature Summary:")
        for key, value in all_features.items():
            print(f"\n {key}:")
            print(value if isinstance(value, str) else str(value)[:500] + "..." if isinstance(value, list) else value)
    else:
        print(" Invalid input path. Please provide a valid image file or folder.")
