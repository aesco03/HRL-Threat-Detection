import os
import numpy as np
import pandas as pd
from imageextract import extract_all_features
# Existing function: used when images are grouped in a folder
def aggregate_features_for_pdf_folder(image_folder, pdf_name="unknown.pdf"):
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    return aggregate_features_from_image_list(image_paths, pdf_name)
def aggregate_features_from_image_list(image_paths, pdf_name="unknown.pdf"):
    features_list = []
    for img_path in image_paths:
        try:
            features = extract_all_features(img_path)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    num_images = len(features_list)

    def agg(key, func, default=np.nan):
        values = [f.get(key) for f in features_list if f.get(key) is not None]
        return func(values) if values else default

    def flag_any(key):
        return any(f.get(key) for f in features_list)

    def frac_above_zero(key):
        values = [f.get(key, 0) for f in features_list]
        return sum(1 for v in values if v > 0) / len(values)

 return {
        "filename": pdf_name,
        "num_images": num_images,
        "pdf_img_mean_size_bytes": agg("File_Size_Bytes", np.mean),
        "pdf_img_max_size_bytes": agg("File_Size_Bytes", np.max),
        "pdf_img_min_size_bytes": agg("File_Size_Bytes", np.min),
        "pdf_img_std_size_bytes": agg("File_Size_Bytes", np.std),
        "pdf_img_alpha_rate": agg("Alpha_Channel", np.mean, 0),
        "pdf_img_alpha": flag_any("Alpha_Channel"),
        "pdf_img_min_size_to_entropy": agg("Size_to_Entropy_Ratio", np.min),
        "pdf_img_mean_size_to_entropy": agg("Size_to_Entropy_Ratio", np.mean),
        "pdf_img_std_size_to_entropy": agg("Size_to_Entropy_Ratio", np.std),
        "pdf_img_mean_global_entropy": agg("Global_Entropy", np.mean),
        "pdf_img_max_global_entropy": agg("Global_Entropy", np.max),
        "pdf_img_min_global_entropy": agg("Global_Entropy", np.min),
        "pdf_img_std_global_entropy": agg("Global_Entropy", np.std),
        "pdf_img_mean_noise_std": agg("Noise_Std", np.mean),
        "pdf_img_max_noise_std": agg("Noise_Std", np.max),
        "pdf_img_min_noise_std": agg("Noise_Std", np.min),
        "pdf_img_std_noise_std": agg("Noise_Std", np.std),
        "pdf_img_max_LSB_entropy": agg("LSB_Entropy", np.max),
        "pdf_img_min_LSB_entropy": agg("LSB_Entropy", np.min),
        "pdf_img_mean_LSB_entropy": agg("LSB_Entropy", np.mean),
        "pdf_img_min_LSB_uniformity": agg("LSB_Uniformity_Score", np.min),
        "pdf_img_mean_LSB_uniformity": agg("LSB_Uniformity_Score", np.mean),
        "pdf_img_std_LSB_uniformity": agg("LSB_Uniformity_Score", np.std),
        "pdf_img_mean_edge_lsb": agg("Edge_LSB_Mean", np.mean),
        "pdf_img_std_edge_lsb": agg("Edge_LSB_Mean", np.std),
        "pdf_img_mean_nonedge_lsb": agg("NonEdge_LSB_Mean", np.mean),
        "pdf_img_std_nonedge_lsb": agg("NonEdge_LSB_Mean", np.std),
        "pdf_img_mean_dct": agg("DCT_Mean", np.mean),
        "pdf_img_std_dct": agg("DCT_Std", np.std),
        "pdf_img_max_color_skewness": agg("Color_Skewness_Score", np.max),
        "pdf_img_mean_color_skewness": agg("Color_Skewness_Score", np.mean),
        "pdf_img_mean_corr_BG": agg("Corr_BG", np.mean),
        "pdf_img_mean_corr_BR": agg("Corr_BR", np.mean),
        "pdf_img_mean_corr_GR": agg("Corr_GR", np.mean),
        "pdf_img_max_appended_payload_score": agg("Appended_Payload_Score", np.max),
        "pdf_img_payload_flag": flag_any("Appended_Payload"),
        "pdf_img_max_OCR_suspicion_score": agg("OCR_Suspicion_Score", np.max),
        "pdf_img_frac_ocr_suspicious": frac_above_zero("OCR_Suspicion_Score"),
        "pdf_img_max_hex_hits": agg("Hex_Pattern_Hits", np.max),
        "pdf_img_hex_hits_flag": flag_any("Hex_Pattern_Hits"),
        "pdf_img_max_keyword_score": agg("Keyword_Score", np.max),
    }
