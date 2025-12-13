import os
import magic
import pandas as pd
import imageextract as image_extractor
import pdfextract as pdf_extractor
import pdfimageextract as pdf_image_extractor

# All possible features
ALL_FEATURE_KEYS = [
    "filename", "file_type",
    # Image features
    "LSB_Mean_Deviation", "LSB_Uniformity_Score", "OCR_Suspicion_Score", "Appended_Payload_Score",
    "Color_Skewness_Score", "DCT_Mean", "DCT_Std", "Global_Entropy", "Alpha_Channel",
    "Image_Format", "Image_Size", "Image_Mode", "LSB_Entropy", "Edge_LSB_Mean", "NonEdge_LSB_Mean",
    "Hex_Pattern_Hits", "Keyword_Score", "File_Size_Bytes", "Size_to_Entropy_Ratio", "Noise_Std",
    "Corr_BG", "Corr_BR", "Corr_GR",
    # PDF features
    "PdfSize", "MetadataSize", "Pages", "XrefLength", "TitleCharacters", "isEncrypted",
    "EmbeddedFiles", "Images", "Text", "Obj", "EndObj", "Stream", "Endstream",
    "Xref", "Trailer", "StartXref", "PageNo", "Encrypt", "ObjStm", "JS", "Javascript",
    "AA", "OpenAction", "Acroform", "JBIG2Decode", "RichMedia", "Launch", "EmbeddedFile", "XFA", "Colors",
    # PDF-image features
    "num_images", "pdf_img_mean_size_bytes", "pdf_img_max_size_bytes", "pdf_img_min_size_bytes", "pdf_img_std_size_bytes",
    "pdf_img_alpha_rate", "pdf_img_alpha", "pdf_img_min_size_to_entropy", "pdf_img_mean_size_to_entropy", "pdf_img_std_size_to_entropy",
    "pdf_img_mean_global_entropy", "pdf_img_max_global_entropy", "pdf_img_min_global_entropy", "pdf_img_std_global_entropy",
    "pdf_img_mean_noise_std", "pdf_img_max_noise_std", "pdf_img_min_noise_std", "pdf_img_std_noise_std",
    "pdf_img_max_LSB_entropy", "pdf_img_min_LSB_entropy", "pdf_img_mean_LSB_entropy",
    "pdf_img_min_LSB_uniformity", "pdf_img_mean_LSB_uniformity", "pdf_img_std_LSB_uniformity",
    "pdf_img_mean_edge_lsb", "pdf_img_std_edge_lsb", "pdf_img_mean_nonedge_lsb", "pdf_img_std_nonedge_lsb",
    "pdf_img_mean_dct", "pdf_img_std_dct", "pdf_img_max_color_skewness", "pdf_img_mean_color_skewness",
    "pdf_img_mean_corr_BG", "pdf_img_mean_corr_BR", "pdf_img_mean_corr_GR",
    "pdf_img_max_appended_payload_score", "pdf_img_payload_flag",
    "pdf_img_max_OCR_suspicion_score", "pdf_img_frac_ocr_suspicious",
    "pdf_img_max_hex_hits", "pdf_img_hex_hits_flag", "pdf_img_max_keyword_score",
    #Added for one_hot encoding
    "file_type_pdf_image"
]

def detect_file_type(file_path):
    mime = magic.Magic(mime=True)
    try:
        return mime.from_file(file_path)
    except Exception as e:
        print(f"Error detecting MIME type: {e}")
        return "unknown"

def get_label_from_path(file_path, pos_labels, neg_labels):
    parts = [p.lower() for p in file_path.split(os.sep)]
    if any(p in pos_labels for p in parts):
        return 1
    if any(p in neg_labels for p in parts):
        return 0
    print(f"WARNING: Defaulting to benign for {file_path}")
    return 0

def normalize_features(raw_features, file_type, filename, label):
    normalized = {key: None for key in ALL_FEATURE_KEYS}
    normalized.update(raw_features)
    normalized["file_type"] = file_type
    normalized["filename"] = filename
    normalized["label"] = label
    return normalized

def process_file(file_path, pos_labels, neg_labels):
    file_type = detect_file_type(file_path)
    filename = os.path.basename(file_path)
    label = get_label_from_path(file_path, pos_labels, neg_labels)

    combined_features = {}

    if file_type.startswith("image") or file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        raw_features = image_extractor.extract_all_features(file_path)
        combined_features.update(raw_features)
        return [normalize_features(combined_features, "image", filename, label)]

    elif file_type == "application/pdf":
        raw_features = pdf_extractor.extract_info(file_path)
        if raw_features:
            combined_features.update(raw_features)
            
            combined_features["file_type_pdf_image"] = 1 if raw_features.get("Images", 0) > 0 else 0

            extracted_images = raw_features.get("ExtractedImageFiles", [])
            image_folder = "/home/azureuser/malware_data/features/pdf_images_features"
            image_paths = [
                os.path.join(image_folder, img_path)
                for img_path in extracted_images
                if os.path.isfile(os.path.join(image_folder, img_path))
            ]
            image_paths = list(set(image_paths))

            if image_paths:
                pdf_image_features = pdf_image_extractor.aggregate_features_from_image_list(image_paths, filename)
                combined_features.update(pdf_image_features)

            return [normalize_features(combined_features, "pdf", filename, label)]
        else:
            print(f"Skipped unreadable PDF: {file_path}")
            return []

    return []

def batch_process(input_path, output_csv, pos_labels, neg_labels):
    rows = []
    if os.path.isfile(input_path):
        rows.extend(process_file(input_path, pos_labels, neg_labels))
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                full_path = os.path.join(root, file)
                rows.extend(process_file(full_path, pos_labels, neg_labels))

    df = pd.DataFrame(rows, columns=ALL_FEATURE_KEYS + ["label"])

    # One-hot encoding for file types (only image and pdf now)
    df["file_type_image"] = (df["file_type"] == "image").astype(int)
    df["file_type_pdf"] = (df["file_type"] == "pdf").astype(int)
    # Ensure file_type_pdf_image exists (fill missing with 0)
    if "file_type_pdf_image" not in df.columns:
        df["file_type_pdf_image"] = 0


    # Drop original file_type column
    df.drop(columns=["file_type"], inplace=True)

    # Replace NaNs with 0
    df.fillna(0, inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows to {output_csv}")
    print(f"Columns in CSV: {list(df.columns)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified feature extractor for images, PDFs, and PDF-embedded images.")
    parser.add_argument("input", help="Path to a file or folder")
    parser.add_argument("--csv", help="Output CSV filename", default="unified_file_features.csv")
    parser.add_argument("--pos", nargs="*", default=["malicious_pdf", "stego_images"], help="Positive label folder names")
    parser.add_argument("--neg", nargs="*", default=["benign_pdf", "clean_images"], help="Negative label folder names")
    args = parser.parse_args()
    pos_labels = set([s.lower() for s in args.pos])
    neg_labels = set([s.lower() for s in args.neg])
    batch_process(args.input, args.csv, pos_labels, neg_labels)
