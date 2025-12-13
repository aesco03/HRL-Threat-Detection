from pathlib import Path
import fitz
import os
from PyPDF2 import PdfReader, generic
from PyPDF2.generic import DictionaryObject, IndirectObject
import csv
import re
import magic
from PIL import Image
import io

def count_title_characters(pdf_path):
    reader = PdfReader(pdf_path)
    metadata = reader.metadata
    if metadata and metadata.title:
        return len(metadata.title)
    return 0

def has_text(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text().strip()
        if text:
            return "Yes"
    return "No"

def get_metadata_size(pdf_path):
    with open(pdf_path, "rb") as f:
        content = f.read()
    trailer_match = re.search(rb'trailer\s*<<.*?/Info\s+(\d+)\s+0\s+R', content, re.DOTALL)
    if not trailer_match:
        return 0
    obj_num = trailer_match.group(1)
    info_pattern = rb'%s\s+0\s+obj(.*?)endobj' % obj_num
    info_match = re.search(info_pattern, content, re.DOTALL)
    if not info_match:
        return 0
    metadata_bytes = info_match.group(1)
    return len(metadata_bytes)

def detect_file_type_from_bytes(file_bytes):
    mime = magic.Magic(mime=True)
    return mime.from_buffer(file_bytes)

def extract_and_export_images(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    pdf_prefix = os.path.splitext(os.path.basename(pdf_path))[0]

    #  Create unique subfolder for this PDF
    pdf_output_dir = os.path.join(output_dir, pdf_prefix)
    os.makedirs(pdf_output_dir, exist_ok=True)

    image_refs = []
    # Inline images
    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"{pdf_prefix}_page{page_index+1}_img{img_index+1}.{ext}"
            filepath = os.path.join(pdf_output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            image_refs.append(os.path.join(pdf_prefix, filename))  #  relative path

    # Embedded files
    for fname in doc.embfile_names():
        file_bytes = doc.embfile_get(fname)
        mime_type = detect_file_type_from_bytes(file_bytes)
        if mime_type.startswith("image/"):
            ext = mime_type.split("/")[-1]
            filename = f"{pdf_prefix}_embedded_{fname or 'unnamed'}.{ext}"
            filepath = os.path.join(pdf_output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(file_bytes)
            image_refs.append(os.path.join(pdf_prefix, filename))

    return image_refs

def count_embedded_files(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        return doc.embfile_count()
    except Exception:
        return 0

def scan_for_keys(obj, target_keys, visited=None):
    if visited is None:
        visited = set()
    key_counts = {k: 0 for k in target_keys}
    if isinstance(obj, IndirectObject):
        obj_id = (obj.pdf, obj.idnum, obj.generation)
        if obj_id in visited:
            return key_counts
        visited.add(obj_id)
        try:
            obj = obj.get_object()
        except Exception:
            return key_counts
    if isinstance(obj, DictionaryObject):
        for key, value in obj.items():
            if key in target_keys:
                key_counts[key] += 1
            nested_counts = scan_for_keys(value, target_keys, visited)
            for k in key_counts:
                key_counts[k] += nested_counts[k]
    elif isinstance(obj, list):
        for item in obj:
            nested_counts = scan_for_keys(item, target_keys, visited)
            for k in key_counts:
                key_counts[k] += nested_counts[k]
    return key_counts

def scan_pdf_for_keys(pdf_path, target_keys):
    total_counts = {k: 0 for k in target_keys}
    try:
        reader = PdfReader(pdf_path)
        root = reader.trailer.get('/Root')
        if root:
            counts = scan_for_keys(root, target_keys)
            for k in total_counts:
                total_counts[k] += counts[k]
    except Exception:
        pass
    return total_counts

def size_pdf(pdf_path):
    return os.path.getsize(pdf_path) // 1024

def is_encrypted(pdf_path):
    doc = fitz.open(pdf_path)
    return int(doc.is_encrypted)

def extract_info(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        target_keys = [
            '/Xref', '/Trailer', '/StartXref', '/PageNo', '/Encrypt', '/ObjStm', '/JS', '/Javascript', '/AA',
            '/OpenAction', '/Acroform', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile', '/XFA', '/Colors'
        ]
        key_counts = scan_pdf_for_keys(pdf_path, target_keys)
        info = {
            "filename": os.path.basename(pdf_path),
            "PdfSize": size_pdf(pdf_path),
            "MetadataSize": get_metadata_size(pdf_path),
            "Pages": page_count,
            "XrefLength": doc.xref_length(),
            "TitleCharacters": count_title_characters(pdf_path),
            "isEncrypted": is_encrypted(pdf_path),
            "EmbeddedFiles": count_embedded_files(pdf_path),
            "Text": has_text(pdf_path),
            "Obj": len(re.findall(rb"\bobj\b", open(pdf_path, "rb").read())),
            "EndObj": len(re.findall(rb"\bendobj\b", open(pdf_path, "rb").read())),
            "Stream": len(re.findall(rb"\bstream\b", open(pdf_path, "rb").read())),
            "Endstream": len(re.findall(rb"\bendstream\b", open(pdf_path, "rb").read())),
        }
        #  Extract images into unique subfolder
        image_refs = extract_and_export_images(pdf_path, "/home/azureuser/malware_data/features/pdf_images_features")
        info["Images"] = len(image_refs)
        info["ExtractedImageFiles"] = image_refs
        for k, v in key_counts.items():
            info[k.strip('/')] = v
        return info
    except Exception as e:
        print(f"Skipping unreadable file: {pdf_path} — {e}")
        return None
