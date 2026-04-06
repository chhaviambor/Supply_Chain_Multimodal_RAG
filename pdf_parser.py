import fitz  # PyMuPDF
import pdfplumber
import os
import json
import re
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text from PDF page by page with error handling."""
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        return []
        
    pages_text = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                pages_text.append({
                    "page": page_num + 1,
                    "content": text
                })
        doc.close()
    except Exception as e:
        print(f"Error opening/parsing text from {pdf_path}: {e}")
    return pages_text

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts tables from PDF safely."""
    tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table:
                            # Filter out empty rows/cols
                            cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                            if any(any(row) for row in cleaned_table): # Ensure table is not empty
                                tables_data.append({
                                    "page": i + 1,
                                    "table": cleaned_table
                                })
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
    return tables_data

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """Extracts images safely."""
    images_info = []
    try:
        doc = fitz.open(pdf_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                images_info.append({
                    "page": page_num + 1,
                    "path": image_path,
                    "filename": image_filename
                })
        doc.close()
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")
    return images_info

def chunk_text(pages_text: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunks text into smaller sections with overlap."""
    chunks = []
    for page in pages_text:
        text = page["content"]
        page_num = page["page"]
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "page": page_num,
                "content": chunk,
                "type": "text"
            })
            start += (chunk_size - overlap)
    return chunks

if __name__ == "__main__":
    # Test extraction
    PDF_PATH = "data/supplychain.pdf"
    if os.path.exists(PDF_PATH):
        print("Extracting text...")
        text = extract_text_from_pdf(PDF_PATH)
        print(f"Extracted {len(text)} pages of text.")
        
        print("Extracting tables...")
        tables = extract_tables_from_pdf(PDF_PATH)
        print(f"Extracted {len(tables)} tables.")
        
        print("Extracting images...")
        images = extract_images_from_pdf(PDF_PATH, "temp_images")
        print(f"Extracted {len(images)} images.")
    else:
        print(f"Error: {PDF_PATH} not found.")
