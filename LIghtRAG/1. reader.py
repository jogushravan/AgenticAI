# reader.py
import fitz  # PyMuPDF
import io
from PIL import Image

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    extracted = []

    for page in doc:
        text = page.get_text("text")
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            extracted.append({"text": text, "image": image})
    return extracted
