
import os
import re
import pymupdf

from config import IMAGE_DIR


def process_pdf(pdf_path):
    pages = []

    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)

    document = pymupdf.open(pdf_path)

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            # Extract text
            page_text = page.get_text("text").strip()

            # Render page as image
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False
            )

            image_filename = f"{safe_filename}_page_{page_number}.jpg"
            image_path = os.path.join(IMAGE_DIR, image_filename)

            pixmap.save(image_path)

            pages.append({
                "text": page_text,
                "page": page_number,
                "image_path": image_path
            })

    finally:
        document.close()

    return pages
