
import pypdf
import os

pdf_path = "Teknologi Sains Data - Program Studi - Fakultas Teknologi Maju dan Multidisiplin _ Universitas Airlangga.pdf"
output_path = "pdf_content.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            f.write(f"--- Page {page_num + 1} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"Successfully extracted text to {output_path}")
except Exception as e:
    print(f"Error extracting text: {e}")

