"""Extract text from reference documents for reading."""
import sys
from pathlib import Path

def extract_docx(path):
    from docx import Document
    doc = Document(path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def extract_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)

ref_dir = Path("docs/reference_materials")
out_dir = Path("docs/reference_materials/extracted")
out_dir.mkdir(exist_ok=True)

for f in sorted(ref_dir.iterdir()):
    if f.is_dir():
        continue
    print(f"Processing: {f.name}")
    try:
        if f.suffix == ".docx":
            text = extract_docx(f)
        elif f.suffix == ".pdf":
            text = extract_pdf(f)
        else:
            continue
        out_path = out_dir / (f.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(text)
        print(f"  -> {out_path} ({len(text)} chars)")
    except Exception as e:
        print(f"  ERROR: {e}")
