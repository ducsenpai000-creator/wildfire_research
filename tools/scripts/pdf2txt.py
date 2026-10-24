"""Extract text from a PDF into a .txt file next to it (page markers included)."""
import sys
from pathlib import Path

from pypdf import PdfReader


def main(pdf_path: str) -> None:
    pdf = Path(pdf_path)
    reader = PdfReader(str(pdf))
    out = pdf.with_suffix(".txt")
    with out.open("w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages, 1):
            f.write(f"\n===== PAGE {i} =====\n")
            f.write(page.extract_text() or "")
    print(f"{out} pages={len(reader.pages)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
