import PyPDF2
import os

pdf_files = [
    "chain Quality Problem-Gold.pdf",
    "Chain Quality Problem-Silver.pdf"
]

output_file = "pdf_summary.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            out.write(f"\n--- FILE NOT FOUND: {pdf_path} ---\n")
            continue
            
        out.write(f"\n--- START OF {pdf_path} ---\n")
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    out.write(f"\n[Page {page_num+1}]\n")
                    out.write(text)
        except Exception as e:
            out.write(f"\nError extracting {pdf_path}: {e}\n")
            
        out.write(f"\n--- END OF {pdf_path} ---\n")

print(f"Extracted content to {output_file}")
