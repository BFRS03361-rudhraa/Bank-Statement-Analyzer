# parser_main.py
import argparse
from parse_gemini_hsbc import process_pdf
def run_pipeline(input_path, output_dir="./out1t"):
    """
    Your main PDF parsing logic here.
    input_path: single PDF or folder
    output_dir: output location
    """
    # existing logic goes here
    print(f"Parsing {input_path} into {output_dir}")
    # call your OCR + LLM pipeline
    # return whatever results
    result = process_pdf(input_path, output_dir)
    print(f"Completed parsing {input_path}")

    return {"file": input_path, "status": "done"}

def parse_pdf(pdf_path, output_dir="./output", max_pages=0):
    """
    Wrapper used by Redis worker jobs. Matches signature expected by enqueue.py
    """
    return process_pdf(pdf_path, output_dir, max_pages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Financial Document Page-by-Page into JSON and Excel')
    parser.add_argument('input', help='Path to PDF file or folder')
    parser.add_argument('--out', default='./out1', help='Output directory')

    args = parser.parse_args()
    run_pipeline(args.input, args.out)
