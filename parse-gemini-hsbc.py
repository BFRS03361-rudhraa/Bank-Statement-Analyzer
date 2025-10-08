import array
import os
import json
import re
import string
import time
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image,ImageEnhance, ImageFilter, ImageOps
import pandas as pd

# Load API key and configure model
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash', generation_config=genai.types.GenerationConfig(temperature=0))

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    print(f"[convert] Ensuring folder exists: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"[convert] Converting PDF to images @ {dpi} DPI: {pdf_path}")
    t0 = time.time()
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"[convert] Converted {len(images)} page(s) in {time.time() - t0:.2f}s")
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f'page_{i+1}.jpg')
        img.save(path, 'JPEG')
        image_paths.append(path)
    print(f"[convert] Saved {len(image_paths)} image(s) to {output_folder}")
    return image_paths

def preprocess_image_for_ocr(image_path):
    """Enhance contrast and clarity before passing to Gemini."""
    img = Image.open(image_path).convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def split_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))

    top_path = image_path.replace('.jpg', '_top.jpg')
    bottom_path = image_path.replace('.jpg', '_bottom.jpg')

    top_half.save(top_path, 'JPEG')
    bottom_half.save(bottom_path, 'JPEG')

    return top_path, bottom_path

def prompt_hsbc(header_instruction,hsbc_narrative_instruction) :
    
    prompt=f"""
        You are an expert bank statement extractor.

        Extract the data from this financial document image as a structured JSON object with two keys:
        - "metadata": should ONLY contain account/customer information like account number, name, address, statement period, etc., present before the transaction table. Metadata should NEVER contain transaction rows, table headers, or any financial data. If no key-value metadata is found, extract the entire metadata block as a single string field under "raw_metadata".
        - "transactions": a list of objects, each representing a transaction with exact column names and values as present in the document.
        - "Narrative" : a list of object , each having a column named Narrative and it's value as present in the document.

        Do NOT normalize, modify, or clean the data. Preserve every detail exactly as it appears with 100 percent accuracy. The values should be present in their exact columns with 100% accuracy , no values should be misplaced into other columns or adjacent empty columns so recheck every line properly .
        Do NOT invent any new keys or columns. If a column is missing a value, leave it as an empty string. Never shift values between columns.
        Only extract data that is directly visible. DO NOT MAKE UP ANY TRANSACTION THAT IS NOT VISIBLE.

        {header_instruction}
        
        {hsbc_narrative_instruction}

        Example output:
        {{
        "metadata": {{
            "Account Name": "...",
            "Account Number": "...",
            ...
        }},
        "transactions": [
            {{ "Txn Date": "...", "Debit": "...", "Credit": "...", ... }},
            ...
        ],
        "Narrative": [
            {{ "Narrative": "..." }}
        ]
        }}
    """
    return prompt


def ocr_with_gemini(image_path,canonical_header=None, hsbc_case=None):
    header_instruction = ""
    if canonical_header:
        header_instruction = f"""
        Use the following column headers as the canonical header for the transaction table:
        {json.dumps(canonical_header, ensure_ascii=False)}
        Ensure that every transaction entry uses these exact column names as keys.Output no other keys besides the canonical header, and do not invent any new fields."""
    
    hsbc_narrative_instruction=""
    if hsbc_case:
        hsbc_narrative_instruction = f""" 
            Seperately parse all the rows which has two columns, first one is fixed named Narrative and second one is a description.Treat the first one as the column name (Narrative) and second one as the value stored in it.
            """
    if hsbc_case:
        prompt = prompt_hsbc(header_instruction,hsbc_narrative_instruction)
    else :
        prompt = f"""
        You are an expert bank statement extractor.

        Extract the data from this financial document image as a structured JSON object with two keys:
        - "metadata": should ONLY contain account/customer information like account number, name, address, statement period, etc., present before the transaction table. Metadata should NEVER contain transaction rows, table headers, or any financial data. If no key-value metadata is found, extract the entire metadata block as a single string field under "raw_metadata".
        - "transactions": a list of objects, each representing a transaction with exact column names and values as present in the document.

        Do NOT normalize, modify, or clean the data. Preserve every detail exactly as it appears with 100 percent accuracy. The values should be present in their exact columns with 100% accuracy , no values should be misplaced into other columns or adjacent empty columns so recheck every line properly .
        Do NOT invent any new keys or columns. If a column is missing a value, leave it as an empty string. Never shift values between columns.
        Only extract data that is directly visible. DO NOT MAKE UP ANY TRANSACTION THAT IS NOT VISIBLE.

        {header_instruction}  
        Example output:
        {{
        "metadata": {{
            "Account Name": "...",
            "Account Number": "...",
            ...
        }},
        "transactions": [
            {{ "Txn Date": "...", "Debit": "...", "Credit": "...", ... }},
            ...
        ]
        }}
        """
    print(f"[ocr] Preprocessing image for OCR: {image_path}")
    image = preprocess_image_for_ocr(image_path)
    print("[ocr] Calling Gemini model…")
    t0 = time.time()
    response = model.generate_content([prompt, image])
    dt = time.time() - t0
    text = response.text or ""
    print(f"[ocr] Response received in {dt:.2f}s, length={len(text)}")
    return text


def ocr_with_gemini_with_fallback(image_path, canonical_header=None, hsbc=None):
    text = ocr_with_gemini(image_path, canonical_header=canonical_header,hsbc_case=hsbc)
    parsed = parse_json_from_llm(text,hsbc=hsbc)

    print("parsed", parsed)

    # Fallback if no data extracted
    if not parsed.get('transactions') and not parsed.get('metadata'):
        print("[ocr fallback] Initial OCR failed, splitting image and retrying…")
        top_img, bottom_img = split_image(image_path)

        top_text = ocr_with_gemini(top_img, canonical_header=canonical_header, hsbc_case=hsbc)
        bottom_text = ocr_with_gemini(bottom_img, canonical_header=canonical_header, hsbc_case=hsbc)
        print("top", top_text)
        print("bottom", bottom_text)

        top_parsed = parse_json_from_llm(top_text, hsbc=hsbc) or {"metadata": {}, "transactions": []}
        bottom_parsed = parse_json_from_llm(bottom_text, hsbc=hsbc) or {"metadata": {}, "transactions": []}
        

        print("sgr", top_parsed, bottom_parsed)

        combined_metadata = {
            **top_parsed.get('metadata', {}),
            **bottom_parsed.get('metadata', {})
        }

        combined_transactions = top_parsed.get('transactions', []) + bottom_parsed.get('transactions', [])
        combined_narrative = []

        if hsbc: combined_narrative =top_parsed.get('Narrative',[])+ bottom_parsed.get('Narrative',[])
        # Clean up split images
        os.remove(top_img)
        os.remove(bottom_img)

        return {"metadata": combined_metadata, "transactions": combined_transactions,"Narrative":combined_narrative}

    return parsed



def parse_json_from_llm(text: str, hsbc=None):
    """
    Parse the first valid JSON object from the LLM response text.
    Returns a dict with 'metadata' and 'transactions' keys.
    """
    if not text or not text.strip():
        if(hsbc):
            return {"metadata": {}, "transactions": [], "Narrative":[]}
        else:return {"metadata": {}, "transactions": []}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if(hsbc):
                return {
                    "metadata": obj.get("metadata", {}),
                    "transactions": obj.get("transactions", []),
                    "Narrative": obj.get("Narrative",[])
                }
            else :
                return {
                    "metadata": obj.get("metadata", {}),
                    "transactions": obj.get("transactions", []),
                }
    except Exception:
        pass

    # Fallback: try extracting the largest JSON block
    block = _extract_largest_json_object(text,hsbc=hsbc)
    if block:
        # try:
        #     if isinstance(block, dict):
        #         return {
        #             "metadata": obj.get("metadata", {}),
        #             "transactions": obj.get("transactions", [])
        #         }
        # except Exception:
        #     pass
        return block

    if hsbc: return {"metadata": {}, "transactions": [],"Narrative":[]}
    return {"metadata": {}, "transactions": []}

def _extract_largest_json_object(text: str,hsbc):
    """Return the first JSON object that contains 'metadata' or 'transactions'."""
    if not text:
        return None
    matches = re.findall(r"\{[\s\S]*\}", text)
    largest_obj = None

    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                # Keep the largest JSON block (most keys/length)
                if not largest_obj or len(json.dumps(obj)) > len(json.dumps(largest_obj)):
                    largest_obj = obj
        except Exception:
            continue

    if largest_obj:
        if(hsbc):
                return {
                    "metadata": obj.get("metadata", {}),
                    "transactions": obj.get("transactions", []),
                    "Narrative": obj.get("Narrative",[])
                }
        else :
                return {
                    "metadata": obj.get("metadata", {}),
                    "transactions": obj.get("transactions", []),
                }
    if hsbc: return {"metadata": {}, "transactions": [],"Narrative":[]}
    return {"metadata": {}, "transactions": []}

def parse_json_loose(text: str):
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    block = _extract_largest_json_object(text)
    if block:
        try:
            return json.loads(block)
        except Exception:
            return None
    return None

def _extract_largest_json_block(text: str):
    if not text:
        return None
    # Strip common markdown fences
    # Still rely on greedy JSON block extraction
    candidates = []
    for m in re.finditer(r"\{[\s\S]*?\}", text):
        candidates.append(m.group(0))
    for m in re.finditer(r"\[[\s\S]*?\]", text):
        candidates.append(m.group(0))
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]

def parse_json_loose_structured(text: str):
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    block = _extract_largest_json_block(text)
    if block:
        try:
            return json.loads(block)
        except Exception:
            return None
    return None


def structure_metadata_with_llm(raw_metadata_block: dict) -> dict:
    prompt = f"""
    You are an expert at recognizing structured metadata from financial documents.

    Given the following metadata block extracted from a document (which may contain random key-value pairs or a raw text block), extract the following important fields exactly as they appear:
    - account_name
    - account_number
    - ifsc_code
    - micr_code
    - statement_period

    If a field is not present, leave it empty. Retain any existing key-value pairs that are clearly identified.

    Metadata block:
    {json.dumps(raw_metadata_block, ensure_ascii=False)}

    Output structured JSON with at least the important fields.
    Just output the json , no explanation or extra text around the json
    """
    response = model.generate_content([prompt])
    structured_json = parse_json_loose_structured(response.text)
    return structured_json or {}

def normalize_transactions(transactions: list) -> list:
    """
    Normalize a list of transaction dicts:
    - Remove newlines in keys and values.
    - Replace multiple spaces/newlines in values with a single space.
    """
    normalized = []
    for txn in transactions:
        new_txn = {}
        for key, value in txn.items():
            # Clean keys: remove \n and strip
            clean_key = key.replace("\n", " ").strip()
            # Clean string values: remove \n and extra spaces
            if isinstance(value, str):
                clean_value = ' '.join(value.split())
            else:
                clean_value = value
            new_txn[clean_key] = clean_value
        normalized.append(new_txn)
    return normalized


def apply_canonical_header(transactions, canonical_header):
    """
    Rebuild each transaction dict using the canonical header.

    - transactions: list of dicts from LLM (keys may be wrong)
    - canonical_header: list of column names from first page header

    The function will use the values in the order of appearance,
    and assign them correctly to the canonical header columns.
    """
    normalized = []
    for txn in transactions:
        # Extract values from txn dict in order of appearance
        row_values = list(txn.values())
        new_txn = {}
        for idx, key in enumerate(canonical_header):
            if idx < len(row_values):
                new_txn[key] = row_values[idx] if row_values[idx] is not None else ""
            else:
                new_txn[key] = ""  # Fill empty if fewer values
        normalized.append(new_txn)
    print(normalized)
    return normalized


def process_pdf(pdf_path, output_dir,hsbc=None, pages=0):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_dir = os.path.join(output_dir, f"{base_name}_images")
    print(f"[start] Processing file: {pdf_path}")
    image_paths = convert_pdf_to_images(pdf_path, image_dir)
    if pages > 0:
        print(f"[start] Limiting to first {pages} page(s)")
        image_paths = image_paths[:pages]

    os.makedirs(output_dir, exist_ok=True)

    aggregated_transactions = []
    aggregated_narrative=[]
    raw_metadata_blocks = []
    canonical_header = None  #

    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i+1}/{len(image_paths)}...")
        if i == 0:
            page_obj = ocr_with_gemini_with_fallback(image_path, hsbc=hsbc)
        else:
            page_obj = ocr_with_gemini_with_fallback(image_path, canonical_header=canonical_header, hsbc=hsbc)

        raw_json_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.json")
        # with open(raw_json_path, 'w', encoding='utf-8') as f:
        #     f.write(gemini_json)
        print(f"Saved raw JSON for page {i+1}: {raw_json_path}")

        # page_obj = parse_json_from_llm(gemini_json)
        if page_obj is None:
            print(f"Warning: Failed to parse JSON on page {i+1}. Skipping.")
            continue

        if i == 0:
            # Extract canonical header from first page
            if page_obj.get('transactions'):
                canonical_header = list(page_obj['transactions'][0].keys())
            print(f"Canonical header set: {canonical_header}")
            
        if canonical_header and page_obj.get('transactions'):
            page_obj['transactions'] = apply_canonical_header(page_obj['transactions'], canonical_header)

            # Validation step
            for txn in page_obj['transactions']:
                extra_keys = set(txn.keys()) - set(canonical_header)
                missing_keys = set(canonical_header) - set(txn.keys())
                if extra_keys or missing_keys:
                    print(f"[WARNING] Transaction keys mismatch on page {i+1}:")
                    print(f"  Extra keys: {extra_keys}")
                    print(f"  Missing keys: {missing_keys}")

        if i < 2:
            raw_metadata_blocks.append(page_obj.get('metadata', {}))

        aggregated_transactions.extend(page_obj.get('transactions', []))
        if (hsbc) : aggregated_narrative.extend(page_obj.get('Narrative',[]))
    # print(f"[page {i+1}] Added {len(txns)} transactions. Total so far={len(aggregated_transactions)}")

    # Structure metadata from aggregated metadata blocks
    print(f"[metadata] Structuring metadata from {len(raw_metadata_blocks)} block(s)…")
    structured_metadata = structure_metadata_with_llm({"metadata_blocks": raw_metadata_blocks})

    # Save final JSON files
    aggregated_json_path = os.path.join(output_dir, f"{base_name}_aggregated.json")
    with open(aggregated_json_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_transactions, f, ensure_ascii=False, indent=2)
    print(f"[save] Aggregated transactions JSON: {aggregated_json_path} (rows={len(aggregated_transactions)})")

    structured_metadata_path = os.path.join(output_dir, f"{base_name}_structured_metadata.json")
    with open(structured_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(structured_metadata, f, ensure_ascii=False, indent=2)
    print(f"[save] Structured metadata JSON: {structured_metadata_path} (keys={len(structured_metadata)})")

    aggregated_transactions = normalize_transactions(aggregated_transactions)

    # Append Narrative column for HSBC right after normalization and before DataFrame conversion
    if hsbc:
        narrative_values = []
        for item in aggregated_narrative:
            if isinstance(item, dict):
                val = item.get("Narrative", "")
                narrative_values.append(' '.join(val.split()) if isinstance(val, str) else val)
            else:
                narrative_values.append(str(item) if item is not None else "")
        for idx, txn in enumerate(aggregated_transactions):
            txn["Narrative"] = narrative_values[idx] if idx < len(narrative_values) else ""
    # Save to Excel
    metadata_df = pd.DataFrame(list(structured_metadata.items()), columns=['Key', 'Value'])
    transactions_df = pd.DataFrame(aggregated_transactions)

    excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        transactions_df.to_excel(writer, sheet_name='Transactions', index=False)

    print(f"[save] Excel file: {excel_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='OCR Financial Document Page-by-Page into JSON and Excel')
    parser.add_argument('input', help='Path to PDF file or folder')
    parser.add_argument('--out', default='./output', help='Output directory')
    parser.add_argument('--hsbc' ,default=None, help='Processing for hsbc statement' )
    parser.add_argument('--pages', type=int, default=0, help='Max pages to process (0 = all)')
    
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(args.input, file)
                process_pdf(pdf_path, args.out,args.hsbc, args.pages)
    else:
        process_pdf(args.input, args.out, args.hsbc ,args.pages)

if __name__ == '__main__':
    main()