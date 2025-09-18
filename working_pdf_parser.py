#!/usr/bin/env python3
"""
Working PDF parser that outputs:
- Unstructured text (per page)
- Tables (per page) as cell matrices

Outputs for each PDF:
- <stem>.json  -> {"pages":[{"page":1,"text":"...","tables":[[...]]}, ...]}
- <stem>.xlsx  -> sheet "Text" + sheets p{page}_t{idx}
"""

import os, sys, json, tempfile, glob, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import fitz
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf_page(page) -> str:
    """Extract text content from a single PDF page"""
    try:
        text = page.get_text()
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"Error extracting text from page: {e}")
        return ""


def _clean_table_rows(rows: List[List[Any]]) -> List[List[str]]:
    """Ensure each row is list[str] with equal length, None->''."""
    if not rows:
        return []
    width = max(len(r) for r in rows)
    out = []
    for r in rows:
        rr = list(r) + [""] * (width - len(r))
        out.append([("" if c is None else str(c)).strip() for c in rr])
    return out


def extract_tables_from_pdf_page(page) -> List[List[List[str]]]:
    """Extract tables from a single PDF page using multiple strategies for better results"""
    try:
        tables = []
        
        # Strategy 1: Try PyMuPDF's find_tables() first
        tab = page.find_tables()
        if tab.tables:
            logger.info(f"Found {len(tab.tables)} potential table(s) using find_tables()")
            
            # Group tables that might be fragments of the same table
            grouped_tables = _group_fragmented_tables(tab.tables)
            
            for table_group in grouped_tables:
                if table_group:
                    # Merge the grouped tables
                    merged_table = _merge_table_fragments(table_group)
                    if merged_table and len(merged_table) > 1:  # At least header + 1 data row
                        tables.append(merged_table)
                        logger.info(f"Added merged table with {len(merged_table)} rows")
        
        # Strategy 2: If no tables found, try text-based extraction
        if not tables:
            logger.info("No tables found with find_tables(), trying text-based extraction...")
            text_content = page.get_text()
            
            # Try to extract transaction table from text
            transaction_table = _extract_transaction_table_from_text(text_content)
            if transaction_table:
                tables.append(transaction_table)
                logger.info(f"Extracted transaction table with {len(transaction_table)} rows from text")
        
        # Strategy 3: Try to find table-like structures in text
        if not tables:
            logger.info("Trying to find table-like structures in text...")
            text_content = page.get_text()
            table_structures = _find_table_structures_in_text(text_content)
            if table_structures:
                tables.extend(table_structures)
                logger.info(f"Found {len(table_structures)} table-like structures")
        
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting tables from page: {e}")
        return []


def _group_fragmented_tables(tables) -> List[List]:
    """Group tables that might be fragments of the same table"""
    if len(tables) <= 1:
        return [tables]
    
    # Look for tables with similar structure that might be fragments
    groups = []
    processed = set()
    
    for i, table in enumerate(tables):
        if i in processed:
            continue
            
        current_group = [table]
        processed.add(i)
        
        # Look for other tables that might be continuations
        for j, other_table in enumerate(tables[i+1:], i+1):
            if j in processed:
                continue
                
            # Check if tables might be related
            if _tables_might_be_related(table, other_table):
                current_group.append(other_table)
                processed.add(j)
        
        groups.append(current_group)
    
    return groups


def _tables_might_be_related(table1, table2) -> bool:
    """Check if two tables might be fragments of the same table"""
    try:
        # Convert to pandas for easier comparison
        df1 = table1.to_pandas()
        df2 = table2.to_pandas()
        
        # Check if they have similar column structure
        if df1.shape[1] != df2.shape[1]:
            return False
        
        # Check if they contain similar data types (transaction data)
        # Look for common patterns in the first few rows
        sample1 = ' '.join(str(cell) for cell in df1.iloc[0] if pd.notna(cell)).lower()
        sample2 = ' '.join(str(cell) for cell in df2.iloc[0] if pd.notna(cell)).lower()
        
        # Check for transaction-related keywords
        transaction_keywords = ['transaction', 'date', 'amount', 'balance', 'description', 'cr', 'dr', 'no.']
        has_keywords1 = any(keyword in sample1 for keyword in transaction_keywords)
        has_keywords2 = any(keyword in sample2 for keyword in transaction_keywords)
        
        # If both have transaction keywords, they might be related
        if has_keywords1 and has_keywords2:
            return True
        
        # Check if they have similar numeric patterns (amounts, dates)
        numeric_patterns = [r'\d+', r'\d+/\d+/\d+', r'\d+\.\d+', r'[A-Z]{2}']
        import re
        
        for pattern in numeric_patterns:
            if re.search(pattern, sample1) and re.search(pattern, sample2):
                return True
        
        return False
        
    except Exception:
        return False


def _merge_table_fragments(table_group) -> List[List[str]]:
    """Merge a group of table fragments into one cohesive table"""
    if not table_group:
        return []
    
    try:
        # Convert all tables to pandas DataFrames
        dataframes = []
        for table in table_group:
            df = table.to_pandas()
            if not df.empty:
                dataframes.append(df)
        
        if not dataframes:
            return []
        
        # Concatenate all DataFrames
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicate rows based on first few columns
        if len(merged_df.columns) >= 3:
            merged_df = merged_df.drop_duplicates(subset=merged_df.columns[:3])
        
        # Convert back to list of lists
        result = []
        for _, row in merged_df.iterrows():
            row_data = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
            result.append(row_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error merging table fragments: {e}")
        return []


def _extract_transaction_table_from_text(text: str) -> List[List[str]]:
    """Extract transaction table from text using flexible pattern matching"""
    import re
    
    lines = text.split('\n')
    table_rows = []
    
    # Look for transaction patterns
    transaction_patterns = [
        # Pattern 1: Number + Transaction ID + Date + Description + CR/DR + Amount + Balance
        r'(\d+)\s+([A-Z]\d+)\s+(\d{2}/\d{2}/\d{4}).*?([A-Z]{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})',
        # Pattern 2: Number + Date + Description + CR/DR + Amount + Balance
        r'(\d+)\s+(\d{2}/\d{2}/\d{4}).*?([A-Z]{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})',
        # Pattern 3: Date + Description + CR/DR + Amount + Balance
        r'(\d{2}/\d{2}/\d{4}).*?([A-Z]{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})',
    ]
    
    # Find the best matching pattern
    best_matches = []
    best_pattern = None
    
    for pattern in transaction_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) > len(best_matches):
            best_matches = matches
            best_pattern = pattern
    
    if best_matches and len(best_matches) > 0:
        # Create header based on the pattern
        if best_pattern == transaction_patterns[0]:
            header = ['No.', 'Transaction ID', 'Date', 'Description', 'Cr/Dr', 'Amount', 'Balance']
        elif best_pattern == transaction_patterns[1]:
            header = ['No.', 'Date', 'Description', 'Cr/Dr', 'Amount', 'Balance']
        else:
            header = ['Date', 'Description', 'Cr/Dr', 'Amount', 'Balance']
        
        table_rows.append(header)
        
        # Add data rows
        for match in best_matches:
            if best_pattern == transaction_patterns[0]:
                row_num, trans_id, date, cr_dr, amount, balance = match
                # Extract description from the original text
                description = _extract_description_from_line(text, date, cr_dr)
                table_rows.append([row_num, trans_id, date, description, cr_dr, amount, balance])
            elif best_pattern == transaction_patterns[1]:
                row_num, date, cr_dr, amount, balance = match
                description = _extract_description_from_line(text, date, cr_dr)
                table_rows.append([row_num, date, description, cr_dr, amount, balance])
            else:
                date, cr_dr, amount, balance = match
                description = _extract_description_from_line(text, date, cr_dr)
                table_rows.append([date, description, cr_dr, amount, balance])
    
    return table_rows


def _extract_description_from_line(text: str, date: str, cr_dr: str) -> str:
    """Extract description from text between date and CR/DR"""
    try:
        # Find the line containing this date and CR/DR
        lines = text.split('\n')
        for line in lines:
            if date in line and cr_dr in line:
                # Extract text between date and CR/DR
                date_start = line.find(date) + len(date)
                cr_dr_start = line.find(cr_dr)
                if date_start < cr_dr_start:
                    description = line[date_start:cr_dr_start].strip()
                    # Clean up the description
                    description = re.sub(r'\s+', ' ', description).strip()
                    return description
        return ""
    except Exception:
        return ""


def _find_table_structures_in_text(text: str) -> List[List[List[str]]]:
    """Find table-like structures in text using heuristics"""
    tables = []
    lines = text.split('\n')
    
    # Look for lines that might be table headers
    header_indicators = ['no.', 'transaction', 'date', 'amount', 'balance', 'description', 'cr/dr', 'value date', 'cheque']
    
    current_table = []
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Check if this line looks like a header
        if any(indicator in line_lower for indicator in header_indicators):
            # Start a new table
            if current_table:
                tables.append(current_table)
            current_table = [line.split()]
            in_table = True
            continue
        
        # If we're in a table, check if this line looks like data
        if in_table and len(line) > 10:
            # Look for patterns that suggest this is table data
            # (numbers, dates, amounts, etc.)
            if re.search(r'\d+', line) or re.search(r'[A-Z]{2}', line):
                # Split the line into columns (split on multiple spaces)
                columns = re.split(r'\s{2,}', line)
                if len(columns) >= 3:  # At least 3 columns
                    current_table.append(columns)
                else:
                    # Single column, add as is
                    current_table.append([line])
            else:
                # End of table
                if current_table and len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
                in_table = False
    
    # Add the last table if it exists
    if current_table and len(current_table) > 1:
        tables.append(current_table)
    
    return tables


def parse_pdf_document(pdf_path: str) -> Dict[str, Any]:
    """
    Parse PDF document and extract text and tables per page.
    Output JSON format:
    {"pages":[{"page":1,"text":"...","tables":[[...]]}, ...]}
    """
    result_pages = []
    
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Processing PDF with {len(doc)} pages: {pdf_path}")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_idx = page_num + 1
            
            logger.info(f"Processing page {page_idx}")
            
            # Extract text
            text_content = extract_text_from_pdf_page(page)
            
            # Extract tables
            tables = extract_tables_from_pdf_page(page)
            
            # Clean tables
            cleaned_tables = [_clean_table_rows(t) for t in tables]
            
            result_pages.append({
                "page": page_idx,
                "text": text_content,
                "tables": cleaned_tables
            })
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return {"pages": [], "error": str(e)}
    
    return {"pages": result_pages}


def write_json_excel(doc: Dict[str, Any], out_json: str, out_xlsx: str):
    """Write output to both JSON and Excel formats"""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # JSON
    with open(out_json, "w", encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    # Excel: sheet "Text" and table sheets
    pages = doc.get("pages", [])
    rows_text = []
    tables_dict = {}  # sheet_name -> DataFrame
    
    for p in pages:
        rows_text.append({
            "page": p.get("page"), 
            "text": p.get("text", "")[:1000]  # Limit text length for Excel
        })
        
        for ti, tbl in enumerate(p.get("tables", []), start=1):
            if tbl:  # Only add non-empty tables
                df = pd.DataFrame(tbl)
                sheet = f"p{p.get('page')}_t{ti}"
                tables_dict[sheet] = df

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        # Text sheet
        pd.DataFrame(rows_text).to_excel(xw, sheet_name="Text", index=False)
        
        # Table sheets
        for sheet, df in tables_dict.items():
            # Excel sheet name <= 31 chars
            safe_sheet = (sheet[:31]) if len(sheet) > 31 else sheet
            df.to_excel(xw, sheet_name=safe_sheet, header=False, index=False)


def main_cli():
    import argparse, pathlib
    
    ap = argparse.ArgumentParser(description="PDF Parser - Extract text and tables from PDFs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Path to a single PDF")
    g.add_argument("--pdf-dir", help="Directory containing PDFs")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    files = []
    if args.pdf:
        files = [args.pdf]
    else:
        for ext in ("*.pdf","*.PDF"):
            files.extend(sorted(glob.glob(os.path.join(args.pdf_dir, ext))))

    os.makedirs(args.out, exist_ok=True)
    logger.info(f"Found {len(files)} PDF files to process")

    for pdf_path in files:
        logger.info(f"Processing: {pdf_path}")
        stem = pathlib.Path(pdf_path).stem
        out_json = os.path.join(args.out, stem + ".json")
        out_xlsx = os.path.join(args.out, stem + ".xlsx")
        
        doc = parse_pdf_document(pdf_path)
        write_json_excel(doc, out_json, out_xlsx)
        logger.info(f"Completed: {pdf_path} -> {out_json}, {out_xlsx}")


if __name__ == "__main__":
    main_cli()

