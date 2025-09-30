from math import nan
import os
import re
import pandas as pd
from collections import Counter
from difflib import get_close_matches


def clean_amount(value):
    """
    Clean financial amount strings:
    - Keep only numeric value (with sign if present).
    - Remove currency symbols, CR/DR suffixes, text, commas.
    """
    if pd.isna(value):
        return pd.NA

    val = str(value).strip().upper()

    # Handle explicit CR/DR suffix
    sign = 1
    if val.endswith("CR"):
        sign = 1
        val = val[:-2]
    elif val.endswith("DR"):
        sign = -1
        val = val[:-2]

    # Remove currency words/symbols
    val = re.sub(r"[^\d\.\-]", "", val)  # keep only digits, dot, minus

    if val == "" or val == "." or val == "-":
        return pd.NA

    try:
        return sign * float(val)
    except ValueError:
        return pd.NA


def parse_transaction_dates(series):
    """
    Parse a series of transaction date strings using multiple expected formats.
    Returns a datetime series.
    """
    s = series.astype(str).str.strip().str.replace("\u00A0", " ", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    formats = [
        "%d/%m/%Y %I:%M:%S %p",  # 16/07/2024 10:38:12 AM
        "%d/%m/%Y",              # 16/07/2024
        "%Y-%m-%d",              # 2024-09-02
        "%d-%m-%Y",              # 02-09-2024
        "%d/%m/%y %H:%M:%S",     # 02/09/24 13:01:43
        "%d-%b-%Y",              # 16-Nov-2024
        "%d-%B-%Y",              # 16-November-2024
    ]

    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")

    # final fallback: infer with day-first for leftovers
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)

    return parsed

def detect_date_column(df):
    """Detect date column by looking for specific patterns like 'date', 'value date', 'txn date'."""
    # First, look for exact matches with common date column names
    date_patterns = [
       r'txn\s*date', r'transaction\s*date',r'tran\s*date' ,r'txn\s*posted\s*date',r'post\s*date' ,r'date',  # highest priority
          # generic date last
    ]
    for pattern in date_patterns:
        for col in df.columns:
            col_lower = col.lower().strip()
        
            if re.search(pattern, col_lower):
                try:
                    # Verify this column actually contains dates using DD/MM/YYYY format
                    sample = parse_transaction_dates(df[col].dropna().astype(str))
                    if sample.notna().sum() > 0:
                        return col
                except Exception:
                    continue
    
    # If no specific date column found, return None (no fallback)
    return None

def normalize_headers(df):
    """
    Normalize headers to standard format: Date, Credit/Debit, Description, Amount, Balance
    + additional columns. Detects the actual date column dynamically.
    """
    # Standard column mappings
    standard_columns = {
        'Credit/Debit': ['cr/dr','dr/cr', 'cr dr', 'credit debit', 'type', 'transaction type', 'debit credit'],
        'Description': ['description', 'narration', 'particulars', 'details', 'transaction details'],
        'Credit':['credit', 'credit amount', 'desposit', 'cr'],
        'Debit':['debit', 'debit amount','withdrawal', 'dr'],
        'Amount': ['amount', 'transaction amount', 'transaction value', 'amount(inr)', 'transaction amount(inr)'],
        'Balance': ['balance', 'available balance', 'running balance', 'closing balance', 'available balance(inr)'],
        
    }

    new_df = df.copy()

    # ---- Step 1: Detect the actual date column ----
    date_col = detect_date_column(new_df)
    if date_col:
        new_df = new_df.rename(columns={date_col: 'Date'})
    else:
        print("Warning: No date column detected")

    # ---- Step 2: Map other standard columns ----
    column_mapping = {}
    used_standard_cols = set(['Date'] if 'Date' in new_df.columns else [])
    for orig_col in new_df.columns:
        if orig_col in used_standard_cols:
            continue
        orig_col_lower = orig_col.lower().strip()
        mapped = False
        for std_col, patterns in standard_columns.items():
            if std_col not in used_standard_cols:
                for pattern in patterns:
                    if re.search(pattern, orig_col_lower):
                        column_mapping[orig_col] = std_col
                        used_standard_cols.add(std_col)
                        mapped = True
                        break
                if mapped:
                    break
        if not mapped:
            column_mapping[orig_col] = orig_col

    new_df = new_df.rename(columns=column_mapping)
    
    # Step 2b: Handle separate Debit/Credit columns ONLY if both exist
    debit_col_candidates = [c for c in new_df.columns if 'debit' in c.lower() and c != 'Credit/Debit']
    credit_col_candidates = [c for c in new_df.columns if 'credit' in c.lower() and c != 'Credit/Debit']

    if debit_col_candidates and credit_col_candidates:
        debit_col = debit_col_candidates[0]
        credit_col = credit_col_candidates[0]

        # Only apply if BOTH columns exist
        # Safely create Credit/Debit column
        def get_cd(row):
            if pd.notna(row[credit_col]) and row[credit_col] != 0:
                return 'CREDIT'
            elif pd.notna(row[debit_col]) and row[debit_col] != 0:
                return 'DEBIT'
            else:
                return pd.NA

        new_df['Credit/Debit'] = new_df.apply(get_cd, axis=1)

        # Fill Amount column safely
        new_df['Amount'] = new_df.apply(
            lambda row: row[credit_col] 
                        if pd.notna(row['Credit/Debit']) and row['Credit/Debit'] == 'CREDIT'
                        else (row[debit_col] 
                            if pd.notna(row['Credit/Debit']) and row['Credit/Debit'] == 'DEBIT'
                            else pd.NA),
            axis=1
        )
    else:
        # Otherwise, keep the original Amount and Credit/Debit columns as they are
        if 'Amount' not in new_df.columns:
            # if missing, try to infer from existing columns (optional)
            pass


    # ---- Step 3: Reorder columns ----
    final_columns = []
    for std_col in ['Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']:
        if std_col in new_df.columns:
            final_columns.append(std_col)
    for col in new_df.columns:
        if col not in final_columns:
            final_columns.append(col)
    new_df = new_df[final_columns]

    # ---- Step 4: Parse and standardize Date ----
    if 'Date' in new_df.columns:
        new_df['Date'] = parse_transaction_dates(new_df['Date']).dt.strftime('%d/%m/%Y')

    # ---- Step 5: Normalize Credit/Debit values ----
    if 'Credit/Debit' in new_df.columns:
        new_df['Credit/Debit'] = new_df['Credit/Debit'].astype(str).str.upper()
        new_df['Credit/Debit'] = new_df['Credit/Debit'].replace({
            'DR': 'DEBIT',
            'CR': 'CREDIT',
            'D': 'DEBIT',
            'C': 'CREDIT'
        })

    for col in ['Credit', 'Debit', 'Amount', 'Balance']:
        if col in new_df.columns:
            new_df[col] = new_df[col].apply(clean_amount)

    return new_df


def get_file_date_range(file_path):
    """Extract first and last transaction dates from a file (first and last rows only)."""
    try:
        df = pd.read_excel(file_path, sheet_name="Transactions")
        date_col = detect_date_column(df)
        
        if date_col is None:
            print(f"Warning: No date column found in {file_path}")
            return None, None
        
        if df.empty:
            print(f"Warning: No transactions found in {file_path}")
            return None, None
        
        # Convert to datetime using DD/MM/YYYY format
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True )
        
        # Get first and last transaction dates (first and last rows)
        first_date = None
        for val in df[date_col]:
            if pd.notna(val):
                first_date = val
                break
        last_date = None
        for val in reversed(df[date_col]):
            if pd.notna(val):
                last_date = val
                break  # Last row
        
        # Check if dates are valid
        if pd.isna(first_date) or pd.isna(last_date):
            print(f"Warning: Invalid dates in first/last rows of {file_path}")
            return None, None
        
        return first_date, last_date
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def normalize_to_reference(excel_files):
    transactions_list = []
    metadata_list = []

    # Load reference header from the first file and normalize it
    ref_df = pd.read_excel(excel_files[0], sheet_name="Transactions")
    ref_df = normalize_headers(ref_df)
    reference_headers = list(ref_df.columns)

    for file in excel_files:
        print(f"Processing: {file}")

        # ---- Transactions ----
        try:
            df = pd.read_excel(file, sheet_name="Transactions")

            # Normalize headers to standard format
            df = normalize_headers(df)
            
            # Just detect date column, don't modify the data here
            date_col = detect_date_column(df)
            
            transactions_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read Transactions from {file}: {e}")

        # ---- Metadata ----
        try:
            meta_df = pd.read_excel(file, sheet_name="Metadata")
            metadata_dict = dict(zip(meta_df['Key'], meta_df['Value']))
            metadata_list.append(metadata_dict)
        except Exception as e:
            print(f"Warning: Could not read Metadata from {file}: {e}")

    # ---- Consolidated Transactions ----
    combined_transactions = (
        pd.concat(transactions_list, ignore_index=True)
        if transactions_list else pd.DataFrame()
    )

    # No need to sort here since files are already processed in correct order

    # ---- Consolidated Metadata ----
    all_keys = set(k for md in metadata_list for k in md.keys())
    aggregated_metadata = {}
    for key in all_keys:
        values = [md.get(key, '') for md in metadata_list if md.get(key)]
        most_common = Counter(values).most_common(1)
        aggregated_metadata[key] = most_common[0][0] if most_common else ''

    return combined_transactions, aggregated_metadata


def consolidate_excels(input_folder, output_file):
    excel_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.endswith('.xlsx') and not f.startswith('.')
    ]

    if not excel_files:
        print("No Excel files found in the folder.")
        return

    print(f"Found {len(excel_files)} Excel files to process.")

    # Sort files by their first transaction date
    print("Analyzing first transaction dates in files...")
    file_date_info = []
    
    for file_path in excel_files:
        first_date, last_date = get_file_date_range(file_path)
        if first_date is not None and last_date is not None:
            file_date_info.append((file_path, first_date, last_date))
            print(f"  {os.path.basename(file_path)}: First txn: {first_date.date()}, Last txn: {last_date.date()}")
        else:
            print(f"  {os.path.basename(file_path)}: No valid dates found - will be processed last")
            file_date_info.append((file_path, None, None))
    
    # Sort by first transaction date only (files with no date will be last)
    file_date_info.sort(key=lambda x: x[1] if x[1] is not None else pd.Timestamp.max)
    sorted_excel_files = [info[0] for info in file_date_info]
    
    print(f"Files will be processed in chronological order based on first transaction date.")
    
    transactions, metadata = normalize_to_reference(sorted_excel_files)

    # Save
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        if not transactions.empty:
            transactions.to_excel(writer, index=False, sheet_name="Consolidated Transactions")
        pd.DataFrame(list(metadata.items()), columns=["Key", "Value"]).to_excel(
            writer, index=False, sheet_name="Consolidated Metadata"
        )

    print(f"Saved consolidated file: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standardize headers and merge metadata across parsed Excel files")
    parser.add_argument("input_folder", help="Folder containing parsed Excel files")
    parser.add_argument("--output", default="./standardized_output.xlsx", help="Output Excel file path")
    args = parser.parse_args()

    consolidate_excels(args.input_folder, args.output)
