from math import nan
from operator import or_
import os
import re
import pandas as pd
import numpy as np
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
    if val.endswith("CR" or "CR"):
        sign = 1
        val = val[:-2]
    elif val.endswith("DR"):
        sign = 1
        val = val[:-2]
    elif val.endswith("DR."):
        sign = 1
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
        'Description': ['description', 'narration','narrative', 'particulars', 'details', 'transaction details','remarks','naration'],
        'Amount': ['amount', 'withdrawal (dr)/ deposit(cr)','transaction amount', 'transaction value', 'amount(inr)', 'transaction amount(inr)'],
        'Credit':['credit', 'credit amount', 'deposit', 'cr'],
        'Debit':['debit', 'debit amount','withdrawal', 'dr'],
        'Balance': ['balance', 'available balance', 'running balance', 'closing balance', 'available balance(inr)','total'],
        
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
                        # print(f"Mapped {orig_col} to {std_col}")
                        break
                if mapped:
                    break
        if not mapped:
            column_mapping[orig_col] = orig_col

    new_df = new_df.rename(columns=column_mapping)
    # print(f"New dataframe: {new_df}")
    
    # Step 2b: Handle separate Debit/Credit columns ONLY if both exist
    debit_col_candidates = [c for c in new_df.columns if 'debit' in c.lower() and c != 'Credit/Debit']
    credit_col_candidates = [c for c in new_df.columns if 'credit' in c.lower() and c != 'Credit/Debit']

    if debit_col_candidates and credit_col_candidates:
        debit_col = debit_col_candidates[0]
        credit_col = credit_col_candidates[0]
        # print(f"Debit column: {debit_col}, Credit column: {credit_col}")

        # Only apply if BOTH columns exist
        # Safely create Credit/Debit column
        for col in ['Credit', 'Debit']:
            if col in new_df.columns:
                new_df[col] = new_df[col].apply(clean_amount)

        def get_cd(row):
            # print(f"Row: {row}\n")
            if pd.notna(row[credit_col]) and row[credit_col] != 0 and row[credit_col] != '-':
                # print(f"Credit column: {credit_col}")
                return 'CREDIT'
            elif pd.notna(row[debit_col]) and row[debit_col] != 0 and row[debit_col] != '-':
                return 'DEBIT'
            else:
                return pd.NA

        new_df['Credit/Debit'] = new_df.apply(get_cd, axis=1)
        # print(f"New dataframe: {new_df['Credit/Debit']}")
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
    
    # print(f"New dataframe: {new_df}")

    # ---- Step 5: Normalize Credit/Debit values ----
    if 'Credit/Debit' in new_df.columns:
        new_df['Credit/Debit'] = new_df['Credit/Debit'].astype(str).str.upper()
        new_df['Credit/Debit'] = new_df['Credit/Debit'].replace({
            'DR': 'DEBIT',
            'CR': 'CREDIT',
            'DR.': 'DEBIT',
            'CR.': 'CREDIT',
            'D': 'DEBIT',
            'C': 'CREDIT',
            'Transfer Credit': 'CREDIT',
            'Transfer Debit': 'DEBIT',
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
            
            df = validate_credit_debit(df)

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

import pandas as pd

def validate_credit_debit(df):
    """
    Validates and corrects Credit/Debit values by comparing Amount with Balance change.
    - Preserves original transaction order (no sorting)
    - Infers missing Amount if possible
    
    """
    df = df.copy()
    if 'Balance' not in df.columns:
        return df

    # --- ADD THIS ---
    df['Balance'] = (
    df['Balance']
    .apply(clean_amount)
    .replace('', np.nan)
)
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')

# Drop rows where Balance is still NaN (optional)
    df = df[df['Balance'].notna()].reset_index(drop=True)
    if 'Amount' not in df.columns:
        print("Inferring Amount and Credit/Debit from balance difference...")
        df['Balance'] = df['Balance'].apply(clean_amount)
        df['Amount'] = df['Balance'].diff().abs().round(2)
        df['Credit/Debit'] = df['Balance'].diff().apply(lambda x: 'CREDIT' if x > 0 else ('DEBIT' if x < 0 else pd.NA))
        return df




    required_cols = {"Credit/Debit", "Amount", "Balance"}
    if not required_cols.issubset(df.columns):
        print("⚠️  Skipping validation: Missing one or more required columns.")
        return df

    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")

    inferred_types = [None]  # first txn has no previous reference
    inferred_amounts = [df.loc[0, "Amount"]]  # keep original first amount

    for i in range(1, len(df)):
        prev_bal = df.loc[i - 1, "Balance"]
        curr_bal = df.loc[i, "Balance"]
        amt = df.loc[i, "Amount"]

        if pd.isna(prev_bal) or pd.isna(curr_bal):
            inferred_types.append(None)
            inferred_amounts.append(amt)
            continue

        balance_diff = curr_bal - prev_bal

        # If amount missing, infer from balance difference
        if pd.isna(amt):
            amt_inferred = abs(balance_diff)
        else:
            amt_inferred = abs(amt)

        # Determine credit/debit type
        if abs(balance_diff - amt_inferred) < 1e-3:
            inferred_type = "CREDIT"
        elif abs(balance_diff + amt_inferred) < 1e-3:
            inferred_type = "DEBIT"
        else:
            inferred_type = None  # ambiguous

        inferred_types.append(inferred_type)
        inferred_amounts.append(amt if not pd.isna(amt) else amt_inferred)

    df["Inferred_Credit/Debit"] = inferred_types
    df["Inferred_Amount"] = inferred_amounts

    # Identify mismatches
    mismatches = df[
        (df["Inferred_Credit/Debit"].notna()) &
        (df["Credit/Debit"].notna()) &
        (df["Inferred_Credit/Debit"] != df["Credit/Debit"])
    ]

    missing_amounts_fixed = df["Amount"].isna().sum() - df["Inferred_Amount"].isna().sum()
    mismatch_count = len(mismatches)
    total_txns = len(df)

    print("\n===== CREDIT/DEBIT VALIDATION REPORT =====")
    print(f"Total transactions: {total_txns}")
    print(f"Mismatched transaction types: {mismatch_count}")
    print(f"Missing amounts inferred: {missing_amounts_fixed}")

    if mismatch_count > 0:
        print("\n⚠️  Mismatch Details:")
        print(
            mismatches[["Date", "Description", "Credit/Debit", "Inferred_Credit/Debit", "Amount", "Balance"]]
            .to_string(index=False)
        )
    else:
        print("✅ No mismatched credit/debit labels found.")

    # Apply inferred corrections
    df.loc[df["Inferred_Credit/Debit"].notna(), "Credit/Debit"] = df["Inferred_Credit/Debit"]
    df.loc[df["Inferred_Amount"].notna(), "Amount"] = df["Inferred_Amount"]

    # Drop helper columns
    df = df.drop(columns=["Inferred_Credit/Debit", "Inferred_Amount"])

    # Summary
    final_credit_count = (df["Credit/Debit"] == "CREDIT").sum()
    final_debit_count = (df["Credit/Debit"] == "DEBIT").sum()

    print("\n===== FINAL TRANSACTION COUNTS =====")
    print(f"CREDIT: {final_credit_count}")
    print(f"DEBIT:  {final_debit_count}")
    print("====================================\n")

    return df



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
