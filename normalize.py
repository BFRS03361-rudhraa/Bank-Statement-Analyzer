import os
import re
import pandas as pd
from collections import Counter
from difflib import get_close_matches

def detect_date_column(df):
    """Detect date column by looking for specific patterns like 'date', 'value date', 'txn date'."""
    # First, look for exact matches with common date column names
    date_patterns = [
        r'date', r'value\s*date', r'txn\s*date', r'transaction\s*date',
        r'value_date', r'txn_date', r'transaction_date'
    ]
    
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in date_patterns:
            if re.search(pattern, col_lower):
                try:
                    # Verify this column actually contains dates using DD/MM/YYYY format
                    sample = pd.to_datetime(df[col].dropna().astype(str), format='%d/%m/%Y', errors='coerce')
                    if sample.notna().sum() > 0:
                        return col
                except Exception:
                    continue
    
    # If no specific date column found, return None (no fallback)
    return None

def normalize_headers(df):
    """Normalize headers to standard format: Date, Credit/Debit, Description, Amount, Balance + additional columns."""
    
    # Define our standard column mappings
    standard_columns = {
        'Date': ['date', 'value date', 'txn date', 'transaction date', 'value_date', 'txn_date', 'transaction_date'],
        'Credit/Debit': ['cr/dr', 'cr dr', 'credit debit', 'type', 'transaction type', 'debit credit'],
        'Description': ['description', 'narration', 'particulars', 'details', 'transaction details'],
        'Amount': ['amount', 'transaction amount', 'value', 'transaction value', 'amount(inr)', 'transaction amount(inr)'],
        'Balance': ['balance', 'available balance', 'running balance', 'closing balance', 'available balance(inr)']
    }
    
    # Create mapping from original columns to standard columns
    column_mapping = {}
    used_standard_cols = set()
    
    # First pass: try to map each original column to a standard column
    for orig_col in df.columns:
        orig_col_lower = orig_col.lower().strip()
        mapped = False
        
        for std_col, patterns in standard_columns.items():
            if std_col not in used_standard_cols:  # Don't reuse standard columns
                for pattern in patterns:
                    if re.search(pattern, orig_col_lower):
                        column_mapping[orig_col] = std_col
                        used_standard_cols.add(std_col)
                        mapped = True
                        break
                if mapped:
                    break
        
        # If no standard mapping found, keep original column name
        if not mapped:
            column_mapping[orig_col] = orig_col
    
    # Create new dataframe with normalized headers
    new_df = df.copy()
    new_df = new_df.rename(columns=column_mapping)
    
    # Reorder columns: standard columns first, then additional columns
    final_columns = []
    
    # Add standard columns in order (if they exist)
    for std_col in ['Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']:
        if std_col in new_df.columns:
            final_columns.append(std_col)
    
    # Add any additional columns that aren't in our standard set
    for col in new_df.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    # Reorder the dataframe
    new_df = new_df[final_columns]
    
    # Normalize Credit/Debit values to DEBIT and CREDIT
    if 'Credit/Debit' in new_df.columns:
        new_df['Credit/Debit'] = new_df['Credit/Debit'].astype(str).str.upper()
        new_df['Credit/Debit'] = new_df['Credit/Debit'].replace({
            'DR': 'DEBIT',
            'CR': 'CREDIT',
            'D': 'DEBIT',
            'C': 'CREDIT'
        })
    
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
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        
        # Get first and last transaction dates (first and last rows)
        first_date = df[date_col].iloc[0]  # First row
        last_date = df[date_col].iloc[-1]  # Last row
        
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
