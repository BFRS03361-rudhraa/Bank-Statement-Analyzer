import os
import pandas as pd
from collections import Counter

def get_min_date_from_file(file):
    """Extract the minimum valid date from any 'date' column in the Transactions sheet."""
    try:
        df = pd.read_excel(file, sheet_name="Transactions")
        date_cols = [c for c in df.columns if "date" in c.lower()]
        all_dates = []
        for col in date_cols:
            dates = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            all_dates.extend(dates.dropna())
        if all_dates:
            return min(all_dates)
    except Exception as e:
        print(f"Warning: Could not extract dates from {file}: {e}")
    return pd.NaT

def normalize_to_reference(excel_files):
    transactions_list = []
    metadata_list = []

    # Sort files by earliest date in each file
    file_dates = [(f, get_min_date_from_file(f)) for f in excel_files]
    sorted_files = sorted(file_dates, key=lambda x: (pd.isna(x[1]), x[1]))
    sorted_files = [f for f, _ in sorted_files]

    # Use the first file as reference for headers
    ref_df = pd.read_excel(sorted_files[0], sheet_name="Transactions")
    reference_headers = list(ref_df.columns)

    for file in sorted_files:
        print(f"Processing: {file}")

        # ---- Transactions ----
        try:
            df = pd.read_excel(file, sheet_name="Transactions")
            df.columns = reference_headers[:len(df.columns)]

            # Normalize any date columns
            date_cols = [c for c in df.columns if "date" in c.lower()]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

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
    combined_transactions = pd.concat(transactions_list, ignore_index=True) if transactions_list else pd.DataFrame()

    # Sort by all date columns (if any exist)
    date_cols = [c for c in combined_transactions.columns if "date" in c.lower()]
    if date_cols:
        combined_transactions = combined_transactions.sort_values(
            by=date_cols, ascending=True, na_position="last"
        ).reset_index(drop=True)

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

    transactions, metadata = normalize_to_reference(excel_files)

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
