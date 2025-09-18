import os
import pandas as pd
from collections import Counter

def aggregate_metadata(excel_files):
    metadata_list = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, sheet_name='Metadata')
            metadata_dict = dict(zip(df['Key'], df['Value']))
            metadata_list.append(metadata_dict)
        except Exception as e:
            print(f"Warning: Could not read metadata from {file}: {e}")

    # Take the most common value for each key
    all_keys = set(k for md in metadata_list for k in md.keys())
    aggregated_metadata = {}
    for key in all_keys:
        values = [md.get(key, '') for md in metadata_list if md.get(key)]
        most_common = Counter(values).most_common(1)
        aggregated_metadata[key] = most_common[0][0] if most_common else ''

    return aggregated_metadata


def aggregate_transactions(excel_files):
    transactions_df_list = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, sheet_name='Transactions')
            transactions_df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read transactions from {file}: {e}")

    # Combine all transaction rows into one DataFrame
    if transactions_df_list:
        combined_df = pd.concat(transactions_df_list, ignore_index=True)
        # Sort by date if a 'Txn Date' column exists
        if 'Txn Date' in combined_df.columns:
            combined_df['Txn Date'] = pd.to_datetime(combined_df['Txn Date'], errors='coerce')
            combined_df = combined_df.sort_values(by='Txn Date')
        return combined_df
    return pd.DataFrame()


def derive_summary_fields(transactions_df):
    summary = {}

    if not transactions_df.empty and 'Txn Date' in transactions_df.columns:
        txn_dates = transactions_df['Txn Date'].dropna().sort_values()
        summary['Statement Period'] = f"{txn_dates.iloc[0].strftime('%d/%m/%Y')} - {txn_dates.iloc[-1].strftime('%d/%m/%Y')}"

    # Derive Opening and Closing Balance if 'Balance' column exists
    if 'Balance' in transactions_df.columns:
        summary['Opening Balance'] = transactions_df['Balance'].iloc[0]
        summary['Closing Balance'] = transactions_df['Balance'].iloc[-1]

    return summary


def generate_summary(input_folder, output_file):
    excel_files = []
    for file in os.listdir(input_folder):
        print(f"Found file: {file}")  # Debug print
        if file.lower().endswith('.xlsx') and not file.startswith('.'):
            excel_files.append(os.path.join(input_folder, file))

    print(f"Excel files to process: {excel_files}")

    # print(f"Found {len(excel_files)} Excel files to process.")

    metadata = aggregate_metadata(excel_files)
    transactions_df = aggregate_transactions(excel_files)
    derived_summary = derive_summary_fields(transactions_df)

    # Merge metadata and derived fields
    final_summary = {**metadata, **derived_summary}

    # Save Summary
    summary_df = pd.DataFrame(list(final_summary.items()), columns=['Item', 'Details'])
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        transactions_df.to_excel(writer, index=False, sheet_name='Consolidated Transactions')

    print(f"Summary and consolidated transactions saved to {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate Parsed Excel Files into Summary and Consolidated Transactions')
    parser.add_argument('input_folder', help='Folder containing parsed Excel files')
    parser.add_argument('--output', default='./aggregated_summary.xlsx', help='Output Excel file path')
    args = parser.parse_args()

    generate_summary(args.input_folder, args.output)
