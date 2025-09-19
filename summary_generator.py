#!/usr/bin/env python3
"""
Summary Generator - Creates a standardized summary sheet from normalized Excel data.
Takes normalized Excel file and outputs a summary sheet matching the required format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import calendar

def load_normalized_data(normalized_file):
    """Load the consolidated data from the normalized output file."""
    try:
        # Load consolidated transactions
        transactions_df = pd.read_excel(normalized_file, sheet_name='Consolidated Transactions')
        print(f"[INFO] Loaded {len(transactions_df)} consolidated transactions")
        
        # Load consolidated metadata
        metadata_df = pd.read_excel(normalized_file, sheet_name='Consolidated Metadata')
        metadata_dict = dict(zip(metadata_df['Key'], metadata_df['Value']))
        print(f"[INFO] Loaded {len(metadata_dict)} metadata items")
        
        return transactions_df, metadata_dict
    except Exception as e:
        print(f"[ERROR] Could not load normalized data from {normalized_file}: {e}")
        return pd.DataFrame(), {}

def calculate_monthly_average_balance(transactions_df):
    """Calculate monthly average balance from transaction data."""
    if transactions_df.empty:
        return "NA"
    
    balance_cols = [col for col in transactions_df.columns if 'balance' in col.lower()]
    if not balance_cols:
        return "NA"
    
    balance_col = balance_cols[0]
    # Clean balance column (remove commas and convert to numeric)
    balances = transactions_df[balance_col].astype(str).str.replace(',', '').str.replace(' ', '')
    balances = pd.to_numeric(balances, errors='coerce')
    balances = balances.dropna()
    
    if balances.empty:
        return "NA"
    
    return round(balances.mean(), 2)

def calculate_monthly_surplus_balance(transactions_df):
    """Calculate monthly surplus balance (average balance above minimum)."""
    if transactions_df.empty:
        return "NA"
    
    balance_cols = [col for col in transactions_df.columns if 'balance' in col.lower()]
    if not balance_cols:
        return "NA"
    
    balance_col = balance_cols[0]
    # Clean balance column (remove commas and convert to numeric)
    balances = transactions_df[balance_col].astype(str).str.replace(',', '').str.replace(' ', '')
    balances = pd.to_numeric(balances, errors='coerce')
    balances = balances.dropna()
    
    if balances.empty:
        return "NA"
    
    min_balance = balances.min()
    avg_balance = balances.mean()
    surplus = avg_balance - min_balance
    
    return round(surplus, 2)

def detect_emi_patterns(transactions_df):
    """Detect EMI patterns in transactions."""
    if transactions_df.empty or 'Description' not in transactions_df.columns:
        return "FALSE"
    
    # Look for EMI-related keywords in descriptions
    emi_keywords = ['emi', 'loan', 'installment', 'repayment', 'monthly payment']
    descriptions = transactions_df['Description'].astype(str).str.lower()
    
    for keyword in emi_keywords:
        if descriptions.str.contains(keyword, na=False).any():
            return "TRUE"
    
    return "FALSE"

def detect_cheque_bounce(transactions_df):
    """Detect cheque bounce patterns in transactions."""
    if transactions_df.empty or 'Description' not in transactions_df.columns:
        return "FALSE"
    
    # Look for cheque bounce related keywords
    bounce_keywords = ['bounce', 'returned', 'dishonour', 'insufficient funds', 'cheque return']
    descriptions = transactions_df['Description'].astype(str).str.lower()
    
    for keyword in bounce_keywords:
        if descriptions.str.contains(keyword, na=False).any():
            return "TRUE"
    
    return "FALSE"

def get_account_type(metadata_dict):
    """Determine account type from metadata."""
    # This would need more sophisticated logic based on your data
    # For now, return UNK as shown in the template
    return "UNK"

def get_bank_name(metadata_dict):
    """Extract bank name from metadata."""
    # Look for bank-related fields in metadata
    bank_fields = ['bank', 'bank_name', 'institution']
    for field in bank_fields:
        if field in metadata_dict and metadata_dict[field]:
            return metadata_dict[field]
    
    return "BANK NOT FOUND"

def generate_monthwise_analysis(transactions_df):
    """Generate month-wise analysis of transactions."""
    if transactions_df.empty:
        return pd.DataFrame()
    
    # Convert Date column to datetime
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Extract month-year for grouping
    transactions_df['Month_Year'] = transactions_df['Date'].dt.to_period('M')
    
    # Clean Amount column
    transactions_df['Amount_Clean'] = transactions_df['Amount'].astype(str).str.replace(',', '').str.replace(' ', '')
    transactions_df['Amount_Clean'] = pd.to_numeric(transactions_df['Amount_Clean'], errors='coerce')
    
    # Clean Balance column
    transactions_df['Balance_Clean'] = transactions_df['Balance'].astype(str).str.replace(',', '').str.replace(' ', '')
    transactions_df['Balance_Clean'] = pd.to_numeric(transactions_df['Balance_Clean'], errors='coerce')
    
    monthly_data = []
    
    # Group by month
    for month_year, group in transactions_df.groupby('Month_Year'):
        month_name = f"{calendar.month_abbr[month_year.month]}-{str(month_year.year)[-2:]}"
        
        # Calculate metrics
        avg_balance = group['Balance_Clean'].mean() if not group['Balance_Clean'].empty else 0
        max_balance = group['Balance_Clean'].max() if not group['Balance_Clean'].empty else 0
        min_balance = group['Balance_Clean'].min() if not group['Balance_Clean'].empty else 0
        
        # Credit and Debit totals
        credit_total = group[group['Credit/Debit'] == 'CREDIT']['Amount_Clean'].sum() if 'CREDIT' in group['Credit/Debit'].values else 0
        debit_total = group[group['Credit/Debit'] == 'DEBIT']['Amount_Clean'].sum() if 'DEBIT' in group['Credit/Debit'].values else 0
        
        # Monthly surplus (Credit - Debit)
        monthly_surplus = credit_total - debit_total
        
        # Expense/Income ratio (Debit/Credit ratio)
        expense_income_ratio = debit_total / credit_total if credit_total > 0 else 0
        
        # Month maximum and minimum expenses (debits)
        debit_amounts = group[group['Credit/Debit'] == 'DEBIT']['Amount_Clean']
        max_expense = debit_amounts.max() if not debit_amounts.empty else 0
        min_expense = debit_amounts.min() if not debit_amounts.empty else 0
        
        # Month maximum and minimum income (credits)
        credit_amounts = group[group['Credit/Debit'] == 'CREDIT']['Amount_Clean']
        max_income = credit_amounts.max() if not credit_amounts.empty else 0
        min_income = credit_amounts.min() if not credit_amounts.empty else 0
        
        # Balance on specific dates (2nd, 5th, 10th)
        balance_2nd = get_balance_on_date(group, 2)
        balance_5th = get_balance_on_date(group, 5)
        balance_10th = get_balance_on_date(group, 10)
        
        monthly_data.append({
            'Month': month_name,
            'Average Bank Balance': round(avg_balance, 2),
            'Max Balance': round(max_balance, 2),
            'Min Balance': round(min_balance, 2),
            'Total Credit': round(credit_total, 2),
            'Total Debit': round(debit_total, 2),
            'Monthly Surplus': round(monthly_surplus, 2),
            'Expense / Income Ratio': round(expense_income_ratio, 2),
            'Month Maximum Expense': round(max_expense, 2),
            'Month Minimum Expense': round(min_expense, 2),
            'Month Maximum Income': round(max_income, 2),
            'Month Minimum Income': round(min_income, 2),
            'Balance as on 2nd': round(balance_2nd, 2),
            'Balance as on 5th': round(balance_5th, 2),
            'Balance as on 10th': round(balance_10th, 2)
        })
    
    # Create DataFrame
    monthly_df = pd.DataFrame(monthly_data)
    
    # Sort by month
    monthly_df = monthly_df.sort_values('Month')
    
    # Add Total row
    total_row = {
        'Month': 'Total',
        'Average Bank Balance': round(monthly_df['Average Bank Balance'].sum(), 2),
        'Max Balance': round(monthly_df['Max Balance'].sum(), 2),
        'Min Balance': round(monthly_df['Min Balance'].sum(), 2),
        'Total Credit': round(monthly_df['Total Credit'].sum(), 2),
        'Total Debit': round(monthly_df['Total Debit'].sum(), 2),
        'Monthly Surplus': round(monthly_df['Monthly Surplus'].sum(), 2),
        'Expense / Income Ratio': 0,  # Not meaningful for total
        'Month Maximum Expense': round(monthly_df['Month Maximum Expense'].sum(), 2),
        'Month Minimum Expense': round(monthly_df['Month Minimum Expense'].sum(), 2),
        'Month Maximum Income': round(monthly_df['Month Maximum Income'].sum(), 2),
        'Month Minimum Income': round(monthly_df['Month Minimum Income'].sum(), 2),
        'Balance as on 2nd': round(monthly_df['Balance as on 2nd'].sum(), 2),
        'Balance as on 5th': round(monthly_df['Balance as on 5th'].sum(), 2),
        'Balance as on 10th': round(monthly_df['Balance as on 10th'].sum(), 2)
    }
    
    # Add Consolidated row (averages)
    consolidated_row = {
        'Month': 'Consolidated',
        'Average Bank Balance': round(monthly_df['Average Bank Balance'].mean(), 2),
        'Max Balance': round(monthly_df['Max Balance'].mean(), 2),
        'Min Balance': round(monthly_df['Min Balance'].mean(), 2),
        'Total Credit': round(monthly_df['Total Credit'].mean(), 2),
        'Total Debit': round(monthly_df['Total Debit'].mean(), 2),
        'Monthly Surplus': round(monthly_df['Monthly Surplus'].mean(), 2),
        'Expense / Income Ratio': 0,  # Not meaningful for consolidated
        'Month Maximum Expense': round(monthly_df['Month Maximum Expense'].mean(), 2),
        'Month Minimum Expense': round(monthly_df['Month Minimum Expense'].mean(), 2),
        'Month Maximum Income': round(monthly_df['Month Maximum Income'].mean(), 2),
        'Month Minimum Income': round(monthly_df['Month Minimum Income'].mean(), 2),
        'Balance as on 2nd': round(monthly_df['Balance as on 2nd'].mean(), 2),
        'Balance as on 5th': round(monthly_df['Balance as on 5th'].mean(), 2),
        'Balance as on 10th': round(monthly_df['Balance as on 10th'].mean(), 2)
    }
    
    # Add total and consolidated rows
    monthly_df = pd.concat([monthly_df, pd.DataFrame([total_row]), pd.DataFrame([consolidated_row])], ignore_index=True)
    
    return monthly_df

def get_balance_on_date(month_group, day):
    """Get balance on a specific day of the month."""
    # Filter transactions for the specific day
    day_transactions = month_group[month_group['Date'].dt.day == day]
    
    if not day_transactions.empty:
        # Return the last balance for that day
        return day_transactions['Balance_Clean'].iloc[-1]
    else:
        # If no transactions on that day, find the closest previous day
        previous_days = month_group[month_group['Date'].dt.day < day]
        if not previous_days.empty:
            return previous_days['Balance_Clean'].iloc[-1]
        else:
            return 0

def generate_summary_sheet(normalized_file, output_file):
    """Generate the standardized summary sheet."""
    print(f"[INFO] Loading normalized data from {normalized_file}")
    
    # Load the consolidated data
    transactions_df, metadata_dict = load_normalized_data(normalized_file)
    
    if transactions_df.empty:
        print("[ERROR] No transaction data found in normalized file.")
        return
    
    # Create summary data
    summary_data = []
    
    # Customer Name
    customer_name = metadata_dict.get('account_name', 'NA')
    summary_data.append(['Customer Name', customer_name, 'NA'])
    
    # Account Number
    account_number = metadata_dict.get('account_number', 'NA')
    summary_data.append(['Account Number', account_number, 'FALSE'])
    
    # Monthly Average Balance
    monthly_avg_balance = calculate_monthly_average_balance(transactions_df)
    summary_data.append(['Monthly Average Balance', monthly_avg_balance, ''])
    
    # Monthly Surplus Balance
    monthly_surplus = calculate_monthly_surplus_balance(transactions_df)
    summary_data.append(['Monthly Surplus Balance', monthly_surplus, ''])
    
    # EMI Detected
    emi_detected = detect_emi_patterns(transactions_df)
    summary_data.append(['EMI Detected', emi_detected, ''])
    
    # Cheque Bounce
    cheque_bounce = detect_cheque_bounce(transactions_df)
    summary_data.append(['Cheque Bounce', cheque_bounce, ''])
    
    # Account Type
    account_type = get_account_type(metadata_dict)
    summary_data.append(['Account Type', account_type, ''])
    
    # Bank
    bank_name = get_bank_name(metadata_dict)
    summary_data.append(['Bank', bank_name, 'NA'])
    
    # Opening Balance
    balance_cols = [col for col in transactions_df.columns if 'balance' in col.lower()]
    if balance_cols:
        opening_balance = transactions_df[balance_cols[0]].iloc[0]
        summary_data.append(['Opening Balance', opening_balance, 'Yes'])
    else:
        summary_data.append(['Opening Balance', 'NA', ''])
    
    # Closing Balance
    if balance_cols:
        closing_balance = transactions_df[balance_cols[0]].iloc[-1]
        summary_data.append(['Closing Balance', closing_balance, ''])
    else:
        summary_data.append(['Closing Balance', 'NA', ''])
    
    # Start Date and End Date
    date_cols = [col for col in transactions_df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        # Convert dates for proper sorting
        txn_dates = pd.to_datetime(transactions_df[date_col], format='%d/%m/%Y', errors='coerce')
        txn_dates = txn_dates.dropna()
        if not txn_dates.empty:
            txn_dates_sorted = txn_dates.sort_values()
            start_date = txn_dates_sorted.iloc[0].strftime('%d/%m/%Y')
            end_date = txn_dates_sorted.iloc[-1].strftime('%d/%m/%Y')
            summary_data.append(['Start Date', start_date, ''])
            summary_data.append(['End Date', end_date, ''])
        else:
            summary_data.append(['Start Date', 'NA', ''])
            summary_data.append(['End Date', 'NA', ''])
    else:
        summary_data.append(['Start Date', 'NA', ''])
        summary_data.append(['End Date', 'NA', ''])
    
    # IFSC Code
    ifsc_code = metadata_dict.get('ifsc_code', 'NA')
    summary_data.append(['IFSC Code', ifsc_code, ''])
    
    # MICR Code
    micr_code = metadata_dict.get('micr_code', 'NA')
    summary_data.append(['MICR Code', micr_code, ''])
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data, columns=['Item', 'Details', 'Verification'])
    
    # Generate month-wise analysis
    print("[INFO] Generating month-wise analysis...")
    monthly_analysis = generate_monthwise_analysis(transactions_df.copy())
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary - Scorecard')
        monthly_analysis.to_excel(writer, index=False, sheet_name='Month-wise Analysis')
        transactions_df.to_excel(writer, index=False, sheet_name='Xns')
    
    print(f"[SUCCESS] Saved summary sheet to {output_file}")
    print(f"[INFO] Summary contains {len(summary_data)} items")
    print(f"[INFO] Month-wise analysis contains {len(monthly_analysis)} months")
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY - SCORECARD")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Display month-wise analysis
    print("\n" + "="*60)
    print("MONTH-WISE ANALYSIS")
    print("="*60)
    print(monthly_analysis.to_string(index=False))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate standardized summary sheet from normalized data')
    parser.add_argument('normalized_file', help='Path to the normalized consolidated Excel file (output from n3.py)')
    parser.add_argument('--output', default='./summary_scorecard.xlsx', help='Output Excel file path')
    
    args = parser.parse_args()
    generate_summary_sheet(args.normalized_file, args.output)
