#!/usr/bin/env python3
"""
Summary Generator - Creates a standardized summary sheet from normalized Excel data.
Takes normalized Excel file and outputs a summary sheet matching the required format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from rapidfuzz import fuzz
from rc import similarity_threshold

similarity_threshold =70  #for calculating the recurring debit and credit sheet

def _to_numeric(series):
    """Helper: clean commas/spaces and convert to numeric."""
    s = series.astype(str).str.replace(',', '').str.replace(' ', '')
    return pd.to_numeric(s, errors='coerce')


def generate_scoring_details(transactions_df):
    """Compute scoring metrics similar to the Scoring Details sheet.

    Expects standard columns: `Date`, `Credit/Debit`, `Description`, `Amount`, `Balance`.
    Returns a 2-column DataFrame with Description and Value.
    """
    if transactions_df.empty:
        return pd.DataFrame({
            'Description': [
                'Monthly Average Inflow', 'Monthly Average Outflow', 'Average Credit Transactions',
                'Average Debit Transactions', 'Total Credit Amount', 'Total Debit Amount',
                'Total Count of Credit Transactions', 'Total Count of Debit Transactions',
                'Monthly Average Surplus', 'Fixed Obligation To Income Ratio',
                'Maximum Balance', 'Minimum Balance', 'Maximum Credit', 'Minimum Credit',
                'Maximum Debit', 'Minimum Debit',
                'Month end balance in last 90 days', 'Month end balance in last 180 days',
                'Number of Cash withdrawal in last 3 months', 'Number of Cash withdrawal in last 6 months',
                'Count of interest credited in last 3 months', 'Amount of interest credited in last 3 months',
                'Count of interest credited in last 6 months', 'Amount of interest credited in last 6 months',
                'Count of Cheque Bounce in last 3 months', 'Amount of Cheque Bounce in last 3 months',
                'Count of Cheque Bounce in last 6 months', 'Amount of Cheque Bounce in last 6 months',
                'Velocity - (Sum of debits and credits) /AMB in the last 3 months',
                'Velocity - (Sum of debits and credits) /AMB in the last 6 months'
            ],
            'Value': ['NA'] * 30
        })

    df = transactions_df.copy()
    # Dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    # Amounts signed by Credit/Debit
    df['Amount_Clean'] = _to_numeric(df['Amount'])
    # Balance
    df['Balance_Clean'] = _to_numeric(df['Balance']) if 'Balance' in df.columns else pd.NA

    # Derive signed amounts
    df['Credit_Amount'] = df.loc[df['Credit/Debit'] == 'CREDIT', 'Amount_Clean']
    df['Debit_Amount'] = df.loc[df['Credit/Debit'] == 'DEBIT', 'Amount_Clean']

    # Month key
    df['Month'] = df['Date'].dt.to_period('M')

    # Monthly aggregates
    monthly = df.groupby('Month').agg(
        credit_sum=pd.NamedAgg(column='Credit_Amount', aggfunc='sum'),
        debit_sum=pd.NamedAgg(column='Debit_Amount', aggfunc='sum'),
        credit_cnt=pd.NamedAgg(column='Credit_Amount', aggfunc=lambda s: s.notna().sum()),
        debit_cnt=pd.NamedAgg(column='Debit_Amount', aggfunc=lambda s: s.notna().sum()),
        avg_balance=pd.NamedAgg(column='Balance_Clean', aggfunc='mean')
    ).fillna(0)

    # Monthly Average Inflow/Outflow (mean of monthly sums)
    monthly_avg_inflow = round(monthly['credit_sum'].mean(), 2) if not monthly.empty else 0
    monthly_avg_outflow = round(monthly['debit_sum'].mean(), 2) if not monthly.empty else 0

    # Average Credit/Debit Transactions (mean of monthly amounts)
    avg_credit_txn = round(monthly['credit_sum'].mean(), 2) if not monthly.empty else 0
    avg_debit_txn = round(monthly['debit_sum'].mean(), 2) if not monthly.empty else 0

    # Totals
    total_credit_amount = round(df['Credit_Amount'].sum(skipna=True), 2)
    total_debit_amount = round(df['Debit_Amount'].sum(skipna=True), 2)
    total_credit_count = int(df['Credit_Amount'].notna().sum())
    total_debit_count = int(df['Debit_Amount'].notna().sum())

    # Monthly Average Surplus = mean(credit_sum - debit_sum)
    monthly_surplus = round((monthly['credit_sum'] - monthly['debit_sum']).mean(), 2) if not monthly.empty else 0

    # Foir placeholder (needs liabilities to compute). Mark as NA to match screenshot.
    foi_ratio = 'NA'

    # Balance extremes
    max_balance = round(df['Balance_Clean'].max(skipna=True), 2) if 'Balance_Clean' in df else 'NA'
    min_balance = round(df['Balance_Clean'].min(skipna=True), 2) if 'Balance_Clean' in df else 'NA'

    # Credit/Debit extremes
    max_credit = round(df['Credit_Amount'].max(skipna=True), 2) if df['Credit_Amount'].notna().any() else 0
    min_credit = round(df['Credit_Amount'].min(skipna=True), 2) if df['Credit_Amount'].notna().any() else 0
    max_debit = round(df['Debit_Amount'].max(skipna=True), 2) if df['Debit_Amount'].notna().any() else 0
    min_debit = round(df['Debit_Amount'].min(skipna=True), 2) if df['Debit_Amount'].notna().any() else 0

    # Month end balances: last balance of each month
    month_end_balance = (
        df.sort_values('Date').groupby('Month')['Balance_Clean'].last()
    )
    last_90d_cut = df['Date'].max() - pd.Timedelta(days=90)
    last_180d_cut = df['Date'].max() - pd.Timedelta(days=180)
    meb_90 = round(month_end_balance[month_end_balance.index.to_timestamp() >= last_90d_cut].mean(), 2) if not month_end_balance.empty else 0
    meb_180 = round(month_end_balance[month_end_balance.index.to_timestamp() >= last_180d_cut].mean(), 2) if not month_end_balance.empty else 0

    # Helpers for last N months
    def last_n_months(n):
        if df.empty:
            return df.iloc[0:0]
        last_month = df['Month'].max()
        months = [(last_month - i).strftime('%Y-%m') for i in range(n)]
        return df[df['Month'].astype(str).isin(months)]

    # Cash withdrawals: infer by Description contains 'cash' and Debit
    desc = df['Description'].astype(str).str.lower() if 'Description' in df.columns else pd.Series([], dtype=str)
    cash_mask = desc.str.contains('cash', na=False)
    def count_cash_withdrawals(n):
        sub = last_n_months(n)
        if sub.empty:
            return 0
        return int(((sub['Credit/Debit'] == 'DEBIT') & cash_mask.loc[sub.index]).sum())

    cash_3m = count_cash_withdrawals(3)
    cash_6m = count_cash_withdrawals(6)

    # Interest credited: Description contains 'interest' and CREDIT
    interest_mask = desc.str.contains('interest', na=False)
    def stats_interest(n):
        sub = last_n_months(n)
        if sub.empty:
            return 0, 0.0
        m = (sub['Credit/Debit'] == 'CREDIT') & interest_mask.loc[sub.index]
        count = int(m.sum())
        amt = round(sub.loc[m, 'Amount_Clean'].sum(), 2)
        return count, amt

    ic3_cnt, ic3_amt = stats_interest(3)
    ic6_cnt, ic6_amt = stats_interest(6)

    # Cheque bounce: Description contains keywords
    bounce_mask = desc.str.contains('bounce|returned|dishonour|insufficient funds|cheque return', na=False)
    def stats_bounce(n):
        sub = last_n_months(n)
        if sub.empty:
            return 0, 0.0
        m = bounce_mask.loc[sub.index]
        count = int(m.sum())
        amt = round(sub.loc[m, 'Amount_Clean'].sum(), 2)
        return count, amt

    cb3_cnt, cb3_amt = stats_bounce(3)
    cb6_cnt, cb6_amt = stats_bounce(6)

    # Velocity = (sum of debits and credits)/AMB for last N months
    def velocity(n):
        sub = last_n_months(n)
        if sub.empty:
            return 0.0
        # AMB approximate as mean monthly avg balances over N months
        sub_monthly = sub.groupby(sub['Date'].dt.to_period('M')).agg(
            sum_amt=pd.NamedAgg(column='Amount_Clean', aggfunc=lambda s: s.abs().sum()),
            amb=pd.NamedAgg(column='Balance_Clean', aggfunc='mean')
        ).fillna(0)
        amb = sub_monthly['amb'].replace(0, pd.NA).mean(skipna=True)
        total_turnover = sub_monthly['sum_amt'].sum()
        if pd.isna(amb) or amb == 0:
            return 0.0
        return round(total_turnover / amb, 2)

    velocity_3 = velocity(3)
    velocity_6 = velocity(6)

    rows = [
        ('Monthly Average Inflow', monthly_avg_inflow),
        ('Monthly Average Outflow', monthly_avg_outflow),
        ('Average Credit Transactions', avg_credit_txn),
        ('Average Debit Transactions', avg_debit_txn),
        ('Total Credit Amount', round(total_credit_amount, 2)),
        ('Total Debit Amount', round(total_debit_amount, 2)),
        ('Total Count of Credit Transactions', total_credit_count),
        ('Total Count of Debit Transactions', total_debit_count),
        ('Monthly Average Surplus', monthly_surplus),
        ('Fixed Obligation To Income Ratio', foi_ratio),
        ('Maximum Balance', max_balance),
        ('Minimum Balance', min_balance),
        ('Maximum Credit', max_credit),
        ('Minimum Credit', min_credit),
        ('Maximum Debit', max_debit),
        ('Minimum Debit', min_debit),
        ('Month end balance in last 90 days', meb_90),
        ('Month end balance in last 180 days', meb_180),
        ('Number of Cash withdrawal in last 3 months', cash_3m),
        ('Number of Cash withdrawal in last 6 months', cash_6m),
        ('Count of interest credited in last 3 months', ic3_cnt),
        ('Amount of interest credited in last 3 months', ic3_amt),
        ('Count of interest credited in last 6 months', ic6_cnt),
        ('Amount of interest credited in last 6 months', ic6_amt),
        ('Count of Cheque Bounce in last 3 months', cb3_cnt),
        ('Amount of Cheque Bounce in last 3 months', cb3_amt),
        ('Count of Cheque Bounce in last 6 months', cb6_cnt),
        ('Amount of Cheque Bounce in last 6 months', cb6_amt),
        ('Velocity - (Sum of debits and credits) /AMB in the last 3 months', velocity_3),
        ('Velocity - (Sum of debits and credits) /AMB in the last 6 months', velocity_6),
    ]

    return pd.DataFrame(rows, columns=['Description', 'Value'])

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
    print("sgwrg",transactions_df)
    
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
    
    # Sort months chronologically by parsing Month like 'Sep-24'
    if not monthly_df.empty and 'Month' in monthly_df.columns:
        monthly_df['_order'] = pd.to_datetime(monthly_df['Month'], format='%b-%y', errors='coerce')
        monthly_df = monthly_df.sort_values('_order').drop(columns=['_order'])
    
    
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


def generate_eod_balances(transactions_df):
    """Generate End-Of-Day balance matrix: rows are Day 1..31, columns are Month-Year.
    For days without transactions, carry forward the previous day's balance."""
    if transactions_df.empty:
        return pd.DataFrame()

    # Prepare dates and clean balance
    df = transactions_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    balance_cols = [col for col in df.columns if 'balance' in col.lower()]
    if not balance_cols:
        return pd.DataFrame()
    balance_col = balance_cols[0]
    df['Balance_Clean'] = (
        df[balance_col].astype(str).str.replace(',', '').str.replace(' ', '')
    )
    df['Balance_Clean'] = pd.to_numeric(df['Balance_Clean'], errors='coerce')

    # Drop rows without valid date
    df = df.dropna(subset=['Date'])
    if df.empty:
        return pd.DataFrame()

    # Month period
    df['Month_Year'] = df['Date'].dt.to_period('M')

    # Determine chronological list of months present
    months = sorted(df['Month_Year'].unique())

    # Build EOD matrix
    max_days = 31
    data = {'Day': list(range(1, max_days + 1))}

    prev_month_last_balance = np.nan
    for period in months:
        month_df = df[df['Month_Year'] == period].sort_values('Date')
        # Map day -> last balance that day
        day_last_balance = (
            month_df.groupby(month_df['Date'].dt.day)['Balance_Clean'].last()
        )
        eod_values = []
        # Initialize with previous month's closing balance so the new month starts carried-forward
        prev_balance = prev_month_last_balance
        # Determine number of days in month
        year = int(period.year)
        month = int(period.month)
        _, days_in_month = calendar.monthrange(year, month)
        for day in range(1, max_days + 1):
            if day <= days_in_month:
                if day in day_last_balance.index:
                    prev_balance = day_last_balance.loc[day]
                # if no transaction, carry forward previous balance (may still be NaN before first txn)
                eod_values.append(prev_balance)
            else:
                # days beyond actual month -> keep as NaN
                eod_values.append(np.nan)

        col_label = f"{calendar.month_abbr[month]} - {year}"
        data[col_label] = eod_values

        # Update previous month last balance for carry-forward to the next month
        # Use the last non-null within the month's valid days
        month_last_valid = pd.Series(eod_values[:days_in_month]).dropna()
        if not month_last_valid.empty:
            prev_month_last_balance = month_last_valid.iloc[-1]
        # If still NaN (no data this month), keep previous month's value unchanged

    eod_df = pd.DataFrame(data)

    return eod_df

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


def generate_recurring_credit_debit(transactions_df, val):

    if transactions_df.empty:
        return "NA"           
    df_credit = transactions_df[transactions_df['Credit/Debit'].str.upper() == val].copy()

    # Function to assign group based on description similarity
    groups = []

    def assign_group(desc):
        for i, group in enumerate(groups):
            # Compare with first element of the group
            if fuzz.token_set_ratio(desc, group[0]) >= similarity_threshold:
                group.append(desc)
                return i+1
        # If no match, create a new group
        groups.append([desc])
        return len(groups)

    # Assign group index for each description
    df_credit['Group'] = df_credit['Description'].apply(assign_group)

    # Create canonical name for each group (first description in the group)
    # canonical_names = {i: group[0] for i, group in enumerate(groups)}
    # df_credit['Canonical_Description'] = df_credit['Group'].map(canonical_names)

    # Filter out groups with only 1 transaction
    group_counts = df_credit['Group'].value_counts()
    valid_groups = group_counts[group_counts > 1].index
    df_credit = df_credit[df_credit['Group'].isin(valid_groups)].copy()

    # Reassign group numbers sequentially
    old_to_new = {old: new+1 for new, old in enumerate(sorted(valid_groups))}
    df_credit['Group'] = df_credit['Group'].map(old_to_new)

    # Re-map canonical descriptions after removing single-entry groups
    # new_canonical_names = {new: canonical_names[old] for old, new in old_to_new.items()}
    # df_credit['Canonical_Description'] = df_credit['Group'].map(new_canonical_names)


    columns_order = ['Date', 'Group', 'Description', 'Amount', 'Balance']
    output_df = df_credit[columns_order]
    recurring_credit_df = output_df.sort_values(by=['Group', 'Description'])
    # Aggregate amounts per canonical description
    return recurring_credit_df


    print(f"Grouped credit transactions saved to {output_file}")


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
    
    # Generate EOD Balances
    print("[INFO] Generating EOD balances...")
    eod_balances = generate_eod_balances(transactions_df.copy())

    # Generate Scoring Details
    print("[INFO] Generating Scoring Details...")
    scoring_df = generate_scoring_details(transactions_df.copy())

    print("[INFO] Generating Recurring Credit Details...")
    recurring_credit_df = generate_recurring_credit_debit(transactions_df.copy(), val='CREDIT')

    print("[INFO] Generating Recurring Debit Details...")
    recurring_debit_df = generate_recurring_credit_debit(transactions_df.copy(), val='DEBIT')
    
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary - Scorecard')
        monthly_analysis.to_excel(writer, index=False, sheet_name='Month-wise Analysis')
        if not scoring_df.empty:
            scoring_df.to_excel(writer, index=False, sheet_name='Scoring Details')
        if not eod_balances.empty:
            eod_balances.to_excel(writer, index=False, sheet_name='EOD Balances')
        if not recurring_credit_df.empty:
            recurring_credit_df.to_excel(writer, index=False, sheet_name='Recurring Credit')  
        if not recurring_debit_df.empty:
            recurring_debit_df.to_excel(writer, index=False, sheet_name='Recurring Debit')      
        transactions_df.to_excel(writer, index=False, sheet_name='Xns')

    
    print(f"[SUCCESS] Saved summary sheet to {output_file}")
    print(f"[INFO] Summary contains {len(summary_data)} items")
    print(f"[INFO] Month-wise analysis contains {len(monthly_analysis)} months")
    if 'Day' in (eod_balances.columns if isinstance(eod_balances, pd.DataFrame) else []):
        print(f"[INFO] EOD Balances generated for {len([c for c in eod_balances.columns if c != 'Day'])} months")
    
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
