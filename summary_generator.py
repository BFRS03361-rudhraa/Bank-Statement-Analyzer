#!/usr/bin/env python3
"""
Summary Generator - Creates a standardized summary sheet from normalized Excel data.
Takes normalized Excel file and outputs a summary sheet matching the required format.
"""

from fsspec import transaction
import pandas as pd
from fraudsheet import df_duplicates
import numpy as np
from datetime import datetime
import calendar
from rapidfuzz import fuzz
import holidays
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


    columns_order = ['Date', 'Group','Credit/Debit', 'Description', 'Amount', 'Balance']
    output_df = df_credit[columns_order]
    recurring_credit_df = output_df.sort_values(by=['Group', 'Description'])
    # Aggregate amounts per canonical description
    return recurring_credit_df


    print(f"Grouped credit transactions saved to {output_file}")

def generate_return_txn(transactions_df):
    return_keywords = [
        "RETURN",
        "ACCOUNT DOES NOT EXIST",
        "INCORRECT ACCOUNT NUMBER",
        "PAYMENT STOPPED",
        "Closed Account"
    ]

    columns = ['Date', 'Amount', 'Balance', 'Description', 
           'mode', 'category', 'category_2', 'Fund', 'bounce']

    def detect_mode(Description):
        desc = str(Description).upper()
        if "NEFT" in desc:
            return "NEFT"
        elif "RTGS" in desc:
            return "RTGS"
        else:
            return "Other"
    # Read transactions
    df=transactions_df
    # Ensure description is string
    df['Description'] = df['Description'].astype(str)

    # Flag return transactions
    df['is_return'] = df['Description'].str.upper().str.contains("|".join(return_keywords))

    # Separate original transactions and returns
    original_txns = df[~df['is_return']].copy().reset_index(drop=True)
    return_txns = df[df['is_return']].copy().reset_index(drop=True)

    # Track which return transactions have been matched
    return_txns['matched'] = False

    return_txn_rows = []

    for idx, orig in original_txns.iterrows():
        # Try to find a matching return transaction
        # Match by amount (and optionally by mode/type if available)
        matched = return_txns[
            (~return_txns['matched']) &
            (return_txns['Amount'] == orig['Amount']) & (return_txns['Date'] == orig['Date'])
        ]
        
        if not matched.empty:
            # Found a matching return
            ret = matched.iloc[0]
            return_txns.at[ret.name, 'matched'] = True  # mark as used

            # Add original transaction row
            orig_row = {
                'Date': orig['Date'],
                'Amount': orig['Amount'],
                'Balance': orig['Balance'],
                'Description': orig['Description'],
                'mode': detect_mode(orig['Description']),
                'category': 'Payments',
                'category_2': None,
                'Fund': 'FUND SUFFICIENT',
                'bounce': 'Inward Return'
            }
            return_txn_rows.append(orig_row)

            # Add return transaction row
            return_row = {
                'Date': ret['Date'],
                'Amount': ret['Amount'],
                'Balance': ret['Balance'],
                'Description': ret['Description'],
                'mode': detect_mode(ret['Description']),
                'category': 'Return',
                'category_2': None,
                'Fund': 'FUND SUFFICIENT',
                'bounce': 'Inward Return'
            }
            return_txn_rows.append(return_row)

    # Convert to DataFrame and save
    return_txn_df = pd.DataFrame(return_txn_rows, columns=columns)
    if not return_txn_df.empty:
        return_df = return_txn_df
    else: 
        return_df = pd.DataFrame()
    return return_df

def generate_duplicates(recurring_credit_df , recurring_debit_df):

    credit_file = recurring_credit_df
    debit_file = recurring_debit_df
    max_differences = 2  # Maximum number of column differences to consider as duplicate
# ----------------------------

    def clean_amount(amount_str):
        """Clean amount string and convert to float"""
        if pd.isna(amount_str):
            return np.nan
        try:
            # Remove commas and convert to float
            clean_str = str(amount_str).replace(',', '')
            return float(clean_str)
        except:
            return np.nan

    def compare_rows(row1, row2, columns_to_compare):
        """
        Compare two rows with specific logic:
        - Amount MUST match exactly
        - Description should match mostly (minor differences allowed)
        - Other fields should match mostly
        - Balance is excluded from comparison
        Returns: (is_duplicate, differences_list, reason)
        """
        
        differences = []
        is_duplicate = True
        reason = ""
        
        # First check: Amount MUST match exactly (no tolerance)
        amount1 = row1.get('Amount', np.nan)
        amount2 = row2.get('Amount', np.nan)
        
        if pd.isna(amount1) or pd.isna(amount2):
            if not (pd.isna(amount1) and pd.isna(amount2)):
                return False, ["Amount: One is NaN"], "Amount mismatch (NaN)"
        else:
            if abs(float(amount1) - float(amount2)) > 0.01:  # Very small tolerance for floating point
                return False, [f"Amount: {amount1} vs {amount2}"], "Amount mismatch"
        
        # Second check: Description similarity (should be mostly similar)
        desc1 = str(row1.get('Description', ''))
        desc2 = str(row2.get('Description', ''))
        desc_similarity = fuzz.token_set_ratio(desc1, desc2)
        
        if desc_similarity < 80:  # Description should be at least 80% similar
            return False, [f"Description similarity: {desc_similarity}%"], "Description too different"
        
        # Check other fields (excluding Balance and Canonical_Description)
        other_diff_count = 0
        for col in columns_to_compare:
            if col in ['Balance', 'Canonical_Description']:
                continue  # Skip these columns
                
            val1 = row1[col]
            val2 = row2[col]
            
            # Handle NaN values
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2):
                other_diff_count += 1
                differences.append(f"{col}: {val1} vs {val2}")
            # Compare values
            elif val1 != val2:
                other_diff_count += 1
                differences.append(f"{col}: {val1} vs {val2}")
        
        # Allow only 1 difference in other fields (excluding Balance)
        if other_diff_count > 1:
            return False, differences, f"Too many differences in other fields: {other_diff_count}"
        
        # If we reach here, it's a duplicate
        if other_diff_count > 0:
            differences.append(f"Other differences: {other_diff_count}")
        
        return True, differences, "Duplicate found"

    def find_duplicates(df, max_differences=2):
        """
        Find duplicate rows based on refined comparison logic:
        - Amount MUST match exactly
        - Description should be mostly similar (80%+)
        - Other fields can have max 1 difference
        - Balance is excluded from comparison
        """
        print(f"Analyzing {len(df)} rows for duplicates...")
        
        # Get columns to compare (exclude index, ID columns, canonical description, and balance)
        columns_to_compare = [col for col in df.columns if col not in ['index', 'id', 'Index', 'ID', 'Canonical_Description', 'Balance']]
        
        print(f"Comparing columns: {columns_to_compare}")
        print("Criteria: Amount must match exactly, Description 80%+ similar, Other fields max 1 difference")
        
        duplicate_groups = []
        processed_indices = set()
        group_id = 1
        
        for i in range(len(df)):
            if i in processed_indices:
                continue
                
            current_group = [i]
            processed_indices.add(i)
            
            # Compare with all subsequent rows
            for j in range(i + 1, len(df)):
                if j in processed_indices:
                    continue
                    
                is_duplicate, differences, reason = compare_rows(df.iloc[i], df.iloc[j], columns_to_compare)
                
                # If it's a duplicate according to our refined logic
                if is_duplicate:
                    current_group.append(j)
                    processed_indices.add(j)
            
            # Only keep groups with 2 or more rows
            if len(current_group) >= 2:
                duplicate_groups.append({
                    'group_id': group_id,
                    'indices': current_group,
                    'size': len(current_group)
                })
                group_id += 1
        
        return duplicate_groups

# Load data
    print("Loading grouped credit and debit files...")
    df_credits = credit_file
    df_debits = debit_file

    print(f"Credits: {len(df_credits)} rows")
    print(f"Debits: {len(df_debits)} rows")

    # Clean amount columns
    df_credits['Amount'] = df_credits['Amount'].apply(clean_amount)
    df_debits['Amount'] = df_debits['Amount'].apply(clean_amount)

    # Find duplicates in credits
    print("\n=== ANALYZING CREDITS ===")
    credit_duplicates = find_duplicates(df_credits, max_differences)
    print(f"Found {len(credit_duplicates)} duplicate groups in credits")

    # Find duplicates in debits
    print("\n=== ANALYZING DEBITS ===")
    debit_duplicates = find_duplicates(df_debits, max_differences)
    print(f"Found {len(debit_duplicates)} duplicate groups in debits")


    # Create simple output with just duplicate transactions grouped together

    # Collect all duplicate transactions in order
    all_duplicate_rows = []

    # Process credit duplicates
    for group in credit_duplicates:
        group_rows = []
        for idx in group['indices']:
            row = df_credits.iloc[idx].copy()
            row['Credit/Debit'] = 'CREDIT'
            row['Indicator']= 'Data Duplicity'

            group_rows.append(row)
        
        # Sort by original index to maintain order
        group_rows.sort(key=lambda x: x.name if hasattr(x, 'name') else 0)
        all_duplicate_rows.extend(group_rows)

    # Process debit duplicates  
    for group in debit_duplicates:
        group_rows = []
        for idx in group['indices']:
            row = df_debits.iloc[idx].copy()
            row['Credit/Debit'] = 'DEBIT'
            row['Indicator']= 'Data Duplicity'

            group_rows.append(row)
        
        # Sort by original index to maintain order
        group_rows.sort(key=lambda x: x.name if hasattr(x, 'name') else 0)
        all_duplicate_rows.extend(group_rows)

    # Create DataFrame from duplicate rows
    # if all_duplicate_rows:
    #     duplicates_df = pd.DataFrame(all_duplicate_rows)
        
    #     # Select only the original columns (remove any extra columns we added)
    #     original_columns = ['Date', 'Transaction_Type','Description', 'Amount', 'Balance']
    #     available_columns = [col for col in original_columns if col in duplicates_df.columns]
        
    #     # Save simple output
    #     duplicates_df[available_columns].to_excel(output_file, index=False)
    #     print(f"Saved {len(duplicates_df)} duplicate transactions to {output_file}")
    # else:
    #     print("No duplicate transactions found")


    print(f"\n=== SUMMARY ===")
    print(f"Credit duplicate groups: {len(credit_duplicates)}")
    print(f"Debit duplicate groups: {len(debit_duplicates)}")
    print(f"Total duplicate transactions: {len(all_duplicate_rows)}")
    all_duplicate_df =pd.DataFrame(all_duplicate_rows)

    header_row = pd.DataFrame([{
                    'Indicator':'','Date': '', 'Credit/Debit': '', 'Description': '', 'Amount': '', 'Balance': ''
                }])

                # reorder cols to match
    cols = ['Indicator', 'Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']
    if all_duplicate_df.empty:
        print("[INFO] No duplicates found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Indicator", "Date", "Credit/Debit", 
                                     "Description", "Amount", "Balance"])
    all_duplicate_rows = all_duplicate_df[cols]

    dup_df = pd.concat([header_row, all_duplicate_rows], ignore_index=True)
    
    all_dup_rows = pd.DataFrame(dup_df)

    return all_dup_rows


def generate_fraud_sheet(transactions_df, duplicates_df):

    df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
    df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

    
# ---------- CONFIG ----------
    # credit_file = recurring_credit_df
    # debit_file = recurring_debit_df
    duplicates_file = duplicates_df
    # ----------------------------


    def analyze_data_duplicity():
        """Analyze data duplicity from the duplicates file"""
        try:
            df_duplicates = pd.read_excel(duplicates_file)
            
            # Count duplicates by transaction type
            credit_duplicates = len(df_duplicates[df_duplicates['Transaction_Type'] == 'CREDIT'])
            debit_duplicates = len(df_duplicates[df_duplicates['Transaction_Type'] == 'DEBIT'])
            
            # Analyze duplicate patterns
            duplicate_analysis = {
                'Total_Duplicate_Transactions': len(df_duplicates),
                'Credit_Duplicates': credit_duplicates,
                'Debit_Duplicates': debit_duplicates,
                'Duplicate_Percentage': (len(df_duplicates) / (270 * 2)) * 100,  # 270 rows each for credit and debit
                'Status': 'YES' if len(df_duplicates) > 0 else 'NO'
            }
            
            return duplicate_analysis
        except Exception as e:
            return {
                'Total_Duplicate_Transactions': 0,
                'Credit_Duplicates': 0,
                'Debit_Duplicates': 0,
                'Duplicate_Percentage': 0,
                'Status': 'NO',
                'Error': str(e)
            }

    def analyze_balance_reconciliation(transactions_df):
        """Analyze balance reconciliation issues"""

        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        credit_amounts = pd.to_numeric(df_credits['Amount'].astype(str).str.replace(',', ''), errors='coerce')
        debit_amounts = pd.to_numeric(df_debits['Amount'].astype(str).str.replace(',', ''), errors='coerce')
        
        credit_total = credit_amounts.sum() if 'Amount' in df_credits.columns else 0
        debit_total = debit_amounts.sum() if 'Amount' in df_debits.columns else 0
        
        balance_diff = abs(credit_total - debit_total)
        balance_diff_percentage = (balance_diff / max(credit_total, debit_total)) * 100 if max(credit_total, debit_total) > 0 else 0
        
        return {
            'Credit_Total': credit_total,
            'Debit_Total': debit_total,
            'Difference': balance_diff,
            'Difference_Percentage': balance_diff_percentage,
            'Status': 'YES' if balance_diff_percentage > 1 else 'NO'  # Flag if difference > 1%
        }

    def analyze_equal_debit_credit(transactions_df):
        """Analyze equal debit and credit transactions"""
        # Look for transactions with same amount on same day
        equal_transactions = 0

        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        
        for _, credit_row in df_credits.iterrows():
            credit_amount = pd.to_numeric(str(credit_row['Amount']).replace(',', ''), errors='coerce')
            credit_date = credit_row.get('Date', '')
            
            if pd.notna(credit_amount) and credit_date:
                # Look for matching debit amount on same date
                matching_debits = df_debits[
                    (pd.to_numeric(df_debits['Amount'].astype(str).str.replace(',', ''), errors='coerce') == credit_amount) &
                    (df_debits['Date'] == credit_date)
                ]
                equal_transactions += len(matching_debits)
        
        return {
            'Equal_Transactions': equal_transactions,
            'Status': 'YES' if equal_transactions > 0 else 'NO'
        }

    def analyze_suspected_income_infusion(transactions_df):
        """Analyze suspected income infusion patterns"""
        # Look for unusually large credit amounts
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()

        credit_amounts = pd.to_numeric(df_credits['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
        
        if len(credit_amounts) > 0:
            # Define threshold as 95th percentile
            threshold = credit_amounts.quantile(0.95)
            large_credits = len(credit_amounts[credit_amounts > threshold])
            
            return {
                'Large_Credits': large_credits,
                'Threshold': threshold,
                'Status': 'YES' if large_credits > 5 else 'NO'
            }
        
        return {'Large_Credits': 0, 'Threshold': 0, 'Status': 'NO'}

    def analyze_negative_eod_balance(transactions_df):
        """Analyze negative end-of-day balances"""
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        if 'Balance' in df_credits.columns:
            credit_balances = pd.to_numeric(df_credits['Balance'].astype(str).str.replace(',', ''), errors='coerce')
            credit_negative = len(credit_balances[credit_balances < 0])
        else:
            credit_negative = 0
            
        if 'Balance' in df_debits.columns:
            debit_balances = pd.to_numeric(df_debits['Balance'].astype(str).str.replace(',', ''), errors='coerce')
            debit_negative = len(debit_balances[debit_balances < 0])
        else:
            debit_negative = 0
        
        return {
            'Credit_Negative': credit_negative,
            'Debit_Negative': debit_negative,
            'Total_Negative': credit_negative + debit_negative,
            'Status': 'YES' if (credit_negative + debit_negative) > 0 else 'NO'
        }

    def analyze_bank_holidays_transactions(transactions_df):
        """Analyze transactions on bank holidays"""
        # Get Indian bank holidays for 2024
        india_holidays = holidays.India(years=2024)
        
        holiday_transactions = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        
        for df in [df_credits, df_debits]:
            for _, row in df.iterrows():
                date_str = row.get('Date', '')
                if date_str:
                    try:
                        # Parse date
                        if '/' in date_str:
                            date_obj = datetime.strptime(date_str, '%d/%m/%Y').date()
                        else:
                            date_obj = pd.to_datetime(date_str).date()
                        
                        if date_obj in india_holidays:
                            holiday_transactions += 1
                    except:
                        continue
        
        return {
            'Holiday_Transactions': holiday_transactions,
            'Status': 'YES' if holiday_transactions > 0 else 'NO'
        }

    def analyze_suspicious_rtgs(transactions_df):
        """Analyze suspicious RTGS transactions"""
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        rtgs_credits = len(df_credits[df_credits['Description'].str.contains('RTGS', case=False, na=False)])
        rtgs_debits = len(df_debits[df_debits['Description'].str.contains('RTGS', case=False, na=False)])
        
        # Look for RTGS transactions with unusual amounts or patterns
        suspicious_rtgs = 0
        
        for df in [df_credits, df_debits]:
            rtgs_transactions = df[df['Description'].str.contains('RTGS', case=False, na=False)]
            amounts = pd.to_numeric(rtgs_transactions['Amount'].astype(str).str.replace(',', ''), errors='coerce')
            
            # Flag RTGS transactions below 2 lakhs (RTGS minimum)
            low_rtgs = len(amounts[amounts < 200000])
            suspicious_rtgs += low_rtgs
        
        return {
            'RTGS_Credits': rtgs_credits,
            'RTGS_Debits': rtgs_debits,
            'Suspicious_Low_RTGS': suspicious_rtgs,
            'Status': 'YES' if suspicious_rtgs > 0 else 'NO'
        }

    def analyze_suspicious_tax_payments(transactions_df):
        """Analyze suspicious tax payments (SGST/CGST patterns)"""
        tax_patterns = ['SGST', 'CGST', 'IGST', 'GST']
        suspicious_tax = 0
        suspicious_txns = []

        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()
        for df in [df_credits, df_debits]:
            for pattern in tax_patterns:
                cols_to_keep = ['Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']

                tax_transactions = df.loc[df['Description'].str.contains(pattern, case=False, na=False), cols_to_keep].copy()
                if not tax_transactions.empty:
                    tax_transactions['Indicator'] = 'Suspicious Transactions'

                    suspicious_txns.append(tax_transactions)
                
                # Group by date and amount to find potential duplicates
                if len(tax_transactions) > 0:
                    grouped = tax_transactions.groupby(['Date', 'Amount']).size()
                    duplicate_tax = len(grouped[grouped > 1])
                    suspicious_tax += duplicate_tax
        suspicious_txns_df = pd.concat(suspicious_txns, ignore_index=True) if suspicious_txns else pd.DataFrame()
        return {
            'Suspicious_Tax_Payments': suspicious_tax,
            'Status': 'YES' if suspicious_tax > 0 else 'NO',
            'Transactions': suspicious_txns_df
        }

    def analyze_irregular_credit_card_payments(transactions_df):
        """Analyze irregular credit card payment patterns"""
        cc_keywords = ['CREDIT CARD', 'CC PAYMENT', 'CARD PAYMENT', 'VISA', 'MASTERCARD']
        cc_transactions = 0
        irregular_patterns = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        
        for df in [df_credits, df_debits]:
            for keyword in cc_keywords:
                cc_txns = df[df['Description'].str.contains(keyword, case=False, na=False)]
                cc_transactions += len(cc_txns)
                
                # Look for irregular amounts (non-round numbers for CC payments)
                amounts = pd.to_numeric(cc_txns['Amount'].astype(str).str.replace(',', ''), errors='coerce')
                non_round = len(amounts[amounts % 100 != 0])  # Not multiples of 100
                irregular_patterns += non_round
        
        return {
            'CC_Transactions': cc_transactions,
            'Irregular_CC_Patterns': irregular_patterns,
            'Status': 'YES' if irregular_patterns > 10 else 'NO'
        }

    def analyze_irregular_salary_credits(transactions_df):
        """Analyze irregular salary credit patterns"""
        salary_keywords = ['SALARY', 'SAL', 'PAYROLL', 'WAGES']
        salary_transactions = 0
        irregular_salary = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()

        for keyword in salary_keywords:
            salary_txns = df_credits[df_credits['Description'].str.contains(keyword, case=False, na=False)]
            salary_transactions += len(salary_txns)
            
            # Look for salary credits on non-month-end dates
            if len(salary_txns) > 0:
                for _, row in salary_txns.iterrows():
                    date_str = row.get('Date', '')
                    if date_str:
                        try:
                            if '/' in date_str:
                                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                            else:
                                date_obj = pd.to_datetime(date_str)
                            
                            # Check if not on month-end (last 3 days of month)
                            if date_obj.day < 28:
                                irregular_salary += 1
                        except:
                            continue
        
        return {
            'Salary_Transactions': salary_transactions,
            'Irregular_Salary_Dates': irregular_salary,
            'Status': 'YES' if irregular_salary > 0 else 'NO'
        }

    def analyze_unchanged_salary_credit_amount(transactions_df):
        """Analyze unchanged salary credit amounts"""
        salary_keywords = ['SALARY', 'SAL', 'PAYROLL']
        salary_amounts = []
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()

        for keyword in salary_keywords:
            salary_txns = df_credits[df_credits['Description'].str.contains(keyword, case=False, na=False)]
            amounts = pd.to_numeric(salary_txns['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
            salary_amounts.extend(amounts.tolist())
        
        if len(salary_amounts) > 1:
            # Check for unchanged amounts
            unique_amounts = len(set(salary_amounts))
            unchanged_percentage = (len(salary_amounts) - unique_amounts) / len(salary_amounts) * 100
            
            return {
                'Total_Salary_Transactions': len(salary_amounts),
                'Unique_Amounts': unique_amounts,
                'Unchanged_Percentage': unchanged_percentage,
                'Status': 'YES' if unchanged_percentage > 80 else 'NO'
            }
        
        return {'Total_Salary_Transactions': 0, 'Unique_Amounts': 0, 'Unchanged_Percentage': 0, 'Status': 'NO'}

    def analyze_irregular_transfers_to_parties(transactions_df):
        """Analyze irregular transfers to external parties"""
        # Look for transfers to external parties with unusual patterns
        party_keywords = ['TRANSFER', 'NEFT', 'IMPS', 'TO']
        suspicious_transfers = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()
        
        for df in [df_credits, df_debits]:
            for keyword in party_keywords:
                transfers = df[df['Description'].str.contains(keyword, case=False, na=False)]
                
                # Look for transfers with unusual amounts or timing
                amounts = pd.to_numeric(transfers['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
                
                # Flag transfers above certain threshold
                high_transfers = len(amounts[amounts > 100000])  # Above 1 lakh
                suspicious_transfers += high_transfers
        
        return {
            'Suspicious_Transfers': suspicious_transfers,
            'Status': 'YES' if suspicious_transfers > 20 else 'NO'
        }

    def analyze_irregular_interest_charges(transactions_df):
        """Analyze irregular interest charges"""
        interest_keywords = ['INTEREST', 'INT', 'CHARGES', 'PENALTY']
        interest_transactions = 0
        irregular_interest = 0
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()
        for keyword in interest_keywords:
            interest_txns = df_debits[df_debits['Description'].str.contains(keyword, case=False, na=False)]
            interest_transactions += len(interest_txns)
            
            # Look for unusually high interest charges
            amounts = pd.to_numeric(interest_txns['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
            high_interest = len(amounts[amounts > 10000])  # Above 10k
            irregular_interest += high_interest
        
        return {
            'Interest_Transactions': interest_transactions,
            'High_Interest_Charges': irregular_interest,
            'Status': 'YES' if irregular_interest > 5 else 'NO'
        }

    def analyze_decimals_in_atm_withdrawal(transactions_df):
        """Analyze decimals in ATM withdrawal amounts"""
        atm_keywords = ['ATM', 'CASH WITHDRAWAL', 'WITHDRAWAL']
        atm_transactions = 0
        decimal_atm = 0
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()
        for keyword in atm_keywords:
            atm_txns = df_debits[df_debits['Description'].str.contains(keyword, case=False, na=False)]
            atm_transactions += len(atm_txns)
            
            # Look for ATM withdrawals with decimal amounts (unusual)
            amounts = pd.to_numeric(atm_txns['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
            decimal_amounts = len(amounts[amounts % 1 != 0])  # Not whole numbers
            decimal_atm += decimal_amounts
        
        return {
            'ATM_Transactions': atm_transactions,
            'Decimal_ATM_Amounts': decimal_atm,
            'Status': 'YES' if decimal_atm > 0 else 'NO'
        }

    def analyze_high_withdrawal(transactions_df):
        """Analyze high withdrawal amounts"""
        withdrawal_keywords = ['WITHDRAWAL', 'ATM', 'CASH']
        high_withdrawals = 0

        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        for keyword in withdrawal_keywords:
            withdrawals = df_debits[df_debits['Description'].str.contains(keyword, case=False, na=False)]
            amounts = pd.to_numeric(withdrawals['Amount'].astype(str).str.replace(',', ''), errors='coerce').dropna()
            
            # Flag withdrawals above 50k (high amount)
            high_amounts = len(amounts[amounts > 50000])
            high_withdrawals += high_amounts
        
        return {
            'High_Withdrawals': high_withdrawals,
            'Status': 'YES' if high_withdrawals > 10 else 'NO'
        }

    def analyze_nonexistent_date(transactions_df):
        """Analyze transactions with non-existent dates"""
        invalid_dates = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        for df in [df_credits, df_debits]:
            for _, row in df.iterrows():
                date_str = row.get('Date', '')
                if date_str:
                    try:
                        if '/' in date_str:
                            datetime.strptime(date_str, '%d/%m/%Y')
                        else:
                            pd.to_datetime(date_str)
                    except:
                        invalid_dates += 1
        
        return {
            'Invalid_Dates': invalid_dates,
            'Status': 'YES' if invalid_dates > 0 else 'NO'
        }

    def analyze_duplicate_reference_number(transactions_df):
        """Analyze duplicate reference numbers"""
        all_refs = []

        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        
        for df in [df_credits, df_debits]:
            # Look for reference numbers in descriptions
            for _, row in df.iterrows():
                desc = str(row.get('Description', ''))
                # Extract potential reference numbers (alphanumeric patterns)
                import re
                refs = re.findall(r'[A-Z0-9]{8,}', desc)
                all_refs.extend(refs)
        
        # Count duplicates
        ref_counts = pd.Series(all_refs).value_counts()
        duplicate_refs = len(ref_counts[ref_counts > 1])
        
        return {
            'Total_References': len(all_refs),
            'Duplicate_References': duplicate_refs,
            'Status': 'YES' if duplicate_refs > 5 else 'NO'
        }

    def analyze_font_mismatch(transactions_df):
        """Analyze font mismatch in descriptions (placeholder - would need OCR analysis)"""
        # This would typically require OCR analysis of scanned documents
        # For now, we'll look for unusual character patterns
        
        font_issues = 0
        df_credits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'CREDIT'].copy()
        df_debits = transactions_df[transactions_df['Credit/Debit'].str.upper() == 'DEBIT'].copy()

        
        for df in [df_credits, df_debits]:
            for _, row in df.iterrows():
                desc = str(row.get('Description', ''))
                
                # Look for mixed character sets (basic check)
                has_ascii = any(ord(c) < 128 for c in desc)
                has_non_ascii = any(ord(c) >= 128 for c in desc)
                
                if has_ascii and has_non_ascii:
                    # Count unusual character combinations
                    if len(set(desc)) / len(desc) < 0.3:  # Low character diversity
                        font_issues += 1
        
        return {
            'Font_Mismatch_Issues': font_issues,
            'Status': 'YES' if font_issues > 10 else 'NO'
        }

    def generate_comprehensive_fraud_summary(transactions_df):
        """Generate comprehensive fraud detection summary with all indicators"""
        print("Loading data files...")

        print("Analyzing comprehensive fraud patterns...")
        
        # Perform all analyses
        duplicity_analysis = analyze_data_duplicity()
        balance_analysis = analyze_balance_reconciliation(transactions_df)
        equal_debit_credit = analyze_equal_debit_credit(transactions_df)
        income_infusion = analyze_suspected_income_infusion(transactions_df)
        negative_balance = analyze_negative_eod_balance(transactions_df)
        bank_holidays = analyze_bank_holidays_transactions(transactions_df)
        suspicious_rtgs = analyze_suspicious_rtgs(transactions_df)
        tax_payments = analyze_suspicious_tax_payments(transactions_df)
        cc_payments = analyze_irregular_credit_card_payments(transactions_df)
        salary_credits = analyze_irregular_salary_credits(transactions_df)
        unchanged_salary = analyze_unchanged_salary_credit_amount(transactions_df)
        transfers_parties = analyze_irregular_transfers_to_parties(transactions_df)
        interest_charges = analyze_irregular_interest_charges(transactions_df)
        atm_decimals = analyze_decimals_in_atm_withdrawal(transactions_df)
        high_withdrawal = analyze_high_withdrawal(transactions_df)
        nonexistent_date = analyze_nonexistent_date(transactions_df)
        duplicate_refs = analyze_duplicate_reference_number(transactions_df)
        font_mismatch = analyze_font_mismatch(transactions_df)
        
        # Create comprehensive summary dataframe
        summary_data = [
            {
                'Fraud_Indicator': 'Balance Reconciliation',
                'Identified': balance_analysis['Status'],
                'Description': f"Credit total: ₹{balance_analysis['Credit_Total']:,.2f}, Debit total: ₹{balance_analysis['Debit_Total']:,.2f}. Difference: ₹{balance_analysis['Difference']:,.2f} ({balance_analysis['Difference_Percentage']:.2f}%)"
            },
            {
                'Fraud_Indicator': 'Equal Debit Credit',
                'Identified': equal_debit_credit['Status'],
                'Description': f"Found {equal_debit_credit['Equal_Transactions']} transactions with equal debit and credit amounts on same date"
            },
            {
                'Fraud_Indicator': 'Suspected Income Infusion',
                'Identified': income_infusion['Status'],
                'Description': f"Found {income_infusion['Large_Credits']} large credit transactions above ₹{income_infusion['Threshold']:,.2f}"
            },
            {
                'Fraud_Indicator': 'Negative EOD Balance',
                'Identified': negative_balance['Status'],
                'Description': f"Found {negative_balance['Total_Negative']} transactions with negative balances. Credit: {negative_balance['Credit_Negative']}, Debit: {negative_balance['Debit_Negative']}"
            },
            {
                'Fraud_Indicator': 'Transactions on Bank Holidays',
                'Identified': bank_holidays['Status'],
                'Description': f"Found {bank_holidays['Holiday_Transactions']} transactions on bank holidays"
            },
            {
                'Fraud_Indicator': 'Suspicious RTGS Transactions',
                'Identified': suspicious_rtgs['Status'],
                'Description': f"Found {suspicious_rtgs['Suspicious_Low_RTGS']} suspicious RTGS transactions. Total RTGS: Credit {suspicious_rtgs['RTGS_Credits']}, Debit {suspicious_rtgs['RTGS_Debits']}"
            },
            {
                'Fraud_Indicator': 'Suspicious Tax Payments',
                'Identified': tax_payments['Status'],
                'Description': f"Found {tax_payments['Suspicious_Tax_Payments']} suspicious tax payment patterns",
            },
            {
                'Fraud_Indicator': 'Irregular Credit Card Payments',
                'Identified': cc_payments['Status'],
                'Description': f"Found {cc_payments['Irregular_CC_Patterns']} irregular credit card payment patterns out of {cc_payments['CC_Transactions']} CC transactions"
            },
            {
                'Fraud_Indicator': 'Irregular Salary Credits',
                'Identified': salary_credits['Status'],
                'Description': f"Found {salary_credits['Irregular_Salary_Dates']} salary credits on irregular dates out of {salary_credits['Salary_Transactions']} salary transactions"
            },
            {
                'Fraud_Indicator': 'Unchanged Salary Credit Amount',
                'Identified': unchanged_salary['Status'],
                'Description': f"Found {unchanged_salary['Unchanged_Percentage']:.1f}% unchanged salary amounts out of {unchanged_salary['Total_Salary_Transactions']} salary transactions"
            },
            {
                'Fraud_Indicator': 'Irregular Transfers to Parties',
                'Identified': transfers_parties['Status'],
                'Description': f"Found {transfers_parties['Suspicious_Transfers']} suspicious high-value transfers to external parties"
            },
            {
                'Fraud_Indicator': 'Data Duplicity',
                'Identified': duplicity_analysis['Status'],
                'Description': f"Found {duplicity_analysis['Total_Duplicate_Transactions']} duplicate transactions ({duplicity_analysis['Duplicate_Percentage']:.1f}% of total). Credit: {duplicity_analysis['Credit_Duplicates']}, Debit: {duplicity_analysis['Debit_Duplicates']}"
            },
            {
                'Fraud_Indicator': 'Irregular Interest Charges',
                'Identified': interest_charges['Status'],
                'Description': f"Found {interest_charges['High_Interest_Charges']} high interest charges out of {interest_charges['Interest_Transactions']} interest transactions"
            },
            {
                'Fraud_Indicator': 'Decimals in ATM Withdrawal',
                'Identified': atm_decimals['Status'],
                'Description': f"Found {atm_decimals['Decimal_ATM_Amounts']} ATM withdrawals with decimal amounts out of {atm_decimals['ATM_Transactions']} ATM transactions"
            },
            {
                'Fraud_Indicator': 'High Withdrawal',
                'Identified': high_withdrawal['Status'],
                'Description': f"Found {high_withdrawal['High_Withdrawals']} high-value withdrawals above ₹50,000"
            },
            {
                'Fraud_Indicator': 'Non Existent Date',
                'Identified': nonexistent_date['Status'],
                'Description': f"Found {nonexistent_date['Invalid_Dates']} transactions with invalid/non-existent dates"
            },
            {
                'Fraud_Indicator': 'Duplicate Reference Number',
                'Identified': duplicate_refs['Status'],
                'Description': f"Found {duplicate_refs['Duplicate_References']} duplicate reference numbers out of {duplicate_refs['Total_References']} total references"
            },
            {
                'Fraud_Indicator': 'Font Mismatch',
                'Identified': font_mismatch['Status'],
                'Description': f"Found {font_mismatch['Font_Mismatch_Issues']} potential font mismatch issues in transaction descriptions"
            }
        ]
        
        # Create summary dataframe
        
        summary_df = pd.DataFrame(summary_data)
        

        tax_txns_df = tax_payments['Transactions']
        if not tax_txns_df.empty:
            # Add a header row as separator
                header_row = pd.DataFrame([{
                    'Indicator':'','Date': '', 'Credit/Debit': '', 'Description': '', 'Amount': '', 'Balance': ''
                }])

                # reorder cols to match
                cols = ['Indicator', 'Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']
                tax_txns_df = tax_txns_df[cols]

                details_df = pd.concat([header_row, tax_txns_df], ignore_index=True)
        else:
            details_df = pd.DataFrame()
        
        print(f"\n=== COMPREHENSIVE FRAUD DETECTION SUMMARY ===")
       
        return (summary_df, details_df)

    fraud_summary_df= generate_comprehensive_fraud_summary(transactions_df)

    return fraud_summary_df

def generate_xns_sheet(transactions_df):

    def detect_mode(description):
        desc = str(description).upper()
        if "NEFT" in desc:
            return "NEFT"
        elif "RTGS" in desc:
            return "RTGS"
        elif "UPI" in desc:
            return "UPI"
        elif "DD" in desc or "DEMAND DRAFT" in desc:
            return "Demand Draft"
        elif "CHQ" in desc or "CHEQUE" in desc:
            return "Cheque"
        elif "IMPS" in desc:
            return "IMPS"
        elif "INF" in desc or "INTERNET BANKING" in desc:
            return "Internet Banking"
        elif "ACH" in desc:
            return "ACH"
        elif "CASH" in desc:
            return "CASH"
        else:
            return "Other"

    
    transactions_df['Mode'] = transactions_df['Description'].apply(detect_mode)

    cols = transactions_df.columns.tolist()
    if 'Balance' in cols:
        balance_idx = cols.index('Balance')
        # Remove mode from current position
        cols.remove('Mode')
        # Insert mode right after balance
        cols.insert(balance_idx + 1, 'Mode')

    # Reorder the DataFrame
    transactions_df = transactions_df[cols]
    return transactions_df




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

    print("[INFO] Generating Return Transaction Details...")
    return_txn_df = generate_return_txn(transactions_df.copy())

    
    print("[INFO] Generating Transaction Details...")
    xns_txn_df = generate_xns_sheet(transactions_df.copy())

    print("[INFO] Generating Recurring Credit Details...")
    recurring_credit_df = generate_recurring_credit_debit(transactions_df.copy(), val='CREDIT')

    print("[INFO] Generating Recurring Debit Details...")
    recurring_debit_df = generate_recurring_credit_debit(transactions_df.copy(), val='DEBIT')
    
    print("[INFO] Generating Fraud Sheet Details...")
    # if not recurring_credit_df.empty or not recurring_debit_df.empty:
    duplicates_df = generate_duplicates(recurring_credit_df , recurring_debit_df)
    fraud_sheet_df,sus_txns_df = generate_fraud_sheet(transactions_df.copy(), duplicates_df)
    # else: 
    #     fraud_sheet_df,sus_txns_df = generate_fraud_sheet(transactions_df.copy())

    
    
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
        if not return_txn_df.empty:
            return_txn_df.to_excel(writer, index=False, sheet_name='Return Txn')
        if not duplicates_df.empty:
            duplicates_df.to_excel(writer ,index=False , sheet_name='Duplicates_Fraud_Sheet')      
        if not fraud_sheet_df.empty:
            fraud_sheet_df.to_excel(writer, index=False, sheet_name='Fraud Check Sheet', startrow=0)
            sus_txns_df.to_excel(writer , index=False ,sheet_name='Fraud Check Sheet',startrow=len(fraud_sheet_df)+3 )   
            if not duplicates_df.empty:
               duplicates_df.to_excel(writer ,index=False , sheet_name='Fraud Check Sheet', startrow=len(fraud_sheet_df+sus_txns_df))
        if not xns_txn_df.empty:
            xns_txn_df.to_excel(writer, index=False, sheet_name='Xns')

    
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
