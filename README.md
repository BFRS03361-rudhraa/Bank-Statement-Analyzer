# Bank Statement Parser & Analyzer

A Python-based solution for parsing and structuring bank statements from multiple banks into a standardized format for analysis. This project is to handle diverse statement formats and improve extraction accuracy and analysis.

---

## Features

- **Multi-bank support:** Works with different bank statement formats like HDFC , Axis , Federal , Canara , ICICI etc..
- **Structured extraction:** Separates **metadata** (account details, statement period, balances) and **transactions** (date, description, credit , debit, amount, balance) into structured Excel files.
- Across a total of 21 different bank statements, the current parsing script that uses Gemini 2.O Flash successfully produces correctly formatted raw text for 18 banks.
- **Excel output:** Saves parsed metadata and transaction tables in separate sheets for carrying out the next analysis.
- **OCR pipeline:** Uses Gemini 2.0 Flash OCR for text extraction and can be adapted for other OCR tools.
- Currently working on the calculation part of the produced raw text to replicate Auth Bridge results.

---

