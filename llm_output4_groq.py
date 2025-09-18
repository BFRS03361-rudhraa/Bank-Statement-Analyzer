import argparse
import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from groq import Groq
import requests
import re


def call_groq(prompt: str, model: str, timeout: int, max_tokens: int, json_mode: bool, verbose: bool = False) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Missing GROQ_API_KEY in environment")
    client = Groq(api_key=api_key, timeout=timeout)
    if verbose:
        print(f"[groq] chat.completions.create model={model} max_tokens={max_tokens} json_mode={json_mode}")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature":0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


# --------- Inlined helpers from llm_output2 ---------
TEMPLATE_FROM_EXCEL = """
You are a bank statement metadata extractor.
You are given two plain-text inputs:
1. "first_page_full_text": the parsed raw text of the first page from the bank statement (definitely contains metadata).
2. "page1_table1": the parsed text of the first pages table (may contain both metadata rows and transactions).

Output a STRICT JSON object in this structure:
{{
  "metadata": {{
    ... all customer/account metadata found in first_page_full_text and page1_table1 BEFORE the first transaction table ...
  }},
  "table_start_index": <integer row number in p1_t1 where the first transaction table begins>,
  "table_header": "<the exact header line of the transaction table from the first_page_full_text>"
}}

CRITICAL RULES:
- "metadata" should ONLY contain account/customer information like account number, name, address, statement period, etc.
- "metadata" should NEVER contain transaction rows, table headers, or any financial data.
- "table_start_index" should point to the row in p1_t1 that contains the actual table header.
- Use "text_sheet" ONLY for extracting the headers.

Table Detection Rules:
- Look for a row in p1_t1 that contains typical transaction table headers like: Date, Description, Narration, Particulars, Details, Debit, Dr, Credit, Cr, Balance, etc.
- The table header row should be followed by actual transaction data (rows with dates, amounts, descriptions).
- If row 0 in p1_t1 looks like a transaction header, then table_start_index = 0.
- Otherwise, find the first row that contains header-like text (not transaction data).

Validation:
- The table_header you extract should match the structure of the transaction rows that follow.
- If you're unsure, prefer to extract less metadata rather than including transaction data in metadata.

--- INPUT CONTENT ---
[first_page_full_text]
{text_sheet}

[page1_table1]
{p1_t1_sheet}

[transaction_sample]
{p1_sample_sheet}
"""


def read_excel_parts(xlsx_path: str) -> dict:
    xls = pd.ExcelFile(xlsx_path)
    sheet_map = {s.lower(): s for s in xls.sheet_names}

    if "Text" not in xls.sheet_names:
        raise SystemExit("Missing required sheet: text")
    text_df = pd.read_excel(xls, sheet_map.get("text"), dtype=str, header=None)
    text_df = text_df.iloc[:, :2]
    text_lines = [str(v).strip() for v in text_df.iloc[1, :].dropna()]
    text_sheet = "\n".join(text_lines)

    if "p1_t1" not in xls.sheet_names:
        raise SystemExit("Missing required sheet: p1_t1")
    p1_df = pd.read_excel(xls, sheet_map.get("p1_t1"), dtype=str, header=None)
    p1_lines: List[str] = []
    for i, row in p1_df.iterrows():
        vals = [(str(v) if pd.notna(v) else "").strip() for v in row.tolist()]
        while vals and vals[-1] == "":
            vals.pop()
        if vals:
            p1_lines.append(f"{i}: " + "\t".join(vals))
    p1_t1_sheet = "\n".join(p1_lines)

    return {"text_sheet": text_sheet, "p1_t1_sheet": p1_t1_sheet, "xls": xls, "sheet_map": sheet_map}


def parse_json_strict_or_loose(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


# --------- Inlined helpers from header_verifier ---------
def read_excel_three(xlsx_path: str) -> Dict[str, str]:
    xls = pd.ExcelFile(xlsx_path)
    sheet_map = {s.lower(): s for s in xls.sheet_names}

    if "Text" not in xls.sheet_names:
        raise SystemExit("Missing required sheet: Text")
    text_df = pd.read_excel(xls, sheet_map.get("text"), dtype=str, header=None)
    text_df = text_df.iloc[:, :2]
    text_lines = [str(v).strip() for v in text_df.iloc[1, :].dropna()]
    text_sheet = "\n".join(text_lines)

    if "p1_t1" not in xls.sheet_names:
        raise SystemExit("Missing required sheet: p1_t1")
    p1_df = pd.read_excel(xls, sheet_map.get("p1_t1"), dtype=str, header=None)
    p1_lines: List[str] = []
    for i, row in p1_df.iterrows():
        vals = [(str(v) if pd.notna(v) else "").strip() for v in row.tolist()]
        while vals and vals[-1] == "":
            vals.pop()
        if vals:
            p1_lines.append(f"{i}: " + "\t".join(vals))
    p1_t1_sheet = "\n".join(p1_lines)

    p2_t1_sheet = ""
    if "p2_t1" in xls.sheet_names:
        p2_df = pd.read_excel(xls, sheet_map.get("p2_t1"), dtype=str, header=None)
        p2_lines: List[str] = []
        for i, row in p2_df.iterrows():
            vals = [(str(v) if pd.notna(v) else "").strip() for v in row.tolist()]
            while vals and vals[-1] == "":
                vals.pop()
            if vals:
                p2_lines.append(f"{i}: " + "\t".join(vals))
        p2_t1_sheet = "\n".join(p2_lines)

    return {"text_sheet": text_sheet, "p1_t1_sheet": p1_t1_sheet, "p2_t1_sheet": p2_t1_sheet}


TEMPLATE_VERIFY = """
You are a meticulous bank statement table header verifier.
Inputs:
- text_sheet: first page full text (metadata area; may contain a header-like line)
- p1_t1_sheet: page 1 table rows (each line prefixed with row index), likely contains header + transactions
- p2_t1_sheet: page 2 table rows (optional), likely contains header + transactions

Task:
1) Extract at most one best header candidate from each source: text_sheet, p1_t1_sheet, p2_t1_sheet.
   - Normalize header into pipe-separated form: "Col1 | Col2 | Col3 | ...".
   - Also return header_columns as an array.
   - Apply normalization merges like: Branch Code, Value Date, Ref No, Chq No, Debit Amount, Credit Amount, Balance Amount.
2) From p1_t1_sheet and p2_t1_sheet, identify 3-8 early valid transaction rows after the header.
   - A valid row contains plausible date(s), description/narration text, debit/credit markers (DR/CR), amounts, and running balance.
   - Return them as arrays of tokens (split by tabs) with their row indices.
3) Compare the extracted candidates against the transaction rows.
   - Choose a final header that contains proper pipe seperated form and aligns completely with the transaction rows.
   - If the extracted candidates matches with the transaction rows correctly in all columns then choose that.
   - Provide header_valid (true/false) and a short reason.

Output STRICT JSON with exactly these keys, everything should be present:
{{
  "initial_header": "...",
  "candidates": {{
    "text": {{"header": "...", "header_columns": [..]}} | null,
    "p1_t1": {{"header": "...", "header_columns": [..]}} | null,
    "p2_t1": {{"header": "...", "header_columns": [..]}}| null
  }},
  "transactions": {{
    "p1_t1_sample": [{{"row_index": 0, "cells": ["..."]}}],
    "p2_t1_sample": [{{"row_index": 0, "cells": ["..."]}}]
  }},
  "final": {{
    "table_header": "Col1 | Col2 | ...",
    "header_columns": ["Col1", "Col2", ...],
    "header_valid": true,
    "header_reason": "...",
    "source": "initial|text|p1_t1|p2_t1"
  }}
}}

Do not output anything except valid JSON.

--- INPUTS ---
initial_header:
{initial_header}

[text_sheet]
{text_sheet}

[p1_t1_sheet]
{p1_t1_sheet}

[p2_t1_sheet]
{p2_t1_sheet}
"""


def build_transactions_simple(xls, data, header_line: str, start_idx: int) -> pd.DataFrame:
    # Prefer pipe separator else split by tabs or 3+ spaces
    header_cols = [c.strip() for c in header_line.split("|")] if "|" in header_line else __import__("re").split(r"\t+|\s{2,}", header_line)
    frames = []
    for sheet in xls.sheet_names:
        if not (sheet.lower().startswith("p") and "_t" in sheet.lower()):
            continue
        df = pd.read_excel(xls, sheet, header=None, dtype=str).dropna(how="all")
        if sheet.lower().startswith("p1_t1") and isinstance(start_idx, int):
            df = df.iloc[start_idx:, :]
        if df.empty:
            continue
        mask = df.apply(lambda row: list(row.dropna()) == header_cols[:len(row.dropna())], axis=1)
        df = df[~mask]
        col_count = df.shape[1]
        cols = header_cols[:col_count] if len(header_cols) >= col_count else header_cols + [f"Unnamed_{i}" for i in range(col_count - len(header_cols))]
        df.columns = cols
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=header_cols)


def main():
    parser = argparse.ArgumentParser(description="Extract with Groq, then verify header with Groq, and output Excel")
    parser.add_argument("--input-xlsx", required=True)
    parser.add_argument("--groq-model", default="qwen/qwen3-32b")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--text-lines", type=int, default=120)
    parser.add_argument("--p1-lines", type=int, default=120)
    parser.add_argument("--p2-lines", type=int, default=120)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--excel-out", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1) Initial extraction using llm_output2 flow
    print("[step] Reading Excel (text, p1_t1) ...")
    parts = read_excel_parts(args.input_xlsx)
    # Trim long inputs to avoid token overflow
    text_trim = "\n".join(parts["text_sheet"].splitlines()[: args.text_lines])
    p1_trim = "\n".join(parts["p1_t1_sheet"].splitlines()[: args.p1_lines])
    prompt = TEMPLATE_FROM_EXCEL.format(
        text_sheet=text_trim,
        p1_t1_sheet=p1_trim,
        p1_sample_sheet=p1_trim,
    )
    print(f"[step] Groq initial extract model={args.groq_model} ...")
    try:
        raw = call_groq(prompt, args.groq_model, args.timeout, args.max_tokens, json_mode=True, verbose=args.verbose)
    except Exception as e:
        # Repair pass if JSON mode caused failure
        print(f"[extract] JSON mode failed ({e}); retrying without json_mode ...")
        repair_prompt = "Return ONLY strict JSON with keys metadata, table_start_index, table_header. No prose.\n\n" + prompt
        raw = call_groq(repair_prompt, args.groq_model, args.timeout, max(args.max_tokens, 6000), json_mode=False, verbose=args.verbose)
    try:
        data = parse_json_strict_or_loose(raw)
    except Exception:
        repair = "Return ONLY strict JSON with keys metadata, table_start_index, table_header (no prose).\n\n" + raw
        raw2 = call_groq(repair, args.groq_model, args.timeout, max(args.max_tokens, 4000), json_mode=True, verbose=args.verbose)
        data = parse_json_strict_or_loose(raw2)

    initial_header = data.get("table_header", "")
    start_idx = data.get("table_start_index")

    # 2) Header verification using header_verifier logic with Groq
    print("[step] Reading Excel (text, p1_t1, p2_t1) for verification ...")
    parts3 = read_excel_three(args.input_xlsx)
    # Trim for verifier
    v_text = "\n".join(parts3["text_sheet"].splitlines()[: args.text_lines])
    v_p1 = "\n".join(parts3["p1_t1_sheet"].splitlines()[: args.p1_lines])
    v_p2 = "\n".join(parts3["p2_t1_sheet"].splitlines()[: args.p2_lines])
    verify_prompt = TEMPLATE_VERIFY.format(
        initial_header=initial_header,
        text_sheet=v_text,
        p1_t1_sheet=v_p1,
        p2_t1_sheet=v_p2,
    )
    print("[step] Groq verifying/correcting header ...")
    try:
        vraw = call_groq(verify_prompt, args.groq_model, args.timeout, args.max_tokens, json_mode=True, verbose=args.verbose)
    except Exception as e:
        print(f"[verify] JSON mode failed ({e}); retrying without json_mode ...")
        v_repair = "Return ONLY strict JSON with all required keys including final. No prose.\n\n" + verify_prompt
        vraw = call_groq(v_repair, args.groq_model, args.timeout, args.max_tokens, json_mode=False, verbose=args.verbose)
    verification_report: Dict[str, Any] = {}
    print(f"[verify] initial_header: {initial_header}")
    try:
        verification_report = json.loads(vraw)
        # Print candidates if present
        candidates = verification_report.get("candidates", {}) or {}
        cand_text = candidates.get("text")
        cand_p1 = candidates.get("p1_t1")
        cand_p2 = candidates.get("p2_t1")
        if cand_text:
            print(f"[verify] candidate from text: {cand_text.get('header')}")
        if cand_p1:
            print(f"[verify] candidate from p1_t1: {cand_p1.get('header')}")
        if cand_p2:
            print(f"[verify] candidate from p2_t1: {cand_p2.get('header')}")

        # Print transactions sample summary
        tx = verification_report.get("transactions", {}) or {}
        p1s = tx.get("p1_t1_sample", []) or []
        p2s = tx.get("p2_t1_sample", []) or []
        if p1s:
            print(f"[verify] p1_t1_sample rows: {len(p1s)} (e.g., row {p1s[0].get('row_index')})")
        if p2s:
            print(f"[verify] p2_t1_sample rows: {len(p2s)} (e.g., row {p2s[0].get('row_index')})")

        # Final decision
        final_block = verification_report.get("final", {}) or {}
        final_header = cand_text.get('header')
        final_cols = cand_text.get('header_columns')
        print(final_header, final_cols)
        if final_cols and "|" not in final_header:
            final_header = " | ".join(final_cols)
        print(f"[verify] final header: {final_header}")
        if "header_valid" in final_block:
            print(f"[verify] header_valid: {final_block.get('header_valid')} source: {final_block.get('source')}\n[verify] reason: {final_block.get('header_reason')}")
    except Exception:
        print("[verify] non-JSON response from verifier; using initial header")
        final_header = initial_header

    # Fallback: if no final in report, auto-pick best candidate by column count vs sample
    # def _split_header(h: str) -> list:
    #     if not h:
    #         return []
    #     if "|" in h:
    #         return [c.strip() for c in h.split("|") if c.strip()]
    #     return re.split(r"\t+|\s{3,}", h.strip())

    # def _infer_sample_colcount(sheet_text: str) -> int:
    #     counts: Dict[int, int] = {}
    #     lines = sheet_text.splitlines()
    #     # skip potential header (row 0), use next 10 lines with tabs
    #     for line in lines[1:11]:
    #         if ":" in line:
    #             # strip row index prefix 'N: '
    #             line = line.split(":", 1)[1].strip()
    #         cells = [c for c in line.split("\t") if c != ""]
    #         if cells:
    #             counts[len(cells)] = counts.get(len(cells), 0) + 1
    #     if not counts:
    #         return 0
    #     # most frequent length
    #     return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # if not verification_report or not isinstance(verification_report, dict) or not verification_report.get("final"):
    #     # Build candidate set
    #     cands: Dict[str, str] = {"initial": initial_header}
    #     if isinstance(verification_report, dict):
    #         cand_block = verification_report.get("candidates") or {}
    #         for key in ("text", "p1_t1", "p2_t1"):
    #             val = cand_block.get(key)
    #             if isinstance(val, dict) and val.get("header"):
    #                 cands[key] = val["header"]
    #     # Determine expected column count from p1_t1 then p2_t1
    #     exp_cols = _infer_sample_colcount(parts3.get("p1_t1_sheet", "")) or _infer_sample_colcount(parts3.get("p2_t1_sheet", ""))
    #     best_name, best_header, best_diff = None, initial_header, float("inf")
    #     for name, hdr in cands.items():
    #         diff = abs(len(_split_header(hdr)) - exp_cols) if exp_cols else (0 if hdr else 999)
    #         # prefer non-initial on tie if it has pipes
    #         tie_break = (0 if ("|" in hdr and name != "initial") else 1)
    #         key = (diff, tie_break)
    #         if best_name is None or key < (best_diff, 1):
    #             best_name, best_header, best_diff = name, hdr, diff
    #     final_header = best_header or initial_header
    #     final_cols = _split_header(final_header)
    #     verification_report = verification_report if isinstance(verification_report, dict) else {}
    #     verification_report["final"] = {
    #         "table_header": final_header,
    #         "header_columns": final_cols,
    #         "header_valid": exp_cols == len(final_cols) if exp_cols else True,
    #         "header_reason": "auto-selected best match by column count vs sample",
    #         "source": best_name or "initial",
    #     }

    # 3) Build transactions and write outputs
    print("[step] Building transactions with final header ...")
    tx_df = build_transactions_simple(parts["xls"], data, final_header, start_idx)
    out = {
        "metadata": data.get("metadata", {}),
        "table_start_index": start_idx,
        "table_header": final_header,
        "verification": verification_report or {"raw": vraw},
    }
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with pd.ExcelWriter(args.excel_out, engine="openpyxl") as writer:
        pd.DataFrame(out["metadata"].items(), columns=["Field", "Value"]).to_excel(writer, sheet_name="metadata", index=False)
        tx_df.to_excel(writer, sheet_name="transactions", index=False)
    print(f"✅ Wrote JSON: {args.json_out}\n✅ Wrote Excel: {args.excel_out}")


if __name__ == "__main__":
    main()


