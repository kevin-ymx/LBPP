"""
Clean binding energy result CSVs by removing rows that:
  (1) have all zeros in pb_bond_encoding (no atom binds to Pb2+), or
  (2) have any H (atomic number 1) or C (atomic number 6) binding to Pb2+.

Input: folder containing any number of CSV files.
Output: cleaned CSVs with same filenames in output folder, or a single merged CSV with --merge.

Usage:
  python clean_binding_results.py --input_dir /path/to/shards --output_dir /path/to/shards_cleaned
  python clean_binding_results.py --input_dir /path/to/shards --output_dir /path/to/out --merge
"""
import argparse
import ast
import csv
import glob
import json
import os
import sys
from typing import List, Optional, Tuple

from tqdm import tqdm

REQUIRED_COLUMNS = ["cid", "functional_group", "formula", "pb_bond_encoding", "adsorption_energy", "config_name", "adsorbate_structure"]


def parse_pb_bond_encoding(s: str) -> Optional[List[int]]:
    """Parse pb_bond_encoding string to list of 0/1. Returns None if invalid."""
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        # Handle "[0,0,0,0,0,1,0,0,0,0]"
        out = ast.literal_eval(s)
        if not isinstance(out, list):
            return None
        if not all(x in (0, 1) for x in out):
            return None
        return out
    except (ValueError, SyntaxError):
        return None


def parse_adsorbate_structure(s: str) -> Optional[dict]:
    """Parse adsorbate_structure JSON string. Returns None if invalid."""
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def get_element_numbers(struct: dict) -> Optional[List[int]]:
    """Get elements['number'] list from parsed adsorbate_structure. Same index as pb_bond_encoding."""
    if not struct or not isinstance(struct, dict):
        return None
    elements = struct.get("elements")
    if not elements or not isinstance(elements, dict):
        return None
    numbers = elements.get("number")
    if not numbers or not isinstance(numbers, list):
        return None
    return numbers


def should_remove_row(encoding: List[int], element_numbers: Optional[List[int]]) -> Tuple[bool, str]:
    """
    Decide if row should be removed.
    Returns (remove, reason).
    - Remove if encoding is all zeros.
    - Remove if any binding atom (encoding[i]==1) has atomic number 1 (H).
    - Remove if any binding atom (encoding[i]==1) has atomic number 6 (C).
    - If we cannot parse encoding or structure, remove to be safe.
    """
    if not encoding:
        return True, "empty_encoding"
    if all(x == 0 for x in encoding):
        return True, "all_zero"
    if element_numbers is None:
        return True, "no_structure"
    if len(element_numbers) != len(encoding):
        return True, "length_mismatch"
    for i, v in enumerate(encoding):
        if v != 1:
            continue
        an = element_numbers[i]
        if an == 1:
            return True, "H_binds"
        if an == 6:
            return True, "C_binds"
    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Clean binding result CSVs: drop all-zero encoding, H-binding, and C-binding rows.")
    parser.add_argument("--input_dir", required=True, help="Folder containing CSV files")
    parser.add_argument("--output_dir", required=True, help="Output folder (or single file if --merge)")
    parser.add_argument("--merge", action="store_true", help="Write one merged CSV instead of per-file outputs")
    parser.add_argument("--suffix", default=".csv", help="Only process files with this suffix (default: .csv)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover all CSV files in the folder
    pattern = os.path.join(input_dir, "*" + args.suffix)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No files matching {pattern} found.", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(input_files)} file(s) in {input_dir}")

    if not args.merge:
        os.makedirs(output_dir, exist_ok=True)

    total_read = 0
    total_written = 0
    removed_all_zero = 0
    removed_h_binds = 0
    removed_c_binds = 0
    removed_other = 0

    merge_fieldnames = None
    merged_rows = [] if args.merge else None

    for in_path in tqdm(input_files, desc="Files", unit="file"):
        filename = os.path.basename(in_path)

        with open(in_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            for col in REQUIRED_COLUMNS:
                if col not in fieldnames:
                    print(f"Error: missing column '{col}' in {in_path}. Found: {fieldnames}", file=sys.stderr)
                    sys.exit(1)
            rows = list(reader)

        kept = []
        for row in tqdm(rows, desc=filename, leave=False, unit="row"):
            total_read += 1
            enc_str = row.get("pb_bond_encoding", "")
            struct_str = row.get("adsorbate_structure", "")
            encoding = parse_pb_bond_encoding(enc_str)
            struct = parse_adsorbate_structure(struct_str)
            element_numbers = get_element_numbers(struct)

            remove, reason = should_remove_row(encoding, element_numbers)
            if remove:
                if reason == "all_zero":
                    removed_all_zero += 1
                elif reason == "H_binds":
                    removed_h_binds += 1
                elif reason == "C_binds":
                    removed_c_binds += 1
                else:
                    removed_other += 1
                continue
            kept.append(row)
            total_written += 1

        if args.merge:
            if merge_fieldnames is None:
                merge_fieldnames = fieldnames
            merged_rows.extend(kept)
        else:
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)

    if args.merge and merge_fieldnames:
        out_path = os.path.join(output_dir, "min_ads_mult1p2_struct_cleaned_merged.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=merge_fieldnames)
            w.writeheader()
            w.writerows(merged_rows)
        print(f"Merged output: {out_path}")

    print(f"Total rows read:    {total_read}")
    print(f"Total rows kept:   {total_written}")
    print(f"Removed (all 0):   {removed_all_zero}")
    print(f"Removed (H binds): {removed_h_binds}")
    print(f"Removed (C binds): {removed_c_binds}")
    print(f"Removed (other):   {removed_other}")


if __name__ == "__main__":
    main()
