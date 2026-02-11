"""
Extract PubChem CIDs for all molecule names in column 5 of
extracted_results_mol_cleaned_v2.csv. Updates column 6 (molecule_cid) in place.
Uses the same CID lookup rules as abs_extract.py: for names ending with
" (abbreviation)", strip the parenthetical only for the PubChem query.
"""
import argparse
import csv
import os
import re
import time
from typing import Dict, Optional, Set

import requests
from tqdm import tqdm

# -----------------------
# CONFIG (match abs_extract.py)
# -----------------------
INPUT_CSV = "extracted_results_mol_cleaned_v2.csv"
PUBCHEM_API_TIMEOUT = 10.0  # seconds
SLEEP_BETWEEN_CALLS = 0.2  # seconds
_ABBR_PAREN_MAX_LEN = 20

# Column 5 = molecule_name (0-based index 4), column 6 = molecule_cid (0-based index 5).
MOLECULE_NAME_COLUMN_INDEX = 4
MOLECULE_CID_COLUMN_INDEX = 5
MOLECULE_NAME_HEADER = "molecule_name"
MOLECULE_CID_HEADER = "molecule_cid"


def name_for_cid_lookup(molecule_name: str) -> str:
    """
    For CID lookup: strip only a trailing parenthetical that looks like an
    abbreviation (space before '(', content short and no comma).
    """
    if not molecule_name:
        return molecule_name
    name = molecule_name.strip()
    m = re.match(r"^(.+)\s+\(([^)]*)\)\s*$", name)
    if m:
        prefix, in_paren = m.group(1).rstrip(), m.group(2)
        if "," not in in_paren and len(in_paren) <= _ABBR_PAREN_MAX_LEN:
            return prefix
    return name


def get_pubchem_cid(molecule_name: str) -> Optional[int]:
    """Look up PubChem CID by molecule name (CID only, no SMILES)."""
    if not molecule_name or molecule_name.lower() == "null":
        return None
    name = name_for_cid_lookup(molecule_name.strip())
    if not name:
        return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
        response = requests.get(url, timeout=PUBCHEM_API_TIMEOUT)
        if response.status_code != 200:
            return None
        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        return cids[0]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PubChem CIDs for molecule names in column 5; update column 6 (molecule_cid) in place."
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_BETWEEN_CALLS,
        help="Seconds to sleep between PubChem API calls",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input

    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    # Read full file and find column indices
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = list(next(reader))
        name_col = header.index(MOLECULE_NAME_HEADER) if MOLECULE_NAME_HEADER in header else MOLECULE_NAME_COLUMN_INDEX
        cid_col = header.index(MOLECULE_CID_HEADER) if MOLECULE_CID_HEADER in header else MOLECULE_CID_COLUMN_INDEX
        rows = list(reader)

    # Unique non-empty molecule names
    unique_names: Set[str] = set()
    for row in rows:
        if len(row) > name_col:
            name = (row[name_col] or "").strip()
            if name:
                unique_names.add(name)

    name_to_cid: Dict[str, Optional[int]] = {}
    to_lookup = sorted(unique_names)
    for name in tqdm(to_lookup, desc="PubChem CID lookup"):
        cid = get_pubchem_cid(name)
        name_to_cid[name] = cid
        time.sleep(args.sleep)

    # Fill molecule_cid in each row and write back to the same file
    for row in rows:
        while len(row) <= cid_col:
            row.append("")
        name = (row[name_col] or "").strip() if len(row) > name_col else ""
        cid = name_to_cid.get(name) if name else None
        row[cid_col] = str(cid) if cid is not None else ""

    while len(header) <= cid_col:
        header.append(MOLECULE_CID_HEADER if len(header) == cid_col else "")
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Updated {input_path}: column {cid_col + 1} ({MOLECULE_CID_HEADER}) filled with PubChem CIDs.")


if __name__ == "__main__":
    main()
