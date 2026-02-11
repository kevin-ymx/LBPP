"""
Data cleaning script for extracted_results_mol.csv: keep only rows where
molecule_name is a specific molecule name. Uses a model API (OpenAI) to classify
each unique molecule_name as specific vs generic, then filters the CSV.
Supports resume: classification cache is saved to a JSON file and reused on restart.
"""
import argparse
import csv
import json
import os
import time
from typing import Dict, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
# API key: set OPENAI_API_KEY env var, or assign below (same as abs_extract.py)
OPENAI_API_KEY = "api_key"

MODEL_NAME = "gpt-5-mini"  # match abs_extract.py (e.g. gpt-5-mini) if needed
INPUT_CSV = "extracted_results_mol_cleaned.csv"
OUTPUT_CSV = "extracted_results_mol_cleaned_v2.csv"
CACHE_SUFFIX = "_name_cache.json"  # resume cache: {molecule_name: is_specific}
SLEEP_BETWEEN_CALLS = 0.1  # seconds (rate limit safety)

SYSTEM_PROMPT = """You are a chemistry/data curator. Your task is to decide whether a given string is a SPECIFIC molecule/compound name or NOT.

SPECIFIC molecule names include:
- Named compounds (e.g. benzoguanamine, guanabenz acetate salt, Me-4PACz, PCBM)
- IUPAC-style or systematic names (e.g. 1,3-diaminopropane dihydroiodide)
- Well-known molecules with a single identity (e.g. fullerene, C-60, isopropanol, phosphonic acid, carbazole, triphenylamine)
- Salts or adducts with a defined composition (e.g. dimethylphenethylsulfonium iodide (DMPESI))

NOT specific (generic/category) include:
- Plural or category phrases (e.g. "sulfonium-based molecules", "passivator molecules", "conventional ionic molecules")
- Material/layer descriptions (e.g. "two-dimensional perovskite interlayers", "hole-transporting layers (HTL)", "perovskite absorber")
- Vague descriptors (e.g. "a passivating dipole layer having high molecular polarity", "evaporable organic molecules", "self-assembled monolayer molecule")
- Class names (e.g. "diradical SAMs", "Organic self-assembled molecules (SAMs)" when referring to a class)
- Element names or element abbreviations (e.g. lead, tin, oxygen, Pb, Sn, O, Cd, Zn, Hg)
- A specific perovskite type or perovskite material class (e.g. "formamidinium lead iodide perovskite", "tin-lead mixed perovskite", "metal halide perovskites")
- ITO (indium tin oxide)
- Water
- MA molecule or FA molecule (methylammonium / formamidinium as a category, or the abbreviations MA, FA when referring to that molecule)
- Binary compounds (e.g. NiO, SnO2, TiO2, ZnO, NiOx)
- Ions (e.g. iodide, bromide, chloride, cation, anion, or names that are clearly an ion)
- Empty or whitespace-only strings

Reply with exactly one word: YES if the string is a specific molecule name, NO otherwise."""

USER_PROMPT_TEMPLATE = """Molecule name to classify (exactly as in the dataset):

"{molecule_name}"

Answer (YES or NO):"""


def get_client() -> OpenAI:
    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Set the OPENAI_API_KEY environment variable "
            "or assign it in this script (OPENAI_API_KEY = '...')."
        )
    return OpenAI(api_key=api_key)


def is_specific_molecule_name(client: OpenAI, name: str) -> bool:
    """Call the model API to classify name as specific (True) or not (False)."""
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(molecule_name=name)}"
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        raw = (response.output_text or "").strip().upper()
        # First token (allow "YES." or "YES\n" etc.)
        first = raw.split(None, 1)[0] if raw else ""
        return first.startswith("YES")
    except Exception as e:
        # On API error, treat as not specific to avoid keeping ambiguous rows
        print(f"  [API error for '{name[:50]}...']: {e}")
        return False


def get_cache_path(input_csv_path: str) -> str:
    """Path for resume cache: same directory and base name as input, with CACHE_SUFFIX."""
    base, _ = os.path.splitext(os.path.abspath(input_csv_path))
    return base + CACHE_SUFFIX


def load_cache(cache_path: str) -> Dict[str, bool]:
    """Load classification cache from JSON; return {} if file missing or invalid."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): bool(v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  [Cache invalid or empty, starting fresh]: {e}")
        return {}


def save_cache(cache_path: str, name_to_specific: dict[str, bool]) -> None:
    """Write classification cache to JSON for resume."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(name_to_specific, f, indent=0, ensure_ascii=False)


def load_unique_molecule_names(csv_path: str) -> Set[str]:
    """Return set of unique non-empty molecule_name values from the CSV."""
    names = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "molecule_name" not in (reader.fieldnames or []):
            raise ValueError("CSV must have a 'molecule_name' column")
        for row in reader:
            name = (row.get("molecule_name") or "").strip()
            if name:
                names.add(name)
    return names


def classify_all_names(
    client: OpenAI,
    names: Set[str],
    sleep: float = SLEEP_BETWEEN_CALLS,
    cache: Optional[Dict[str, bool]] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Classify each name; return dict name -> True if specific, False otherwise.
    If cache is provided, only classifies names not in cache and updates cache
    (and saves to cache_path after each call if cache_path is set) for resume.
    """
    result = dict(cache) if cache is not None else {}
    to_classify = sorted(names - result.keys())
    if not to_classify:
        return result
    for name in tqdm(to_classify, desc="Classifying molecule names"):
        result[name] = is_specific_molecule_name(client, name)
        if cache_path:
            save_cache(cache_path, result)
        time.sleep(sleep)
    return result


def clean_csv(
    input_path: str,
    output_path: str,
    name_to_specific: Dict[str, bool],
) -> Tuple[int, int]:
    """
    Read input CSV, keep only rows where molecule_name is non-empty and
    classified as specific; write to output_path. Returns (rows_kept, rows_removed).
    """
    kept = 0
    removed = 0
    rows_out = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            name = (row.get("molecule_name") or "").strip()
            if not name:
                removed += 1
                continue
            if name_to_specific.get(name, False):
                rows_out.append(row)
                kept += 1
            else:
                removed += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return kept, removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean extracted_results_mol.csv by keeping only rows with specific molecule names (API-based classification)."
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load CSV and list unique molecule names; do not call API or write output.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_BETWEEN_CALLS,
        help=f"Seconds to sleep between API calls (default: {SLEEP_BETWEEN_CALLS})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing cache and reclassify all molecule names.",
    )
    parser.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="Path to classification cache JSON (default: <input_base>%s)" % CACHE_SUFFIX,
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    unique_names = load_unique_molecule_names(input_path)
    print(f"Found {len(unique_names)} unique non-empty molecule names in {input_path}")

    if args.dry_run:
        for n in sorted(unique_names)[:30]:
            print(f"  {n}")
        if len(unique_names) > 30:
            print(f"  ... and {len(unique_names) - 30} more")
        return

    cache_path = args.cache if args.cache else get_cache_path(input_path)
    cache = {} if args.no_resume else load_cache(cache_path)
    if cache:
        remaining = len(unique_names - cache.keys())
        print(f"Resume: loaded {len(cache)} cached classifications from {cache_path}; {remaining} names left to classify.")

    client = get_client()
    name_to_specific = classify_all_names(
        client,
        unique_names,
        sleep=args.sleep,
        cache=cache,
        cache_path=cache_path,
    )
    specific_count = sum(1 for v in name_to_specific.values() if v)
    print(f"Classified: {specific_count} specific, {len(name_to_specific) - specific_count} not specific")

    kept, removed = clean_csv(input_path, output_path, name_to_specific)
    print(f"Wrote {output_path}: {kept} rows kept, {removed} rows removed.")


if __name__ == "__main__":
    main()
