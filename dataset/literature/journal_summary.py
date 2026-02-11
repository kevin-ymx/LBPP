"""
Aggregate extracted_results_mol.csv by journal: output a CSV with
Journal name, Impact factor, Paper count. Sorted by impact factor (high to low).
"""
import argparse
import csv
import os
from collections import defaultdict


INPUT_CSV = "extracted_results_mol.csv"
OUTPUT_CSV = "journal_summary.csv"
OUTPUT_COLUMNS = ["Journal name", "Impact factor", "Paper count"]


def parse_impact_factor(value):
    """Parse impact factor to float; return 0.0 if missing or invalid."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return 0.0
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return 0.0


def aggregate_journals(csv_path: str):
    """
    Read CSV and return list of (journal_name, impact_factor, paper_count).
    Paper count = number of unique titles per journal.
    """
    # journal -> (impact_factor, set of titles)
    journal_info = defaultdict(lambda: [None, set()])

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "journal" not in reader.fieldnames:
            raise ValueError("CSV must have a 'journal' column")
        title_col = "title" if "title" in (reader.fieldnames or []) else None
        if_col = "impact_factor" if "impact_factor" in (reader.fieldnames or []) else None

        row_index = 0
        for row in reader:
            row_index += 1
            journal = (row.get("journal") or "").strip()
            if not journal:
                continue
            impact = parse_impact_factor(row.get("impact_factor")) if if_col else 0.0
            # Keep highest IF if we see multiple (should be same per journal)
            prev_if, titles = journal_info[journal]
            if prev_if is None or impact > prev_if:
                journal_info[journal][0] = impact
            if title_col:
                title = (row.get("title") or "").strip()
                if title:
                    journal_info[journal][1].add(title)
            else:
                journal_info[journal][1].add(row_index)  # fallback: count rows per journal

    rows = []
    for journal, (impact, titles) in journal_info.items():
        paper_count = len(titles) if titles else 0
        rows.append((journal, impact, paper_count))

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize extracted_results_mol.csv by journal: Journal name, Impact factor, Paper count, preserving journal order from the input CSV."
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
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    rows = aggregate_journals(args.input)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    print(f"Wrote {args.output}: {len(rows)} journals, preserving journal order from {args.input}.")


if __name__ == "__main__":
    main()
