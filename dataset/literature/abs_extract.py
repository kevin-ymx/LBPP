import csv
import json
import os
import re
import time
from typing import List, Optional, Dict
from openai import OpenAI
import requests
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
# OpenAI API key - paste your key below (replace YOUR_API_KEY_HERE)
OPENAI_API_KEY = "api_key"

MODEL_NAME = "gpt-5-mini"  # or gpt-4.1 / gpt-4o / gpt-4.1-mini / gpt-5-mini
INPUT_FILE = "abstract_LB_525.txt"  # WOS export format (SO=journal, AB=abstract, ER=end record)
OUTPUT_JSON = "extracted_results_LB_525.json"  # JSON output sorted by impact factor (high to low)
OUTPUT_CSV = "extracted_results_LB_525.csv"  # CSV table output (excludes claimed_mechanisms)
SLEEP_BETWEEN_CALLS = 0.2  # seconds (rate limit safety)
PUBCHEM_API_TIMEOUT = 10.0  # seconds

# Initialize OpenAI client
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Set environment variable or configure in script (line 16).")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# PUBCHEM CID AND SMILES LOOKUP
# -----------------------
def get_pubchem_cid_and_smiles(molecule_name: str) -> tuple:
    """
    Look up PubChem Compound ID (CID) and SMILES for a molecule by name.
    
    Args:
        molecule_name: Name of the molecule (can be IUPAC name, common name, or synonym).
        
    Returns:
        Tuple of (CID, SMILES) - both can be None if not found.
    """
    if not molecule_name or molecule_name.lower() == "null":
        return None, None
    
    # Clean up the name: strip whitespace and remove content in parentheses
    # e.g., "phenethylammonium iodide (PEAI)" -> "phenethylammonium iodide"
    name = molecule_name.strip()
    name = re.sub(r'\s*\([^)]*\)', '', name).strip()
    
    if not name:
        return None, None
    
    try:
        # Step 1: Get CID from name
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
        response = requests.get(url, timeout=PUBCHEM_API_TIMEOUT)
        
        if response.status_code != 200:
            return None, None
        
        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None, None
        
        cid = cids[0]  # First matching CID
        
        # Step 2: Get SMILES from CID
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        smiles_response = requests.get(smiles_url, timeout=PUBCHEM_API_TIMEOUT)
        
        smiles = None
        if smiles_response.status_code == 200:
            smiles_data = smiles_response.json()
            properties = smiles_data.get("PropertyTable", {}).get("Properties", [])
            if properties:
                smiles = properties[0].get("CanonicalSMILES")
        
        return cid, smiles
    except Exception:
        return None, None


def get_pubchem_cid_batch(molecule_names: List[str]) -> Dict[str, tuple]:
    """
    Look up PubChem CIDs and SMILES for multiple molecules.
    
    Args:
        molecule_names: List of molecule names.
        
    Returns:
        Dictionary mapping molecule names to (CID, SMILES) tuples.
    """
    results = {}
    for name in molecule_names:
        if name:
            results[name] = get_pubchem_cid_and_smiles(name)
            time.sleep(0.2)  # Rate limiting for PubChem API
    return results


# -----------------------
# JOURNAL IMPACT FACTOR LOOKUP
# -----------------------
# Journals in materials science, chemistry, and energy research with approximate impact factors (2023-2024)
# Note: Impact factors change yearly. Update as needed.
# Ordered by impact factor (high to low) for reference.
JOURNAL_IMPACT_FACTORS = {
    # Very high impact (IF >50)
    "NATURE REVIEWS MATERIALS": 86.2,
    "NATURE": 48.5,
    "CHEMICAL REVIEWS": 55.8,
    "SCIENCE": 45.8,
    "NATURE ENERGY": 60.1,
    "NATURE REVIEWS CHEMISTRY": 51.7,
    # High impact (IF 20-50)
    "JOULE": 35.4,
    "CHEMICAL SOCIETY REVIEWS": 39.0,
    "NATURE MATERIALS": 38.5,
    "NATURE NANOTECHNOLOGY": 37.2,
    "NATURE CHEMISTRY": 20.2,
    "NATURE ELECTRONICS": 40.9,
    "NATURE PHOTONICS": 32.9,
    "ENERGY & ENVIRONMENTAL SCIENCE": 29.1,
    "ADVANCED MATERIALS": 30.2,
    "ADVANCED ENERGY MATERIALS": 26.9,
    "CHEM": 23.5,
    "ACS ENERGY LETTERS": 22.0,
    "MATERIALS TODAY": 21.1,
    "MATTER": 19.7,
    "ADVANCED FUNCTIONAL MATERIALS": 19.5,
    "CARBON ENERGY": 20.5,
    "INFOMAT": 22.7,
    "ENERGY STORAGE MATERIALS": 20.4,
    "APPLIED CATALYSIS B ENVIRONMENTAL": 21.1,
    "SCIENCE BULLETIN": 18.9,
    "NATIONAL SCIENCE REVIEW": 20.6,
    "ADVANCES IN OPTICS AND PHOTONICS": 23.8,
    "OPTO-ELECTRONIC ADVANCES": 22.4,
    # Medium-high impact (IF 15-20)
    "NANO ENERGY": 16.8,
    "NATURE COMMUNICATIONS": 15.7,
    "ANGEWANDTE CHEMIE INTERNATIONAL EDITION": 16.1,
    "ANGEWANDTE CHEMIE": 16.1,
    "ACS NANO": 16.0,
    "JOURNAL OF THE AMERICAN CHEMICAL SOCIETY": 15.6,
    "CHEMICAL ENGINEERING JOURNAL": 15.1,
    "ADVANCED SCIENCE": 14.3,
    "SCIENCE ADVANCES": 13.6,
    "SMALL": 13.3,
    "ACS CATALYSIS": 11.7,
    "JOURNAL OF ENERGY CHEMISTRY": 14.0,
    "MATERIALS HORIZONS": 10.7,
    "SMALL METHODS": 12.4,
    "SMALL STRUCTURES": 12.0,
    "MATERIALS TODAY PHYSICS": 11.5,
    "ECOMAT": 11.8,
    "JOURNAL OF MATERIALS CHEMISTRY A": 11.9,
    "GREEN CHEMISTRY": 11.0,
    "CARBON": 10.9,
    "NANO LETTERS": 9.1,
    "RESEARCH": 11.0,
    # Medium impact (IF 10-15)
    "JOURNAL OF POWER SOURCES": 8.1,
    "MATERIALS TODAY ENERGY": 8.6,
    "CHEMISTRY OF MATERIALS": 7.0,
    "SOLAR ENERGY MATERIALS AND SOLAR CELLS": 6.3,
    "SOLAR RRL": 6.0,
    "ISCIENCE": 5.8,
    "SUSTAINABLE ENERGY & FUELS": 5.8,
    "ELECTROCHIMICA ACTA": 5.5,
    "JOURNAL OF MATERIALS CHEMISTRY C": 6.4,
    "JOURNAL OF MATERIALS CHEMISTRY B": 5.8,
    "JOURNAL OF PHYSICAL CHEMISTRY LETTERS": 4.6,
    "JOURNAL OF PHYSICAL CHEMISTRY C": 3.7,
    "ACS APPLIED ENERGY MATERIALS": 6.4,
    "ACS SUSTAINABLE CHEMISTRY & ENGINEERING": 8.4,
    "INORGANIC CHEMISTRY": 4.6,
    "CHEMSUSCHEM": 8.4,
    "NANOSCALE": 6.7,
    "JOURNAL OF COLLOID AND INTERFACE SCIENCE": 9.9,
    "APPLIED SURFACE SCIENCE": 6.7,
    "ADVANCED OPTICAL MATERIALS": 9.0,
    "ADVANCED ELECTRONIC MATERIALS": 6.2,
    "BATTERIES & SUPERCAPS": 5.3,
    "MATERIALS TODAY ADVANCES": 8.1,
    "MATERIALS TODAY CHEMISTRY": 7.3,
    "MATERIALS TODAY SUSTAINABILITY": 7.1,
    "ADVANCED MATERIALS INTERFACES": 5.4,
    "ADVANCED MATERIALS TECHNOLOGIES": 7.0,
    # Standard impact (IF 5-10)
    "CELL REPORTS PHYSICAL SCIENCE": 8.9,
    "ACS APPLIED MATERIALS & INTERFACES": 8.2,
    "SURFACE AND COATINGS TECHNOLOGY": 5.4,
    "JOURNAL OF ALLOYS AND COMPOUNDS": 6.2,
    "MATERIALS RESEARCH BULLETIN": 5.4,
    "MATERIALS SCIENCE AND ENGINEERING B": 4.6,
    "JOURNAL OF THE ELECTROCHEMICAL SOCIETY": 3.4,
    "ELECTROCHEMISTRY COMMUNICATIONS": 4.1,
    "PHYSICAL CHEMISTRY CHEMICAL PHYSICS": 3.3,
    "DALTON TRANSACTIONS": 4.0,
    "NEW JOURNAL OF CHEMISTRY": 3.3,
    "CRYSTENGCOMM": 2.6,
    "CRYSTAL GROWTH & DESIGN": 3.8,
    "JOURNAL OF CRYSTAL GROWTH": 1.8,
    "RSC ADVANCES": 4.6,
    "MATERIALS CHEMISTRY AND PHYSICS": 4.7,
    "JOURNAL OF SOLID STATE CHEMISTRY": 3.5,
    "SOLID STATE SCIENCES": 3.1,
    "JOURNAL OF PHYSICS D APPLIED PHYSICS": 3.4,
    "APPLIED PHYSICS LETTERS": 3.5,
    "JOURNAL OF APPLIED PHYSICS": 2.9,
    "AIP ADVANCES": 1.6,
    "SCIENTIFIC REPORTS": 4.6,
    "PLOS ONE": 3.7,
    "FRONTIERS IN CHEMISTRY": 5.5,
    "FRONTIERS IN MATERIALS": 3.2,
    "MOLECULES": 4.6,
    "MATERIALS": 3.2,
    "NANOMATERIALS": 5.3,
    "POLYMERS": 5.0,
    "CATALYSTS": 3.9,
    "ENERGIES": 3.2,
    "CHEMELECTROCHEM": 4.2,
    "CHEMPHOTOCHEM": 3.9,
    "CHEMNANOMAT": 3.8,
    "CHEMPHYSCHEM": 3.0,
    "FRONTIERS OF OPTOELECTRONICS": 5.2,
    # Lower impact (IF < 5)
    "MATERIALS LETTERS": 3.0,
    "PHYSICA STATUS SOLIDI RAPID RESEARCH LETTERS": 2.5,
    "PHYSICA STATUS SOLIDI A": 2.0,
    "PHYSICA STATUS SOLIDI B": 1.8,
    "THIN SOLID FILMS": 2.1,
    "IET OPTOELECTRONICS": 1.6,
    "OPTO-ELECTRONICS REVIEW": 0.9,
    "JOURNAL OF OPTOELECTRONICS AND ADVANCED MATERIALS": 0.6,
    "JOURNAL OF THE EUROPEAN OPTICAL SOCIETY - RAPID PUBLICATIONS": 3.2,
    "OPTICS": 1.6,
    "JOURNAL OF OPTOELECTRONIC AND BIOMEDICAL MATERIALS": 1.1
}


def get_journal_impact_factor(journal_name: str) -> Optional[float]:
    """
    Look up approximate impact factor for a journal.
    
    Args:
        journal_name: Name of the journal.
        
    Returns:
        Impact factor if found in database, None otherwise.
    """
    if not journal_name or journal_name.lower() == "null":
        return None
    
    # Normalize name for lookup (uppercase to match dictionary keys)
    name_upper = journal_name.upper().strip()
    
    # Try both "AND" and "&" variants
    name_variants = [
        name_upper,
        name_upper.replace(" AND ", " & "),
        name_upper.replace(" & ", " AND "),
    ]
    
    for name in name_variants:
        # Direct match
        if name in JOURNAL_IMPACT_FACTORS:
            return JOURNAL_IMPACT_FACTORS[name]
    
    # Partial match (journal name contains key or key contains journal name)
    for name in name_variants:
        for key, value in JOURNAL_IMPACT_FACTORS.items():
            if key in name or name in key:
                return value
    
    return None

# -----------------------
# PROMPTS
# -----------------------
SYSTEM_PROMPT = """You are an information extraction system for scientific literature.

Your task is to extract structured information ONLY from the provided abstract.
Do NOT infer, guess, or use outside knowledge.
If information is not explicitly stated, output null.

You must:
- Follow the provided JSON schema exactly
- Return strictly valid JSON
- Use arrays where specified
- Preserve original wording in evidence fields
- Never add explanatory text outside JSON
"""

USER_PROMPT_TEMPLATE = """Extract information from the abstract below and return it strictly in the following JSON schema.

JSON SCHEMA:
{{
  "paper_metadata": {{
    "title": null,
    "year": null,
    "journal": null,
    "impact_factor": null
  }},
  "molecules": [
    {{
      "name": null,
      "cid": null,
      "smiles": null,
      "type": null,
      "functional_groups": [],
      "role": null,
      "interface_location": null,
      "evidence": null
    }}
  ],
  "device_metrics": {{
    "pce_max": {{ "value": null, "units": "%", "evidence": null }},
    "voc": {{ "value": null, "units": "V", "evidence": null }},
    "jsc": {{ "value": null, "units": "mA/cm2", "evidence": null }},
    "ff": {{ "value": null, "units": "%", "evidence": null }}
  }},
  "stability_metrics": [
    {{
      "metric_type": null,
      "value": null,
      "units": null,
      "test_conditions": null,
      "evidence": null
    }}
  ],
  "perovskite_type": {{ "value": null, "evidence": null }},
  "claimed_mechanisms": [
    {{ "mechanism": null, "evidence": null }}
  ]
}}

EXTRACTION RULES:
- Only extract information explicitly stated in the abstract
- If multiple molecules are mentioned, list all separately
- Do not merge molecules
- If a value is missing or unclear, set it to null
- Evidence must be a direct quote or close paraphrase from the abstract
- Do NOT normalize names or abbreviations beyond what is written

ABSTRACT:
<<<
{abstract}
>>>
"""

# -----------------------
# HELPERS
# -----------------------
def parse_wos_record(record: str) -> Dict[str, str]:
    """
    Parse a Web of Science (WOS) record and extract title (TI), journal (SO), and abstract (AB).
    
    Args:
        record: Raw WOS record text.
        
    Returns:
        Dict with 'title', 'journal', and 'abstract' keys.
    """
    title = None
    journal = None
    abstract = None
    
    lines = record.strip().split('\n')
    current_field = None
    current_value = []
    
    # Fields we care about
    target_fields = ('TI', 'SO', 'AB')
    
    for line in lines:
        # Check if line starts with a 2-letter field code
        if len(line) >= 2 and line[:2].isupper() and (len(line) == 2 or line[2] == ' '):
            # Save previous field if it was one we care about
            if current_field == 'TI' and current_value:
                title = ' '.join(current_value).strip()
            elif current_field == 'SO' and current_value:
                journal = ' '.join(current_value).strip()
            elif current_field == 'AB' and current_value:
                abstract = ' '.join(current_value).strip()
            
            # Start new field
            current_field = line[:2]
            current_value = [line[3:].strip()] if len(line) > 3 else []
        elif current_field in target_fields:
            # Continuation line for fields we care about
            current_value.append(line.strip())
    
    # Don't forget the last field
    if current_field == 'TI' and current_value:
        title = ' '.join(current_value).strip()
    elif current_field == 'SO' and current_value:
        journal = ' '.join(current_value).strip()
    elif current_field == 'AB' and current_value:
        abstract = ' '.join(current_value).strip()
    
    return {'title': title, 'journal': journal, 'abstract': abstract}


def load_abstracts(path: str) -> List[Dict[str, str]]:
    """
    Load abstracts from WOS export file.
    
    Each record is separated by 'ER' (end of record) line.
    Extracts title (TI), journal name (SO), and abstract (AB) from each record.
    
    Args:
        path: Path to abstracts.txt file.
        
    Returns:
        List of dicts with 'title', 'journal', and 'abstract' keys.
    """
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by ER (end of record) - WOS format
    records = re.split(r'\nER\s*\n', text)
    
    results = []
    for record in records:
        record = record.strip()
        if not record:
            continue
        
        parsed = parse_wos_record(record)
        if parsed['abstract']:
            results.append(parsed)
    
    return results


def call_gpt(abstract: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(abstract=abstract)}"
    
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )
    
    raw_text = response.output_text
    
    if not raw_text:
        raise ValueError("Empty response from API")
    
    # Clean up response - remove markdown code blocks if present
    text = raw_text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    return json.loads(text)


def enrich_with_external_data(result: dict) -> dict:
    """
    Enrich extracted results with external data:
    - Journal impact factor
    - PubChem CIDs for molecules
    
    Args:
        result: Extracted result from GPT.
        
    Returns:
        Enriched result dictionary.
    """
    # Add journal impact factor
    journal = result.get("paper_metadata", {}).get("journal")
    if journal:
        impact_factor = get_journal_impact_factor(journal)
        result["paper_metadata"]["impact_factor"] = impact_factor
        if impact_factor:
            print(f"    Impact factor for '{journal}': {impact_factor}")
    
    # Add PubChem CIDs and SMILES for molecules
    molecules = result.get("molecules", [])
    if molecules:
        for mol in molecules:
            name = mol.get("name")
            if name and name.lower() != "null":
                cid, smiles = get_pubchem_cid_and_smiles(name)
                mol["cid"] = cid
                mol["smiles"] = smiles
                if cid:
                    print(f"    PubChem CID for '{name}': {cid}")
                if smiles:
                    print(f"    SMILES for '{name}': {smiles}")
                time.sleep(0.2)  # Rate limiting
    
    return result


def get_impact_factor_for_sorting(result: dict) -> float:
    """Get impact factor for sorting (returns 0 if not available)."""
    try:
        impact_factor = result.get("paper_metadata", {}).get("impact_factor")
        if impact_factor is not None:
            return float(impact_factor)
    except (TypeError, ValueError):
        pass
    return 0.0


def results_to_csv_rows(all_results: List[dict]) -> List[List[dict]]:
    """
    Convert extracted results to flat CSV rows grouped by abstract.
    Each molecule gets its own row. Excludes claimed_mechanisms.
    
    Returns:
        List of lists, where each inner list contains rows for one abstract.
    """
    all_rows = []
    
    for result in all_results:
        abstract_rows = []
        
        # Paper metadata
        paper = result.get("paper_metadata", {})
        title = paper.get("title", "")
        year = paper.get("year", "")
        journal = paper.get("journal", "")
        impact_factor = paper.get("impact_factor", "")
        
        # Device metrics
        device = result.get("device_metrics", {})
        pce_max = device.get("pce_max", {}).get("value", "")
        voc = device.get("voc", {}).get("value", "")
        jsc = device.get("jsc", {}).get("value", "")
        ff = device.get("ff", {}).get("value", "")
        
        # Perovskite type
        perovskite = result.get("perovskite_type", {}).get("value", "")
        
        # Stability metrics (combine into one field)
        stability_list = result.get("stability_metrics", [])
        stability_str = "; ".join([
            f"{s.get('metric_type', '')}: {s.get('value', '')} {s.get('units', '')} ({s.get('test_conditions', '')})"
            for s in stability_list if s.get('metric_type')
        ])
        
        # Molecules - one row per molecule
        molecules = result.get("molecules", [])
        if molecules:
            for mol in molecules:
                row = {
                    "title": title,
                    "year": year,
                    "journal": journal,
                    "impact_factor": impact_factor,
                    "molecule_name": mol.get("name", ""),
                    "molecule_cid": mol.get("cid", ""),
                    "molecule_smiles": mol.get("smiles", ""),
                    "molecule_type": mol.get("type", ""),
                    "functional_groups": "; ".join(mol.get("functional_groups", []) or []),
                    "role": mol.get("role", ""),
                    "interface_location": mol.get("interface_location", ""),
                    "pce_max": pce_max,
                    "voc": voc,
                    "jsc": jsc,
                    "ff": ff,
                    "perovskite_type": perovskite,
                    "stability": stability_str,
                }
                abstract_rows.append(row)
        else:
            # No molecules - still create a row for the paper
            row = {
                "title": title,
                "year": year,
                "journal": journal,
                "impact_factor": impact_factor,
                "molecule_name": "",
                "molecule_cid": "",
                "molecule_smiles": "",
                "molecule_type": "",
                "functional_groups": "",
                "role": "",
                "interface_location": "",
                "pce_max": pce_max,
                "voc": voc,
                "jsc": jsc,
                "ff": ff,
                "perovskite_type": perovskite,
                "stability": stability_str,
            }
            abstract_rows.append(row)
        
        all_rows.append(abstract_rows)
    
    return all_rows


# -----------------------
# MAIN
# -----------------------
def load_existing_results(output_file: str) -> tuple:
    """
    Load existing results from output file to enable resuming.
    
    Returns:
        Tuple of (existing_results, processed_titles set)
    """
    existing_results = []
    processed_titles = set()
    
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse JSON objects separated by empty lines
            json_blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
            for block in json_blocks:
                try:
                    result = json.loads(block)
                    existing_results.append(result)
                    # Extract title for duplicate checking
                    title = result.get("paper_metadata", {}).get("title", "")
                    if title:
                        processed_titles.add(title.lower().strip())
                except json.JSONDecodeError:
                    continue
            
            print(f"Loaded {len(existing_results)} previously processed abstracts")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    return existing_results, processed_titles


def write_result_to_file(result: dict, output_file: str, is_first: bool):
    """
    Append a single result to the JSON output file.
    
    Args:
        result: The result dictionary to write.
        output_file: Path to the output file.
        is_first: If True, don't prepend separator.
    """
    with open(output_file, "a", encoding="utf-8") as f:
        if not is_first:
            f.write("\n\n")  # Empty line separator
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    records = load_abstracts(INPUT_FILE)
    print(f"Loaded {len(records)} abstracts from WOS file\n")
    
    # Load existing results to enable resuming
    existing_results, processed_titles = load_existing_results(OUTPUT_JSON)
    
    # Track counts
    existing_count = len(existing_results)
    success_count = existing_count
    error_count = 0
    skipped_count = 0
    
    # Process abstracts with progress bar
    pbar = tqdm(records, desc="Extracting", unit="abstract")
    for idx, record in enumerate(pbar):
        title = record['title']
        journal = record['journal']
        abstract = record['abstract']
        
        # Check if already processed (by title)
        if title and title.lower().strip() in processed_titles:
            skipped_count += 1
            pbar.set_postfix_str(f"[SKIP] {title[:25]}..." if len(title) > 25 else f"[SKIP] {title}")
            continue
        
        # Update progress bar description
        short_title = (title[:30] + "...") if title and len(title) > 30 else (title or "No title")
        pbar.set_postfix_str(f"{short_title}")
        
        try:
            # Extract with GPT
            result = call_gpt(abstract)
            
            # Override title and journal with WOS source (more reliable)
            result.setdefault("paper_metadata", {})
            if title:
                result["paper_metadata"]["title"] = title
            if journal:
                result["paper_metadata"]["journal"] = journal
            
            # Enrich with external data (impact factor, CIDs)
            result = enrich_with_external_data(result)
            
            # Write result to file immediately (on-the-fly)
            is_first = (success_count == 0)
            write_result_to_file(result, OUTPUT_JSON, is_first)
            
            # Add to processed titles
            if title:
                processed_titles.add(title.lower().strip())
            
            success_count += 1
            tqdm.write(f"[OK] Saved: {title[:50]}..." if title and len(title) > 50 else f"[OK] Saved: {title}")
        except Exception as e:
            tqdm.write(f"[ERROR] Abstract {idx+1}: {e}")
            error_count += 1
        
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    new_count = success_count - existing_count
    print(f"\nExtraction complete: {new_count} new, {skipped_count} skipped, {error_count} errors")
    print(f"Total results: {success_count}")
    print(f"Results saved to: {OUTPUT_JSON}")
    
    # Reload all results for sorting and CSV output
    print(f"\nReloading results for sorting...")
    all_results, _ = load_existing_results(OUTPUT_JSON)
    
    # Sort results by impact factor (high to low)
    print(f"Sorting {len(all_results)} results by impact factor (high to low)...")
    all_results.sort(key=get_impact_factor_for_sorting, reverse=True)
    
    # Print sorted order
    print("\nSorted order:")
    for i, result in enumerate(all_results):
        impact_factor = result.get("paper_metadata", {}).get("impact_factor", "N/A")
        journal = result.get("paper_metadata", {}).get("journal", "Unknown")
        title = result.get("paper_metadata", {}).get("title", "Unknown")[:40]
        print(f"  {i+1}. {title}... - {journal} (IF: {impact_factor})")
    
    # Write sorted results to JSON file
    print(f"\nWriting sorted JSON results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
        for i, result in enumerate(all_results):
            fout.write(json.dumps(result, ensure_ascii=False, indent=2))
            if i < len(all_results) - 1:
                fout.write("\n\n")  # Empty line between abstracts
    
    # Write CSV table (excludes claimed_mechanisms)
    print(f"Writing CSV results to {OUTPUT_CSV}...")
    csv_rows_grouped = results_to_csv_rows(all_results)
    total_rows = 0
    if csv_rows_grouped and csv_rows_grouped[0]:
        fieldnames = csv_rows_grouped[0][0].keys()
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for i, abstract_rows in enumerate(csv_rows_grouped):
                writer.writerows(abstract_rows)
                total_rows += len(abstract_rows)
                # Add empty row between abstracts (except after the last one)
                if i < len(csv_rows_grouped) - 1:
                    writer.writerow({field: "" for field in fieldnames})
    
    print(f"\nDone!")
    print(f"  JSON: {len(all_results)} abstracts written to {OUTPUT_JSON}")
    print(f"  CSV: {total_rows} rows written to {OUTPUT_CSV}")
    print("Results are sorted by impact factor (highest first)")


if __name__ == "__main__":
    main()
