import json
import time
from typing import List, Optional, Dict
from openai import OpenAI
import requests

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "gpt-4.1-mini"  # or gpt-4.1 / gpt-4o / gpt-5-mini
INPUT_FILE = "abstracts.txt"
OUTPUT_FILE = "extracted_results.jsonl"
DELIMITER = "### PAPER"
SLEEP_BETWEEN_CALLS = 1.0  # seconds (rate limit safety)
PUBCHEM_API_TIMEOUT = 5.0  # seconds

client = OpenAI()

# -----------------------
# PUBCHEM CID LOOKUP
# -----------------------
def get_pubchem_cid(molecule_name: str) -> Optional[int]:
    """
    Look up PubChem Compound ID (CID) for a molecule by name.
    
    Args:
        molecule_name: Name of the molecule (can be IUPAC name, common name, or synonym).
        
    Returns:
        CID as integer if found, None otherwise.
    """
    if not molecule_name or molecule_name.lower() == "null":
        return None
    
    # Clean up the name
    name = molecule_name.strip()
    
    try:
        # Use PubChem PUG REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
        response = requests.get(url, timeout=PUBCHEM_API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]  # Return first matching CID
        return None
    except Exception:
        return None


def get_pubchem_cid_batch(molecule_names: List[str]) -> Dict[str, Optional[int]]:
    """
    Look up PubChem CIDs for multiple molecules.
    
    Args:
        molecule_names: List of molecule names.
        
    Returns:
        Dictionary mapping molecule names to CIDs (or None if not found).
    """
    results = {}
    for name in molecule_names:
        if name:
            results[name] = get_pubchem_cid(name)
            time.sleep(0.2)  # Rate limiting for PubChem API
    return results


# -----------------------
# JOURNAL IMPACT FACTOR LOOKUP
# -----------------------
# Journals in materials science, chemistry, and energy research with approximate impact factors (2023-2024)
# Note: Impact factors change yearly. Update as needed.
# Ordered by impact factor (high to low) for reference.
JOURNAL_IMPACT_FACTORS = {
    # Ultra-high impact (IF > 50)
    "NATURE": 64.8,
    "CHEMICAL REVIEWS": 62.1,
    "SCIENCE": 56.9,
    "NATURE ENERGY": 56.7,
    "JOULE": 46.0,
    "CHEMICAL SOCIETY REVIEWS": 46.2,
    "NATURE MATERIALS": 41.2,
    # High impact (IF 20-50)
    "NATURE NANOTECHNOLOGY": 38.3,
    "NATURE CHEMISTRY": 24.4,
    "ENERGY & ENVIRONMENTAL SCIENCE": 32.5,
    "ADVANCED MATERIALS": 29.4,
    "ADVANCED ENERGY MATERIALS": 27.8,
    "CHEM": 23.5,
    "ACS ENERGY LETTERS": 22.0,
    "MATERIALS TODAY": 21.1,
    "MATTER": 19.7,
    "ADVANCED FUNCTIONAL MATERIALS": 19.0,
    # Medium-high impact (IF 15-20)
    "NANO ENERGY": 16.8,
    "NATURE COMMUNICATIONS": 16.6,
    "ANGEWANDTE CHEMIE INTERNATIONAL EDITION": 16.6,
    "ANGEWANDTE CHEMIE": 16.6,
    "ACS NANO": 15.8,
    "JOURNAL OF THE AMERICAN CHEMICAL SOCIETY": 15.0,
    # Medium impact (IF 10-15)
    "ADVANCED SCIENCE": 14.3,
    "SMALL": 13.3,
    "JOURNAL OF MATERIALS CHEMISTRY A": 11.9,
    "NANO LETTERS": 10.8,
    "ACS CATALYSIS": 12.9,
    "ACS APPLIED MATERIALS & INTERFACES": 9.5,
    "CELL REPORTS PHYSICAL SCIENCE": 8.9,
    # Standard impact (IF 5-10)
    "JOURNAL OF POWER SOURCES": 8.1,
    "MATERIALS HORIZONS": 12.2,
    "MATERIALS TODAY ENERGY": 7.4,
    "CHEMISTRY OF MATERIALS": 7.2,
    "SOLAR ENERGY MATERIALS AND SOLAR CELLS": 7.1,
    "SOLAR RRL": 6.0,
    "ISCIENCE": 5.8,
    "SUSTAINABLE ENERGY & FUELS": 5.8,
    "ELECTROCHIMICA ACTA": 5.5,
    "JOURNAL OF MATERIALS CHEMISTRY C": 6.4,
    "JOURNAL OF MATERIALS CHEMISTRY B": 6.3,
    "JOURNAL OF PHYSICAL CHEMISTRY LETTERS": 4.8,
    "JOURNAL OF PHYSICAL CHEMISTRY C": 3.7,
    "ACS APPLIED ENERGY MATERIALS": 6.4,
    "ACS SUSTAINABLE CHEMISTRY & ENGINEERING": 8.4,
    "INORGANIC CHEMISTRY": 4.6,
    "CHEMSUSCHEM": 8.4,
    "GREEN CHEMISTRY": 11.0,
    "NANOSCALE": 6.7,
    "CHEMICAL ENGINEERING JOURNAL": 15.1,
    "APPLIED CATALYSIS B ENVIRONMENTAL": 22.1,
    "JOURNAL OF COLLOID AND INTERFACE SCIENCE": 9.9,
    "APPLIED SURFACE SCIENCE": 6.7,
    "CARBON": 10.9,
    "CARBON ENERGY": 20.5,
    "ADVANCED OPTICAL MATERIALS": 9.0,
    "ADVANCED ELECTRONIC MATERIALS": 6.2,
    "SMALL METHODS": 12.4,
    "SMALL STRUCTURES": 12.0,
    "INFOMAT": 22.7,
    "ENERGY STORAGE MATERIALS": 20.4,
    "BATTERIES & SUPERCAPS": 5.3,
    "JOURNAL OF ENERGY CHEMISTRY": 14.0,
    "SCIENCE ADVANCES": 13.6,
    "SCIENCE BULLETIN": 18.9,
    "NATIONAL SCIENCE REVIEW": 20.6,
    "RESEARCH": 11.0,
    "ECOMAT": 11.8,
    "MATERIALS TODAY PHYSICS": 11.5,
    "MATERIALS TODAY ADVANCES": 8.1,
    "MATERIALS TODAY CHEMISTRY": 7.3,
    "MATERIALS TODAY SUSTAINABILITY": 7.1,
    "ADVANCED MATERIALS INTERFACES": 5.4,
    "ADVANCED MATERIALS TECHNOLOGIES": 7.0,
    "PHYSICA STATUS SOLIDI RAPID RESEARCH LETTERS": 2.5,
    "PHYSICA STATUS SOLIDI A": 2.0,
    "PHYSICA STATUS SOLIDI B": 1.8,
    "THIN SOLID FILMS": 2.1,
    "SURFACE AND COATINGS TECHNOLOGY": 5.4,
    "JOURNAL OF ALLOYS AND COMPOUNDS": 6.2,
    "MATERIALS LETTERS": 3.0,
    "MATERIALS RESEARCH BULLETIN": 5.4,
    "MATERIALS SCIENCE AND ENGINEERING B": 3.6,
    "JOURNAL OF THE ELECTROCHEMICAL SOCIETY": 3.4,
    "ELECTROCHEMISTRY COMMUNICATIONS": 4.1,
    "PHYSICAL CHEMISTRY CHEMICAL PHYSICS": 3.3,
    "DALTON TRANSACTIONS": 4.0,
    "NEW JOURNAL OF CHEMISTRY": 3.3,
    "CRYSTENGCOMM": 3.1,
    "CRYSTAL GROWTH & DESIGN": 3.8,
    "JOURNAL OF CRYSTAL GROWTH": 1.8,
    # Lower impact (IF < 5)
    "RSC ADVANCES": 3.9,
    "MATERIALS CHEMISTRY AND PHYSICS": 4.6,
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
    "MATERIALS": 3.4,
    "NANOMATERIALS": 5.3,
    "POLYMERS": 5.0,
    "CATALYSTS": 3.9,
    "ENERGIES": 3.2,
    "CHEMELECTROCHEM": 4.2,
    "CHEMPHOTOCHEM": 3.9,
    "CHEMNANOMAT": 3.8,
    "CHEMPHYSCHEM": 3.0,
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
{
  "paper_metadata": {
    "year": null,
    "journal": null,
    "impact_factor": null
  },
  "molecules": [
    {
      "name": null,
      "cid": null,
      "type": null,
      "functional_groups": [],
      "role": null,
      "interface_location": null,
      "evidence": null
    }
  ],
  "device_metrics": {
    "pce_max": { "value": null, "units": "%", "evidence": null },
    "voc": { "value": null, "units": "V", "evidence": null },
    "jsc": { "value": null, "units": "mA/cm2", "evidence": null },
    "ff": { "value": null, "units": "%", "evidence": null }
  },
  "stability_metrics": [
    {
      "metric_type": null,
      "value": null,
      "units": null,
      "test_conditions": null,
      "evidence": null
    }
  ],
  "perovskite_type": { "value": null, "evidence": null },
  "claimed_mechanisms": [
    { "mechanism": null, "evidence": null }
  ]
}

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
def load_abstracts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split(DELIMITER) if c.strip()]
    return chunks


def call_gpt(abstract: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(abstract=abstract)}
        ],
    )
    return json.loads(response.choices[0].message.content)


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
    
    # Add PubChem CIDs for molecules
    molecules = result.get("molecules", [])
    if molecules:
        for mol in molecules:
            name = mol.get("name")
            if name and name.lower() != "null":
                cid = get_pubchem_cid(name)
                mol["cid"] = cid
                if cid:
                    print(f"    PubChem CID for '{name}': {cid}")
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


# -----------------------
# MAIN
# -----------------------
def main():
    abstracts = load_abstracts(INPUT_FILE)
    print(f"Loaded {len(abstracts)} abstracts")
    
    # Collect all results
    all_results = []
    
    for idx, abstract in enumerate(abstracts):
        try:
            # Extract with GPT
            result = call_gpt(abstract)
            print(f"[OK] Abstract {idx+1}/{len(abstracts)} - GPT extraction complete")
            
            # Enrich with external data (impact factor, CIDs)
            result = enrich_with_external_data(result)
            
            # Store result with original index for reference
            result["_original_index"] = idx + 1
            all_results.append(result)
            print(f"[OK] Abstract {idx+1}/{len(abstracts)} - Enrichment complete")
        except Exception as e:
            print(f"[ERROR] Abstract {idx+1}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    # Sort results by impact factor (high to low)
    print(f"\nSorting {len(all_results)} results by impact factor (high to low)...")
    all_results.sort(key=get_impact_factor_for_sorting, reverse=True)
    
    # Write sorted results to output file
    print(f"Writing sorted results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for result in all_results:
            # Remove temporary index field before writing
            original_idx = result.pop("_original_index", None)
            impact_factor = result.get("paper_metadata", {}).get("impact_factor", "N/A")
            journal = result.get("paper_metadata", {}).get("journal", "Unknown")
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"  Written: Abstract {original_idx} - {journal} (IF: {impact_factor})")
    
    print(f"\nDone! {len(all_results)} results written to {OUTPUT_FILE}")
    print("Results are sorted by impact factor (highest first)")


if __name__ == "__main__":
    main()
