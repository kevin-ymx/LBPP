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
INPUT_FILE = "abstract_mol_4209.txt"  # WOS export format (SO=journal, AB=abstract, ER=end record)
OUTPUT_JSON = "extracted_results_mol_4209.json"  # JSON output sorted by impact factor (high to low)
OUTPUT_CSV = "extracted_results_mol_4209.csv"  # CSV table output (excludes claimed_mechanisms)
SLEEP_BETWEEN_CALLS = 0.1  # seconds (rate limit safety)
PUBCHEM_API_TIMEOUT = 10.0  # seconds

# Initialize OpenAI client
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Set environment variable or configure in script (line 16).")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# PUBCHEM CID AND SMILES LOOKUP
# -----------------------
# Max length of parenthetical content to treat as abbreviation (strip for CID lookup only).
_ABBR_PAREN_MAX_LEN = 20


def _name_for_cid_lookup(molecule_name: str) -> str:
    """
    For CID lookup: strip only a trailing parenthetical that looks like an abbreviation
    (space before '(', content short and no comma). Do not strip descriptive parentheticals
    e.g. "(1,2-dihydro)" or "(2F)" at end with space before paren -> strip "(2F)";
    "name (1,2-dihydro)" -> keep full name.
    """
    name = molecule_name.strip()
    if not name:
        return name
    # Trailing " (X)" with a space before the opening paren
    m = re.match(r"^(.+)\s+\(([^)]*)\)\s*$", name)
    if m:
        prefix, in_paren = m.group(1).rstrip(), m.group(2)
        if "," not in in_paren and len(in_paren) <= _ABBR_PAREN_MAX_LEN:
            return prefix
    return name


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
    
    name = _name_for_cid_lookup(molecule_name.strip())
    
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
# Merged from journal_summary + JCR and original non-zero; ordered by IF (high to low).
JOURNAL_IMPACT_FACTORS = {
"NATURE REVIEWS MATERIALS": 86.2,
"NATURE ENERGY": 60.1,
"NATURE REVIEWS CHEMISTRY": 51.7,
"NATURE": 48.5,
"SCIENCE": 45.8,
"NATURE ELECTRONICS": 40.9,
"NATURE MATERIALS": 38.5,
"NATURE NANOTECHNOLOGY": 37.2,
"NATURE PHOTONICS": 32.9,
"NATURE CHEMISTRY": 20.2,
"NATURE SYNTHESIS": 20.0,
"NATURE COMMUNICATIONS": 15.7,
"SCIENCE ADVANCES": 13.6,
"CHEMICAL REVIEWS": 55.8,
"NANO-MICRO LETTERS": 36.3,
"JOULE": 35.4,
"INTERDISCIPLINARY MATERIALS": 31.6,
"ADVANCED MATERIALS": 30.2,
"ENERGY & ENVIRONMENTAL SCIENCE": 29.1,
"ADVANCED ENERGY MATERIALS": 26.9,
"ADVANCES IN OPTICS AND PHOTONICS": 23.8,
"CHEM": 23.5,
"LIGHT-SCIENCE & APPLICATIONS": 23.4,
"INFOMAT": 22.7,
"OPTO-ELECTRONIC ADVANCES": 22.4,
"ACS ENERGY LETTERS": 22.0,
"ADVANCED COMPOSITES AND HYBRID MATERIALS": 21.8,
"ADVANCED FIBER MATERIALS": 21.3,
"APPLIED CATALYSIS B ENVIRONMENTAL": 21.1,
"MATERIALS TODAY": 21.1,
"NATIONAL SCIENCE REVIEW": 20.6,
"CARBON ENERGY": 20.5,
"ENERGY STORAGE MATERIALS": 20.4,
"MATTER": 19.7,
"ADVANCED FUNCTIONAL MATERIALS": 19.5,
"SCIENCE BULLETIN": 18.9,
"NANO ENERGY": 16.8,
"ANGEWANDTE CHEMIE": 16.1,
"ANGEWANDTE CHEMIE-INTERNATIONAL EDITION": 16.1,
"ACS NANO": 16.0,
"ENERGY MATERIAL ADVANCES": 15.9,
"JOURNAL OF THE AMERICAN CHEMICAL SOCIETY": 15.6,
"CHEMICAL ENGINEERING JOURNAL": 15.1,
"ACCOUNTS OF MATERIALS RESEARCH": 14.7,
"ADVANCED SCIENCE": 14.3,
"JOURNAL OF MATERIALS SCIENCE & TECHNOLOGY": 14.3,
"ENERGY & ENVIRONMENTAL MATERIALS": 14.1,
"JOURNAL OF ENERGY CHEMISTRY": 14.0,
"ACTA PHYSICO-CHIMICA SINICA": 13.7,
"SMALL": 13.3,
"CARBON NEUTRALITY": 12.5,
"SMALL METHODS": 12.4,
"CARBON NEUTRALIZATION": 12.0,
"SMALL STRUCTURES": 12.0,
"JOURNAL OF MATERIALS CHEMISTRY A": 11.9,
"NPJ COMPUTATIONAL MATERIALS": 11.9,
"ECOMAT": 11.8,
"ACS CATALYSIS": 11.7,
"MATERIALS TODAY PHYSICS": 11.5,
"ENERGY MATERIALS": 11.2,
"GREEN CHEMISTRY": 11.0,
"RARE METALS": 11.0,
"RESEARCH": 11.0,
"CARBON": 10.9,
"MATERIALS HORIZONS": 10.7,
"BIOSENSORS & BIOELECTRONICS": 10.5,
"CHINESE JOURNAL OF STRUCTURAL CHEMISTRY": 10.3,
"JOURNAL OF COLLOID AND INTERFACE SCIENCE": 9.9,
"SCIENCE CHINA-CHEMISTRY": 9.7,
"COMMUNICATIONS MATERIALS": 9.6,
"CCS CHEMISTRY": 9.2,
"NANO LETTERS": 9.1,
"RENEWABLE ENERGY": 9.1,
"ADVANCED OPTICAL MATERIALS": 9.0,
"NANO RESEARCH": 9.0,
"CELL REPORTS PHYSICAL SCIENCE": 8.9,
"CHINESE CHEMICAL LETTERS": 8.9,
"ACS MATERIALS LETTERS": 8.7,
"JACS AU": 8.7,
"MATERIALS TODAY ENERGY": 8.6,
"TRANSACTIONS OF TIANJIN UNIVERSITY": 8.5,
"ACS SUSTAINABLE CHEMISTRY & ENGINEERING": 8.4,
"CHEMSUSCHEM": 8.4,
"INTERNATIONAL JOURNAL OF HYDROGEN ENERGY": 8.3,
"SMALL SCIENCE": 8.3,
"ACS APPLIED MATERIALS & INTERFACES": 8.2,
"JOURNAL OF POWER SOURCES": 8.1,
"MATERIALS TODAY ADVANCES": 8.1,
"MATERIALS & DESIGN": 7.9,
"RESULTS IN ENGINEERING": 7.9,
"SENSORS AND ACTUATORS B-CHEMICAL": 7.7,
"PROGRESS IN PHOTOVOLTAICS": 7.6,
"CHEMICAL RECORD": 7.5,
"CHEMICAL SCIENCE": 7.4,
"SCIENCE CHINA-MATERIALS": 7.4,
"MATERIALS TODAY CHEMISTRY": 7.3,
"INTERNATIONAL JOURNAL OF MINERALS METALLURGY AND MATERIALS": 7.2,
"MATERIALS TODAY SUSTAINABILITY": 7.1,
"ADVANCED MATERIALS TECHNOLOGIES": 7.0,
"CHEMISTRY OF MATERIALS": 7.0,
"APPLIED MATERIALS TODAY": 6.9,
"ACS PHOTONICS": 6.7,
"APPLIED SURFACE SCIENCE": 6.7,
"NANOSCALE": 6.7,
"NANOPHOTONICS": 6.6,
"NANOSCALE HORIZONS": 6.6,
"SOLAR ENERGY": 6.6,
"ACS APPLIED ENERGY MATERIALS": 6.4,
"INORGANIC CHEMISTRY FRONTIERS": 6.4,
"JOURNAL OF MATERIALS CHEMISTRY C": 6.4,
"MATERIALS CHEMISTRY FRONTIERS": 6.4,
"JOURNAL OF PHYSICS-ENERGY": 6.3,
"SOLAR ENERGY MATERIALS AND SOLAR CELLS": 6.3,
"SURFACES AND INTERFACES": 6.3,
"ADVANCED ELECTRONIC MATERIALS": 6.2,
"JOURNAL OF ALLOYS AND COMPOUNDS": 6.2,
"ADVANCED SUSTAINABLE SYSTEMS": 6.1,
"JOURNAL OF INDUSTRIAL AND ENGINEERING CHEMISTRY": 6.0,
"SOLAR RRL": 6.0,
"ISCIENCE": 5.8,
"JOURNAL OF MATERIALS CHEMISTRY B": 5.8,
"SUSTAINABLE ENERGY & FUELS": 5.8,
"ADVANCED ENERGY AND SUSTAINABILITY RESEARCH": 5.7,
"CERAMICS INTERNATIONAL": 5.6,
"DIGITAL DISCOVERY": 5.6,
"ACS APPLIED NANO MATERIALS": 5.5,
"CHINESE JOURNAL OF CHEMISTRY": 5.5,
"ELECTROCHIMICA ACTA": 5.5,
"FRONTIERS IN CHEMISTRY": 5.5,
"MATERIALS FOR RENEWABLE AND SUSTAINABLE ENERGY": 5.5,
"ADVANCED MATERIALS INTERFACES": 5.4,
"COLLOIDS AND SURFACES A-PHYSICOCHEMICAL AND ENGINEERING ASPECTS": 5.4,
"INORGANIC CHEMISTRY COMMUNICATIONS": 5.4,
"MATERIALS RESEARCH BULLETIN": 5.4,
"SURFACE AND COATINGS TECHNOLOGY": 5.4,
"BATTERIES & SUPERCAPS": 5.3,
"ENERGY & FUELS": 5.3,
"JOURNAL OF SEMICONDUCTORS": 5.3,
"NANOMATERIALS": 5.3,
"FRONTIERS OF OPTOELECTRONICS": 5.2,
"JOURNAL OF MOLECULAR LIQUIDS": 5.2,
"MACROMOLECULES": 5.2,
"ENERGY REPORTS": 5.1,
"ORGANIC LETTERS": 5.0,
"POLYMERS": 5.0,
"INTERNATIONAL JOURNAL OF MOLECULAR SCIENCES": 4.9,
"JOURNAL OF PHYSICS AND CHEMISTRY OF SOLIDS": 4.9,
"ACS APPLIED ELECTRONIC MATERIALS": 4.7,
"JOURNAL OF MOLECULAR STRUCTURE": 4.7,
"JOURNAL OF PHOTOCHEMISTRY AND PHOTOBIOLOGY A-CHEMISTRY": 4.7,
"MATERIALS ADVANCES": 4.7,
"MATERIALS CHEMISTRY AND PHYSICS": 4.7,
"INORGANIC CHEMISTRY": 4.6,
"JOURNAL OF PHYSICAL CHEMISTRY LETTERS": 4.6,
"MATERIALS SCIENCE AND ENGINEERING B-ADVANCED FUNCTIONAL SOLID-STATE MATERIALS": 4.6,
"MATERIALS SCIENCE IN SEMICONDUCTOR PROCESSING": 4.6,
"MOLECULES": 4.6,
"RSC ADVANCES": 4.6,
"SCIENTIFIC REPORTS": 4.6,
"SPECTROCHIMICA ACTA PART A-MOLECULAR AND BIOMOLECULAR SPECTROSCOPY": 4.6,
"SYNTHETIC METALS": 4.6,
"IEEE ELECTRON DEVICE LETTERS": 4.5,
"MATERIALS TODAY COMMUNICATIONS": 4.5,
"PHYSICAL REVIEW APPLIED": 4.4,
"ACS OMEGA": 4.3,
"CHEMICAL ENGINEERING SCIENCE": 4.3,
"CHEMICAL PHYSICS IMPACT": 4.3,
"ENERGY ADVANCES": 4.3,
"CHEMELECTROCHEM": 4.2,
"CHEMICAL COMMUNICATIONS": 4.2,
"DYES AND PIGMENTS": 4.2,
"INTERNATIONAL JOURNAL OF ENERGY RESEARCH": 4.2,
"OPTICAL MATERIALS": 4.2,
"RESULTS IN CHEMISTRY": 4.2,
"ELECTROCHEMISTRY COMMUNICATIONS": 4.1,
"EMERGENT MATERIALS": 4.1,
"JOURNAL OF ELECTROANALYTICAL CHEMISTRY": 4.1,
"NANOSCALE RESEARCH LETTERS": 4.1,
"CHINESE JOURNAL OF POLYMER SCIENCE": 4.0,
"DALTON TRANSACTIONS": 4.0,
"OPTICAL AND QUANTUM ELECTRONICS": 4.0,
"POLYMER CHEMISTRY": 4.0,
"CATALYSTS": 3.9,
"CHEMPHOTOCHEM": 3.9,
"JOURNAL OF MATERIALS SCIENCE": 3.9,
"LANGMUIR": 3.9,
"VACUUM": 3.9,
"CHEMNANOMAT": 3.8,
"CRYSTAL GROWTH & DESIGN": 3.8,
"ENERGY TECHNOLOGY": 3.8,
"CHEMISTRY-A EUROPEAN JOURNAL": 3.7,
"JOURNAL OF PHYSICAL CHEMISTRY C": 3.7,
"PHYSICAL REVIEW B": 3.7,
"PLOS ONE": 3.7,
"JOURNAL OF LUMINESCENCE": 3.6,
"JOURNAL OF ORGANIC CHEMISTRY": 3.6,
"APPLIED PHYSICS LETTERS": 3.5,
"JOURNAL OF SOLID STATE CHEMISTRY": 3.5,
"NANO SELECT": 3.5,
"RESEARCH ON CHEMICAL INTERMEDIATES": 3.5,
"JOURNAL OF PHYSICS D APPLIED PHYSICS": 3.4,
"JOURNAL OF THE ELECTROCHEMICAL SOCIETY": 3.4,
"MACROMOLECULAR RESEARCH": 3.4,
"PHYSICAL REVIEW MATERIALS": 3.4,
"CHEMISTRY-AN ASIAN JOURNAL": 3.3,
"COMPUTATIONAL MATERIALS SCIENCE": 3.3,
"NEW JOURNAL OF CHEMISTRY": 3.3,
"PHYSICAL CHEMISTRY CHEMICAL PHYSICS": 3.3,
"SILICON": 3.3,
"ENERGIES": 3.2,
"FRONTIERS IN MATERIALS": 3.2,
"IEEE TRANSACTIONS ON ELECTRON DEVICES": 3.2,
"JOURNAL OF PHYSICS D-APPLIED PHYSICS": 3.2,
"JOURNAL OF THE EUROPEAN OPTICAL SOCIETY - RAPID PUBLICATIONS": 3.2,
"KOREAN JOURNAL OF CHEMICAL ENGINEERING": 3.2,
"MATERIALS": 3.2,
"MOLECULAR SYSTEMS DESIGN & ENGINEERING": 3.2,
"PHOTOCHEMICAL & PHOTOBIOLOGICAL SCIENCES": 3.2,
"CHEMICAL PHYSICS LETTERS": 3.1,
"CURRENT APPLIED PHYSICS": 3.1,
"JOURNAL OF CHEMICAL PHYSICS": 3.1,
"JOURNAL OF FLUORESCENCE": 3.1,
"SOLID STATE SCIENCES": 3.1,
"CHEMPHYSCHEM": 3.0,
"JOURNAL OF MOLECULAR GRAPHICS & MODELLING": 3.0,
"MATERIALS LETTERS": 3.0,
"MICROMACHINES": 3.0,
"EUROPEAN PHYSICAL JOURNAL PLUS": 2.9,
"JOURNAL OF APPLIED PHYSICS": 2.9,
"JOURNAL OF PHYSICAL CHEMISTRY B": 2.9,
"ADVANCED PHYSICS RESEARCH": 2.8,
"APPLIED PHYSICS A-MATERIALS SCIENCE & PROCESSING": 2.8,
"CHEMPLUSCHEM": 2.8,
"COATINGS": 2.8,
"JOURNAL OF MATERIALS SCIENCE-MATERIALS IN ELECTRONICS": 2.8,
"JOURNAL OF PHYSICAL CHEMISTRY A": 2.8,
"NANOTECHNOLOGY": 2.8,
"ORGANIC ELECTRONICS": 2.8,
"PHYSICA B-CONDENSED MATTER": 2.8,
"ASIAN JOURNAL OF ORGANIC CHEMISTRY": 2.7,
"BEILSTEIN JOURNAL OF NANOTECHNOLOGY": 2.7,
"EUROPEAN JOURNAL OF ORGANIC CHEMISTRY": 2.7,
"CRYSTENGCOMM": 2.6,
"IEEE JOURNAL OF PHOTOVOLTAICS": 2.6,
"JOURNAL OF PHYSICS-CONDENSED MATTER": 2.6,
"JOURNAL OF SOLID STATE ELECTROCHEMISTRY": 2.6,
"PHYSICA SCRIPTA": 2.6,
"APPLIED SCIENCES-BASEL": 2.5,
"JOURNAL OF COMPUTATIONAL ELECTRONICS": 2.5,
"JOURNAL OF ELECTRONIC MATERIALS": 2.5,
"JOURNAL OF MOLECULAR MODELING": 2.5,
"JOURNAL OF OPTICS-INDIA": 2.5,
"METALS": 2.5,
"PHYSICA STATUS SOLIDI-RAPID RESEARCH LETTERS": 2.5,
"CHEMICAL PHYSICS": 2.4,
"CRYSTALS": 2.4,
"MODELLING AND SIMULATION IN MATERIALS SCIENCE AND ENGINEERING": 2.4,
"SOLID STATE COMMUNICATIONS": 2.4,
"IEEE TRANSACTIONS ON DEVICE AND MATERIALS RELIABILITY": 2.3,
"INTERNATIONAL JOURNAL OF APPLIED CERAMIC TECHNOLOGY": 2.3,
"JOURNAL OF COMPUTATIONAL BIOPHYSICS AND CHEMISTRY": 2.3,
"APPLIED PHYSICS EXPRESS": 2.2,
"ECS JOURNAL OF SOLID STATE SCIENCE AND TECHNOLOGY": 2.2,
"MATERIALS RESEARCH EXPRESS": 2.2,
"STRUCTURAL CHEMISTRY": 2.2,
"TETRAHEDRON": 2.2,
"BULLETIN OF MATERIALS SCIENCE": 2.1,
"JOURNAL OF VACUUM SCIENCE & TECHNOLOGY A": 2.1,
"SEMICONDUCTOR SCIENCE AND TECHNOLOGY": 2.1,
"THIN SOLID FILMS": 2.1,
"CHEMISTRYSELECT": 2.0,
"INTERNATIONAL JOURNAL OF QUANTUM CHEMISTRY": 2.0,
"MOLECULAR SIMULATION": 2.0,
"NEXT MATERIALS": 1.9,
"HELVETICA CHIMICA ACTA": 1.8,
"JAPANESE JOURNAL OF APPLIED PHYSICS": 1.8,
"MOLECULAR PHYSICS": 1.8,
"PHYSICA STATUS SOLIDI B-BASIC SOLID STATE PHYSICS": 1.8,
"MENDELEEV COMMUNICATIONS": 1.7,
"ACTA CHIMICA SINICA": 1.6,
"AIP ADVANCES": 1.6,
"CHIMIA": 1.6,
"JOURNAL OF INORGANIC MATERIALS": 1.6,
"OPTICS": 1.6,
"CHINESE PHYSICS B": 1.5,
"CURRENT NANOSCIENCE": 1.5,
"JOURNAL OF ELECTRON SPECTROSCOPY AND RELATED PHENOMENA": 1.5,
"TETRAHEDRON LETTERS": 1.5,
"KOREAN JOURNAL OF METALS AND MATERIALS": 1.4,
"JOURNAL OF THE BRAZILIAN CHEMICAL SOCIETY": 1.3,
"PROGRESS IN CHEMISTRY": 1.2,
"SURFACE REVIEW AND LETTERS": 1.2,
"CHINA SURFACE ENGINEERING": 1.1,
"JOURNAL OF OPTOELECTRONIC AND BIOMEDICAL MATERIALS": 1.1,
"CANADIAN JOURNAL OF CHEMISTRY": 1.0,
"CHINESE JOURNAL OF CHEMICAL PHYSICS": 1.0,
"JOURNAL OF NANO RESEARCH": 1.0,
"MAIN GROUP CHEMISTRY": 1.0,
"OPTO-ELECTRONICS REVIEW": 0.9,
"ACTA PHYSICA SINICA": 0.8,
"CHINESE JOURNAL OF INORGANIC CHEMISTRY": 0.7,
"JOURNAL OF OPTOELECTRONICS AND ADVANCED MATERIALS": 0.6,
"SCIENTIA SINICA-PHYSICA MECHANICA & ASTRONOMICA": 0.5,
"ADVANCED THEORY AND SIMULATIONS": 2.9,
"CHINESE SCIENCE BULLETIN-CHINESE": 0.0,
"ENERGY MATERIALS AND DEVICES": 0.0,
"NANO RESEARCH ENERGY": 0.0,
"NEXT ENERGY": 0.0,
"OPTIK": 0.0,
"POLYOXOMETALATES": 0.0,
"RESOURCES CHEMICALS AND MATERIALS": 0.0,
}


def _is_nature_or_science_family(journal_name: str) -> bool:
    """True if journal is in the Nature or Science series (sister journals)."""
    u = journal_name.strip().upper()
    # Explicit exclusions: do NOT treat these as front-of-list sisters
    excluded = {
        "SCIENCE BULLETIN",
        "NPJ COMPUTATIONAL MATERIALS",
        "SCIENCE CHINA-CHEMISTRY",
        "COMMUNICATIONS MATERIALS",
        "SCIENCE CHINA-MATERIALS",
        "SCIENTIFIC REPORTS",
    }
    if u in excluded:
        return False
    # Nature series
    if u.startswith("NATURE"):
        return True
    if u.startswith("NPJ "):  # Nature Partner Journals (except excluded above)
        return True
    # Science (AAAS) series
    if u == "SCIENCE" or (u.startswith("SCIENCE ") and not u.startswith("SCIENCE CHINA")):
        return True
    # Science China series (except excluded above)
    if u.startswith("SCIENCE CHINA"):
        return True
    return False


def _reorder_journal_impact_factors(d: dict) -> dict:
    """Put Nature and Science series first (by IF desc), then others (by IF desc)."""
    family = [(k, v) for k, v in d.items() if _is_nature_or_science_family(k)]
    rest = [(k, v) for k, v in d.items() if not _is_nature_or_science_family(k)]
    family.sort(key=lambda x: (-x[1], x[0]))
    rest.sort(key=lambda x: (-x[1], x[0]))
    return dict(family + rest)


JOURNAL_IMPACT_FACTORS = _reorder_journal_impact_factors(JOURNAL_IMPACT_FACTORS)

# Build journal -> rank map for CSV/JSON ordering (follow JOURNAL_IMPACT_FACTORS dict order)
_JOURNAL_ORDER_LIST = list(JOURNAL_IMPACT_FACTORS.keys())
_JOURNAL_ORDER_LEN = len(_JOURNAL_ORDER_LIST)
_JOURNAL_ORDER_RANK = {}
for _i, _k in enumerate(_JOURNAL_ORDER_LIST):
    _JOURNAL_ORDER_RANK[_k] = _i
    for _v in (_k.replace(" AND ", " & "), _k.replace(" & ", " AND ")):
        if _v != _k:
            _JOURNAL_ORDER_RANK[_v] = _i


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
        if name in JOURNAL_IMPACT_FACTORS:
            return JOURNAL_IMPACT_FACTORS[name]
    
    # No partial match: short keys like "SCIENCE" or "MATERIALS" would incorrectly
    # match many journals (e.g. SCIENCE CHINA-CHEMISTRY, MATERIALS SCIENCE IN ...).
    # Only exact key match (with AND/& variants) is used; unknown journals get None.
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
    # Add journal impact factor (None if journal not in JOURNAL_IMPACT_FACTORS)
    journal = result.get("paper_metadata", {}).get("journal")
    impact_factor = get_journal_impact_factor(journal) if journal else None
    result["paper_metadata"]["impact_factor"] = impact_factor
    if impact_factor is not None:
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
    """
    Get impact factor for sorting.
    Entries with no matching journal (IF None) return 0 so they sort last (not ranked).
    """
    try:
        impact_factor = result.get("paper_metadata", {}).get("impact_factor")
        if impact_factor is not None:
            return float(impact_factor)
    except (TypeError, ValueError):
        pass
    return 0.0  # No match → sort last (do not rank)


def get_journal_order_key(result: dict) -> tuple:
    """
    Sort key so results follow JOURNAL_IMPACT_FACTORS dict order, then IF desc, then title.
    Used for CSV and JSON output order.
    """
    meta = result.get("paper_metadata") or {}
    journal = (meta.get("journal") or "").upper().strip()
    rank = _JOURNAL_ORDER_RANK.get(journal)
    if rank is None:
        rank = _JOURNAL_ORDER_RANK.get(journal.replace(" AND ", " & "))
    if rank is None:
        rank = _JOURNAL_ORDER_RANK.get(journal.replace(" & ", " AND "))
    if rank is None:
        rank = _JOURNAL_ORDER_LEN
    if_val = get_impact_factor_for_sorting(result)
    title = (meta.get("title") or "").strip()
    return (rank, -if_val, title)


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
        
        # Paper metadata (impact_factor None when journal not in JOURNAL_IMPACT_FACTORS)
        paper = result.get("paper_metadata", {})
        title = paper.get("title", "")
        year = paper.get("year", "")
        journal = paper.get("journal", "")
        impact_factor = paper.get("impact_factor")
        impact_factor = impact_factor if impact_factor is not None else ""
        
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
    
    # Re-apply impact factor lookup so existing entries get corrected IFs (exact-match only)
    print("Re-applying impact factor lookup to all results...")
    for result in all_results:
        meta = result.get("paper_metadata")
        if meta is not None:
            journal = meta.get("journal")
            meta["impact_factor"] = get_journal_impact_factor(journal)
    
    # Sort results by journal order in JOURNAL_IMPACT_FACTORS, then by IF (high to low), then title
    print(f"Sorting {len(all_results)} results by journal order (JOURNAL_IMPACT_FACTORS)...")
    all_results.sort(key=get_journal_order_key)
    
    # Print sorted order (IF None = not in JOURNAL_IMPACT_FACTORS, shown as N/A, not ranked)
    print("\nSorted order:")
    for i, result in enumerate(all_results):
        impact_factor = result.get("paper_metadata", {}).get("impact_factor")
        if impact_factor is None:
            impact_factor = "N/A"
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
    print("Results follow JOURNAL_IMPACT_FACTORS order (then IF, then title)")


if __name__ == "__main__":
    main()
