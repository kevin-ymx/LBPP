import json
import time
from typing import List
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "gpt-4.1-mini"  # or gpt-4.1 / gpt-4o / gpt-5-mini
INPUT_FILE = "abstracts.txt"
OUTPUT_FILE = "extracted_results.jsonl"
DELIMITER = "### PAPER"
SLEEP_BETWEEN_CALLS = 1.0  # seconds (rate limit safety)

client = OpenAI()

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
    "journal": null
  },
  "molecules": [
    {
      "name": null,
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


# -----------------------
# MAIN
# -----------------------
def main():
    abstracts = load_abstracts(INPUT_FILE)
    print(f"Loaded {len(abstracts)} abstracts")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for idx, abstract in enumerate(abstracts):
            try:
                result = call_gpt(abstract)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[OK] Abstract {idx+1}")
            except Exception as e:
                print(f"[ERROR] Abstract {idx+1}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
