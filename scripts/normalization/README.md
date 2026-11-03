# Entity normalization

## PubChem matching

```bash
python scripts/normalization/pubchem_matching.py --input-dir data/inference --output-dir results/normalization
```

Input JSONL records contain a graph (`nodes`, `relations`) or a `prediction` field
with graph JSON. The matcher processes Fuel names and aliases through PubChem
PUG REST. Single-compound responses establish CID matches; aliases are grouped
by CID agreement, with separate groups for conflicting identities. Responses are
cached in `pubchem_cache.json`.

Outputs:

- `fuel_dictionary_base.json`: canonical names and accepted synonyms.
- `fuel_candidates.json`: mentions selected for contextual resolution.
- `pubchem_cache.json`: cached name queries.

## Contextual resolution

```bash
python scripts/normalization/llm_name_resolution.py --model /path/to/model --xml-dir data/xml --dictionary-dir results/normalization
```

The script retrieves source sentences surrounding each candidate name and asks
a local language model for `normalized_name`, `confidence` and `uncertain`.
High-confidence decisions with `uncertain=false` enter the dictionary update,
subject to CID consistency and abbreviation-scope rules.

Outputs are `fuel_judgments.json` and `fuel_dictionary.json`. Commercial fuels
and formulations are resolved at fuel-concept level. The model environment uses
`scripts/training/requirements.txt`.

## Apply the dictionary

```bash
python scripts/normalization/clean_extractions.py inference_000.jsonl --input_dir data/inference --output_dir results/cleaned --fuel_dictionary results/normalization/fuel_dictionary.json
python scripts/normalization/normalize_extractions.py clean_000.jsonl --input_dir results/cleaned --output_dir results/normalized --fuel_dictionary results/normalization/fuel_dictionary.json
```

The scripts normalize chemical names, aliases, condition types, numerical notation
and units. Recipe-aware and sample-aware identity rules are implemented in
`scripts/data_preparation/canonical_graph.py`.
