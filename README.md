# Fuel Knowledge Graph

Data and code for fuel knowledge extraction, entity normalization, graph construction,
fuel-property prediction and flame-speed comparison.

## Repository structure

```text
fuel-knowledge-graph/
├── data/
│   ├── neo4j/                 # Canonical graph CSV and source evidence
│   ├── prediction/            # Fuel-property training inputs
│   └── validation/            # Flame-speed experiments and reference curves
├── rdf/                       # RDF schema, graph archive and export metadata
├── scripts/
│   ├── data_preparation/      # Section selection, entity merging and RDF export
│   ├── normalization/         # PubChem matching and contextual name resolution
│   ├── training/              # QLoRA training and configuration
│   └── prediction/
│       └── property/          # Nested recipe-group cross-validation models
├── requirements.txt
├── LICENSE
└── README.md
```

## Graph data

The CSV, RDF and web explorer share a canonical graph with **46,133 nodes and
95,914 relationships**. Experiment nodes represent 13 shared experimental categories.

| Node type | Count |
|---|---:|
| Fuel | 1,742 |
| Mixture | 11,927 |
| RealFuel | 732 |
| FuelSample | 435 |
| Experiment | 13 |
| Condition | 12,894 |
| ExpResult | 15,474 |
| Property | 2,916 |

- [Neo4j CSV and import instructions](data/neo4j/README.md)
- [RDFS/OWL schema](rdf/fkg_schema.ttl)
- [RDF archive](rdf/FKG_RDF.zip)
- [RDF representation](rdf/README.md)

Nodes retain source-record identifiers, attribute variants and DOI lists.
Relationship evidence covers 118,631 source assertions, stored in CSV and RDF.
Source DOI and endpoint identifiers support article-specific queries through the
shared experiment categories.

Build the canonical CSV and RDF graph from extraction tables:

```bash
python scripts/data_preparation/export_graph.py --data-dir /path/to/extraction_tables
python scripts/data_preparation/export_rdf.py
```

## Setup

Use Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
```

For QLoRA training and local language-model inference:

```bash
python -m pip install -r scripts/training/requirements.txt
```

## Extraction and normalization

Select article sections using the keyword rules associated with SI Table S3:

```bash
python scripts/data_preparation/filter_sections.py --input data/xml --output results/selected_text
```

Train with chat-formatted JSONL records containing a `messages` field. Training
parameters are listed in `scripts/training/qlora_settings.json`.

```bash
python scripts/training/train_qlora.py --model /path/to/model --data-dir data/training --output-dir results/adapter
```

Build and apply the fuel dictionary using
[PubChem matching and contextual resolution](scripts/normalization/README.md):

```bash
python scripts/normalization/pubchem_matching.py --input-dir data/inference --output-dir results/normalization
python scripts/normalization/llm_name_resolution.py --model /path/to/model --xml-dir data/xml --dictionary-dir results/normalization
python scripts/normalization/clean_extractions.py inference_000.jsonl --input_dir data/inference --output_dir results/cleaned --fuel_dictionary results/normalization/fuel_dictionary.json
python scripts/normalization/normalize_extractions.py clean_000.jsonl --input_dir results/cleaned --output_dir results/normalized --fuel_dictionary results/normalization/fuel_dictionary.json
```

## Prediction

The property dataset contains 464 complete-recipe records and 95 auxiliary records.
[Prediction methods](scripts/prediction/README.md) describe the inputs,
features and model selection.

Configure descriptor and source-graph paths, then run the property models:

```bash
export FKG_FEATURE_DIR=/path/to/component_descriptors
export FKG_GRAPH_RECORD_DIR=/path/to/source_graph_records
python scripts/prediction/property/run_models.py
```

Prediction outputs are written to `results/`.

## Flame-speed comparison data

[flame_speed_figure_data.csv](data/validation/flame_speed_figure_data.csv) contains
CH4/air and H2/air experimental points and supplied GRI-Mech 3.0 reference curves.
Columns are `dataset`, `equivalence_ratio`, `flame_speed_cm_s` and `doi`.
Dataset labels distinguish experiments and reference curves; DOI is source metadata.
See [data notes](data/validation/README.md) for row counts and scope.
