# Neo4j import

The canonical graph contains **46,133 nodes and 95,914 relationships**, sharing
entity identities and connectivity with the web explorer.

## Import

Run an offline bulk import into an empty Neo4j 5 database with the database stopped:

```bash
neo4j-admin database import full --nodes=data/neo4j/nodes.csv --relationships=data/neo4j/relationships.csv --multiline-fields=true fkg
```

Run the command from the repository root. JSON-valued cells are string properties
that can be parsed for structured queries.

## CSV fields

| Field | Meaning |
|---|---|
| `entity_id:ID(FKG)`, `:LABEL` | Node identifier and type |
| `:START_ID(FKG)`, `:END_ID(FKG)`, `:TYPE` | Directed relationship endpoints and type |
| `record_count:int`, `source_count:int` | Source record and document counts |
| `evidence_count:int` | Supporting relationship assertion count |
| `attributes_json` | Agreed node attributes |
| `attribute_variants_json` | Distinct reported attribute values |
| `fields_json` | Shared source fields |
| `source_record_ids_json`, `dois_json` | Source record membership and DOI lists |
| `source_assertions_json` | Supporting edges with source endpoints, DOI and fraction/basis |

## Experiment categories

The 13 Experiment nodes represent shared experimental categories. Article-specific
result records, DOI and source endpoint identifiers preserve experimental context.
Combine source assertions with matching DOI and source endpoints to recover
article-specific fuel, condition and result paths.

## Build

Use extraction tables named `merged_entities.csv` and `merged_relations.csv`:

```bash
python scripts/data_preparation/export_graph.py --data-dir /path/to/extraction_tables
python scripts/data_preparation/export_rdf.py
```

`canonical_graph.py` groups normalized entity names, mixture compositions, condition
values and property values. Chemical formulas retain case-sensitive identities,
and real-fuel names use punctuation and sample-scope normalization.
