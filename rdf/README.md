# RDF representation

`fkg_schema.ttl` defines the FKG classes and properties in RDFS/OWL.
`FKG_RDF.zip` contains `fkg_data.ttl`, the schema and this description.
The canonical graph has **46,133 entities and 95,914 directed relationships**,
matching the Neo4j CSV and web explorer.

## Entity identities

Experiment nodes represent 13 shared apparatus/experiment categories. Mixture
nodes describe formulations, and experimental result records retain source-specific
observations. The schema covers every exported entity and relationship type.

## Relations and provenance

Each canonical relationship has a direct triple and a reified
`rdf:Statement` / `fkg:RelationAssertion`. `evidence_count` gives the number of
supporting source assertions. `source_assertions` is an RDF JSON literal containing
source endpoints, DOI, fractions and related fields. `fields` stores values shared
by all supporting assertions.

Nodes retain agreed attributes, attribute variants, source identifiers and DOI lists.
Relationship evidence covers 118,631 source assertions in the relationship CSV and RDF.
Use source DOI and endpoint identifiers to assemble article-specific experimental
paths. Individual assertions preserve differing fractions and reported values.

## Namespaces and values

Resource identifiers use `https://github.com/Yanpeng126/fuel-knowledge-graph/` with
`ontology/`, `entity/` and `assertion/` suffixes. Complex attributes use RDF JSON
literals; record and evidence counts use integers. Other values retain source
formatting and units.

The RDF file includes entity descriptions, directed relationships and reification
resources.
