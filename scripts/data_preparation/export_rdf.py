"""Export the canonical Neo4j CSV graph to Turtle, retaining source evidence."""
import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote
import zipfile
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
ROOT = Path(__file__).resolve().parents[2]
FKG = Namespace('https://github.com/Yanpeng126/fuel-knowledge-graph/ontology/')
ENTITY = Namespace('https://github.com/Yanpeng126/fuel-knowledge-graph/entity/')
ASSERTION = Namespace('https://github.com/Yanpeng126/fuel-knowledge-graph/assertion/')

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

class CountingGraph(Graph):

    def __init__(self):
        super().__init__()
        self.statement_count = 0

    def add(self, triple):
        self.statement_count += 1
        return self

    def addN(self, quads):
        for _ in quads:
            self.statement_count += 1
        return self

def export(data_dir, output):
    csv.field_size_limit(100000000)
    nodes = read_csv(data_dir / 'nodes.csv')
    edges = read_csv(data_dir / 'relationships.csv')
    ids = {row['entity_id:ID(FKG)']: URIRef(ENTITY + quote(row['entity_id:ID(FKG)'], safe='')) for row in nodes}
    assert len(ids) == len(nodes), 'Duplicate node identifiers'
    assert all((row[':START_ID(FKG)'] in ids and row[':END_ID(FKG)'] in ids for row in edges))
    output.mkdir(parents=True, exist_ok=True)
    schema = Graph()
    schema.bind('fkg', FKG)
    schema.bind('owl', OWL)
    schema.bind('rdfs', RDFS)
    schema.add((URIRef(str(FKG)), RDF.type, OWL.Ontology))
    schema.add((URIRef(str(FKG)), RDFS.comment, Literal('Schema of the canonical Fuel Knowledge Graph, including relationship provenance.')))
    for kind in sorted({row[':LABEL'] for row in nodes} | {'RelationAssertion'}):
        schema.add((FKG[kind], RDF.type, OWL.Class))
        schema.add((FKG[kind], RDFS.label, Literal(kind)))
    for kind in sorted({row[':TYPE'] for row in edges}):
        schema.add((FKG[kind], RDF.type, OWL.ObjectProperty))
        schema.add((FKG[kind], RDFS.label, Literal(kind)))
    keys = {'entity_id', 'original_name', 'attributes', 'attribute_variants', 'fields', 'record_count', 'source_count', 'source_record_ids', 'dois', 'evidence_count', 'source_assertions', 'edge_id'}
    for row in nodes:
        keys.update(json.loads(row['attributes_json']))
    for key in sorted(keys):
        schema.add((FKG[key.replace(' ', '_')], RDF.type, OWL.DatatypeProperty))
    schema.serialize(output / 'fkg_schema.ttl', format='turtle')
    count = 0
    with (output / 'fkg_data.ttl').open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('@prefix fkg: <' + str(FKG) + '> .\n')

        def triple(subject, predicate, obj):
            nonlocal count
            handle.write(f'{subject.n3()} {predicate.n3()} {obj.n3()} .\n')
            count += 1

        def json_field(subject, name, value):
            if value not in ({}, [], None):
                triple(subject, FKG[name], Literal(json.dumps(value, ensure_ascii=False, separators=(',', ':')), datatype=RDF.JSON))
        for row in nodes:
            subject = ids[row['entity_id:ID(FKG)']]
            triple(subject, RDF.type, FKG[row[':LABEL']])
            triple(subject, RDFS.label, Literal(row['name']))
            triple(subject, FKG.entity_id, Literal(row['entity_id:ID(FKG)']))
            triple(subject, FKG.original_name, Literal(row['original_name']))
            for key in ('record_count', 'source_count'):
                triple(subject, FKG[key], Literal(int(row[key + ':int']), datatype=XSD.integer))
            attrs = json.loads(row['attributes_json'])
            for field, value in attrs.items():
                if value is None or value == '' or value == [] or (value == {}):
                    continue
                literal = Literal(json.dumps(value, ensure_ascii=False), datatype=RDF.JSON) if isinstance(value, (dict, list)) else Literal(value)
                triple(subject, FKG[field.replace(' ', '_')], literal)
            for column in ('attributes', 'attribute_variants', 'fields', 'source_record_ids', 'dois'):
                json_field(subject, column, json.loads(row[column + '_json']))
        for row in edges:
            source, target = (ids[row[':START_ID(FKG)']], ids[row[':END_ID(FKG)']])
            predicate = FKG[row[':TYPE']]
            assertion = URIRef(ASSERTION + row['edge_id'])
            triple(source, predicate, target)
            for pred, obj in [(RDF.type, RDF.Statement), (RDF.type, FKG.RelationAssertion), (RDF.subject, source), (RDF.predicate, predicate), (RDF.object, target)]:
                triple(assertion, pred, obj)
            triple(assertion, FKG.edge_id, Literal(row['edge_id']))
            triple(assertion, FKG.evidence_count, Literal(int(row['evidence_count:int']), datatype=XSD.integer))
            for column in ('fields', 'dois', 'source_assertions'):
                json_field(assertion, column, json.loads(row[column + '_json']))
    print(f'Wrote {count} Turtle statements; validating syntax...', flush=True)
    parsed = CountingGraph()
    parsed.parse(output / 'fkg_data.ttl', format='turtle')
    assert parsed.statement_count == count
    Graph().parse(output / 'fkg_schema.ttl', format='turtle')
    report = {'entities': len(nodes), 'relationships': len(edges), 'statements': count}
    documentation = (ROOT / 'rdf/README.md').read_text(encoding='utf-8')
    (output / 'README.md').write_text(documentation, encoding='utf-8')
    with zipfile.ZipFile(output / 'FKG_RDF.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in ('fkg_schema.ttl', 'fkg_data.ttl', 'README.md'):
            archive.write(output / name, name)
    print(json.dumps(report, indent=2), flush=True)
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/neo4j')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/rdf')
    args = parser.parse_args()
    export(args.data_dir, args.output_dir)
