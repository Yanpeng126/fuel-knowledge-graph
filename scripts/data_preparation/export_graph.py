"""Export the canonical explorer graph as Neo4j CSV with source evidence."""
import argparse
import csv
import json
from pathlib import Path
from canonical_graph import Graph

def strip_workflow_metadata(value):
    if isinstance(value, dict):
        return {key: strip_workflow_metadata(item) for key, item in value.items()
                if key not in {'origin', 'record_status'}}
    if isinstance(value, list):
        return [strip_workflow_metadata(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        cleaned = strip_workflow_metadata(parsed)
        if cleaned != parsed:
            return json.dumps(cleaned, ensure_ascii=False, separators=(',', ':'))
    return value


def encode(value):
    return json.dumps(strip_workflow_metadata(value), ensure_ascii=False, separators=(',', ':'))


def export(data_dir, output):
    graph = Graph(data_dir)
    assert not graph.missing_edges, 'Unresolved relationship endpoints'
    output.mkdir(parents=True, exist_ok=True)
    node_fields = ['entity_id:ID(FKG)', ':LABEL', 'name', 'original_name', 'attributes_json', 'attribute_variants_json', 'fields_json', 'record_count:int', 'source_count:int', 'source_record_ids_json', 'dois_json']
    with (output / 'nodes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=node_fields)
        writer.writeheader()
        for identifier, node in graph.nodes.items():
            members = graph.members[identifier]
            dois = sorted({graph.records[i]['fields'].get('doi') for i in members if graph.records[i]['fields'].get('doi')})
            writer.writerow({'entity_id:ID(FKG)': identifier, ':LABEL': node['type'], 'name': node['display_name'], 'original_name': node['name'], 'attributes_json': encode(node['attributes']), 'attribute_variants_json': encode(node['attribute_variants']), 'fields_json': encode(node['fields']), 'record_count:int': node['record_count'], 'source_count:int': node['source_count'], 'source_record_ids_json': encode(members), 'dois_json': encode(dois)})
    edge_fields = ['edge_id', ':START_ID(FKG)', ':TYPE', ':END_ID(FKG)', 'evidence_count:int', 'fields_json', 'dois_json', 'source_assertions_json']
    with (output / 'relationships.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        for edge in graph.edges:
            evidence = graph.edge_evidence[edge['id']]
            dois = sorted({item['fields'].get('doi') for item in evidence if item['fields'].get('doi')})
            writer.writerow({'edge_id': f"r{edge['id']:07d}", ':START_ID(FKG)': edge['source'], ':TYPE': edge['type'], ':END_ID(FKG)': edge['target'], 'evidence_count:int': len(evidence), 'fields_json': encode(edge['fields']), 'dois_json': encode(dois), 'source_assertions_json': encode(evidence)})
    report = {'nodes': len(graph.nodes), 'relationships': len(graph.edges)}
    print(json.dumps(report, indent=2), flush=True)
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Record-oriented extraction CSV directory')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data/neo4j')
    args = parser.parse_args()
    export(args.data_dir, args.output_dir)
