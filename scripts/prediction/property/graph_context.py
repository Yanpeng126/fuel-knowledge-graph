"""Label-free, DOI-excluded fuel neighborhood features for recipe OOF."""
from pathlib import Path
import csv
import json
from collections import defaultdict, Counter
import numpy as np

class GraphContext:

    def __init__(self, root, feature):
        root = Path(root)
        if not (root / 'original_entities.csv').is_file():
            raise FileNotFoundError('Source-level graph context is required for this property model. Set FKG_GRAPH_RECORD_DIR to the training graph record directory; the canonical shared-category graph cannot substitute for source-level context.')
        self.feature = feature
        self.events = defaultdict(list)
        self.cache = {}
        entities = {}
        with (root / 'original_entities.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                entities[row['entity_id']] = (row['entity_type'], row['name'], row['doi'])
        adjacent = defaultdict(set)
        with (root / 'original_relations.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['relation_type'] in {'HAS_EXPERIMENT', 'HAS_RESULT_VALUE', 'HAS_CONDITION'}:
                    adjacent[row['source_id']].add(row['target'])
        paper_fuels = defaultdict(set)
        for kind, name, doi in entities.values():
            if kind == 'Fuel':
                paper_fuels[doi].add(self.identity(name))
        vocabulary = set()
        for eid, (kind, name, doi) in entities.items():
            if kind != 'Fuel':
                continue
            counts = Counter()
            for target in adjacent[eid]:
                tk, tn, _ = entities.get(target, ('', '', ''))
                if tk in {'Experiment', 'ExpResult'}:
                    counts[tk + ':' + tn] += 1
                if tk == 'Experiment':
                    for condition in adjacent[target]:
                        ck, cn, _ = entities.get(condition, ('', '', ''))
                        if ck == 'Condition':
                            counts['Condition:' + cn] += 1
            vocabulary.update(counts)
            self.events[self.identity(name)].append((doi, counts, paper_fuels[doi]))
        self.columns = sorted(vocabulary)
        print('GRAPH', len(self.events), 'fuels', len(self.columns), 'context types', flush=True)

    def identity(self, name):
        f = self.feature(name)
        return str(f['CID']) if f and f.get('CID') else name.strip().casefold()

    def add_fuel_layer(self, root, numeric=False):
        entities = {}
        properties = {}
        widths = {'Density': 0.025, 'Molecular weight': 10, 'H/C ratio': 0.1, 'Yield sooting index (YSI)': 5, 'Threshold sooting index (TSI)': 5, 'Flash point': 10, 'Freezing point': 5, 'Smoke point': 5}
        with (root / 'merged_entities.csv').open(encoding='utf-8-sig', newline='') as stream:
            for r in csv.DictReader(stream):
                if r['entity_type'] != 'Property':
                    entities[r['entity_id']] = (r['entity_type'], r['name'], r['doi'])
                elif numeric and r['property_type'] in widths:
                    try:
                        value = float(r['value'])
                    except ValueError:
                        continue
                    if not np.isfinite(value):
                        continue
                    prop, unit = (r['property_type'], r['unit'].strip().casefold())
                    if prop == 'Density':
                        if unit in {'kg/m3', 'kg/m^3', 'kg/m³'}:
                            value /= 1000
                        elif unit not in {'g/cm3', 'g/cm^3', 'g/cm³', 'g/ml'}:
                            continue
                    elif prop == 'Molecular weight' and unit not in {'g/mol', 'g/mole'}:
                        continue
                    elif prop in {'Flash point', 'Freezing point'}:
                        if unit == 'k':
                            value -= 273.15
                        elif unit not in {'c', '°c', 'degc', 'deg c'}:
                            continue
                    elif prop == 'Smoke point' and unit != 'mm':
                        continue
                    bucket = int(np.clip(round(value / widths[prop]), -40, 200))
                    properties[r['entity_id']] = (f'NeighborProperty:{prop}:bin:{bucket}', r['doi'])
        components, context, surrogate = (defaultdict(set), defaultdict(Counter), set())
        sample_parent = {}
        with (root / 'merged_relations.csv').open(encoding='utf-8-sig', newline='') as stream:
            for r in csv.DictReader(stream):
                src, target = (r['source_id'], r['target'])
                if r['relation_type'] == 'HAS_COMPONENT':
                    components[src].add(target)
                elif r['relation_type'] == 'SURROGATE_FOR':
                    surrogate.add(src)
                elif r['relation_type'] == 'SAMPLE_OF':
                    sample_parent[src] = target
                elif r['relation_type'] == 'HAS_PROPERTY' and target in properties:
                    label, property_doi = properties[target]
                    owner_doi = entities.get(src, ('', '', ''))[2]
                    flags = json.loads(r['attributes_json'] or '{}')
                    if not property_doi.startswith('10.') or property_doi != owner_doi or r['doi'] != property_doi:
                        continue
                    if flags.get('needs_source_review') or flags.get('cross_doi_reference'):
                        continue
                    context[src][label] += 1
                elif r['relation_type'] in {'HAS_EXPERIMENT', 'HAS_RESULT_VALUE'} and target in entities:
                    kind, name, _ = entities[target]
                    if kind in {'Experiment', 'ExpResult'}:
                        context[src][kind + ':' + name] += 1

        def leaves(eid, visited):
            if eid in visited:
                return set()
            kind, name, _ = entities.get(eid, ('Fuel', eid, ''))
            if eid not in components:
                return {self.identity(name)} if kind in {'Fuel', 'PureFuel', 'RealFuel', 'FuelSample'} else set()
            return set().union(*(leaves(c, visited | {eid}) for c in components[eid]))
        added, seen = (0, set())
        for eid in components:
            kind, _, doi = entities.get(eid, ('', '', ''))
            if kind not in {'Mixture', 'FuelSample'} or not doi:
                continue
            fuels = leaves(eid, set())
            key = (doi, tuple(sorted(fuels)))
            if not fuels or (len(fuels) < 2 and (not numeric)) or key in seen:
                continue
            seen.add(key)
            counts = context[eid].copy()
            counts['Mixture:membership'] = 1
            counts['Mixture:surrogate'] = int(eid in surrogate)
            for fuel in fuels:
                self.events[fuel].append((doi, counts, fuels))
            added += 1
        if numeric:
            for eid, counts in context.items():
                kind, name, doi = entities.get(eid, ('', '', ''))
                if kind not in {'Fuel', 'PureFuel', 'RealFuel', 'FuelSample'} or not doi:
                    continue
                values = Counter({k: v for k, v in counts.items() if k.startswith('NeighborProperty:')})
                if not values:
                    continue
                identity = self.identity(name)
                self.events[identity].append((doi, values, {identity}))
                if eid in sample_parent:
                    _, parent_name, _ = entities.get(sample_parent[eid], ('', '', ''))
                    if parent_name:
                        parent = self.identity(parent_name)
                        self.events[parent].append((doi, values, {parent}))
        self.columns = sorted(set(self.columns) | {k for events in self.events.values() for _, cc, _ in events for k in cc})
        self.cache.clear()
        print('FUEL_LAYER', added, 'DOI/component-set contexts;', len(properties), 'non-target properties', flush=True)

    def matrices(self, train, valid, excluded):
        excluded = frozenset(excluded)
        key = excluded
        if key not in self.cache:
            vectors = {}
            for fuel, events in self.events.items():
                counts, dois, neighbors = (Counter(), set(), set())
                for doi, cc, co in events:
                    if doi not in excluded:
                        counts.update(cc)
                        dois.add(doi)
                        neighbors.update(co - {fuel})
                raw = np.array([counts[c] for c in self.columns], dtype=float)
                distribution = raw / max(raw.sum(), 1)
                vectors[fuel] = np.r_[np.log1p([len(dois), len(neighbors), raw.sum()]), np.log1p(raw), distribution]
            self.cache[key] = vectors
        vectors = self.cache[key]
        zero = np.zeros(3 + 2 * len(self.columns))

        def matrix(rows):
            result = []
            for row in rows:
                vv = np.array([vectors.get(self.identity(n), zero) for n in row['names']])
                raw = row['fraction_values']
                w = np.array(raw, dtype=float) if all((v is not None for v in raw)) and len(raw) == len(vv) else np.ones(len(vv))
                w /= w.sum()
                mean = w @ vv
                result.append(np.r_[mean, np.sqrt(w @ (vv - mean) ** 2), np.mean(np.any(vv != 0, axis=1))])
            return np.array(result)
        return (matrix(train), matrix(valid))
