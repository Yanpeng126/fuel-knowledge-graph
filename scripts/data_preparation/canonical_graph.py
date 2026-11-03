"""Entity identities and aggregation rules shared with the graph explorer."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
ROOT = Path(__file__).resolve().parent
COLORS = ['#63dcce', '#92a8ff', '#ffbf73', '#f185aa', '#b99bff', '#91cc87', '#ffe28b', '#6dc4f5', '#eca0e4', '#bec9d7']
TYPE_COLORS = dict(zip(('Condition', 'ExpResult', 'Fuel', 'Mixture', 'Experiment', 'Property', 'RealFuel', 'FuelSample'), COLORS))
HIDDEN_NODE_IDS = {'OLD:10.1016/j.fuel.2023.129285#condition_3'}

def normalize(value):
    return re.sub('\\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()

def has_attribute_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in ('', 'null', 'none', 'nan', '[]', '{}')
    if isinstance(value, (list, dict)):
        return any((has_attribute_value(v) for v in (value.values() if isinstance(value, dict) else value)))
    if isinstance(value, float) and math.isnan(value):
        return False
    return True

def clean_text(value):
    return re.sub('\\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip()

def fuel_name_key(value):
    text = clean_text(value)
    if re.fullmatch('(?:n-|i-|iso-)?(?:[A-Z][a-z]?\\d*)+', text):
        return text
    return text.casefold()

def fuel_identities(records):
    names, aliases = (defaultdict(set), defaultdict(set))
    for node in records.values():
        if node['type'] != 'Fuel':
            continue
        key = fuel_name_key(node['name'])
        names[key].add(clean_text(node['name']))
        values = node['attributes'].get('aliases', [])
        for alias in values if isinstance(values, list) else [values]:
            if isinstance(alias, str) and has_attribute_value(alias):
                aliases[key].add(fuel_name_key(alias))
    parent = {key: key for key in names}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    config = json.loads((ROOT / 'fuel_aliases.json').read_text(encoding='utf-8'))
    preferences = {}
    for group in config['groups']:
        keys = {fuel_name_key(name) for name in group['names']} & names.keys()
        for key in sorted(keys):
            for other in sorted(aliases[key] & keys):
                if root(key) != root(other):
                    parent[root(other)] = root(key)
            preferences[key] = group['canonical']
    components = defaultdict(list)
    for key in names:
        components[root(key)].append(key)
    identities = {}
    for keys in components.values():
        preferred = sorted({preferences[k] for k in keys if k in preferences})
        label = preferred[0] if len(keys) > 1 and preferred else min(names[min(keys)])
        identity = fuel_name_key(label)
        for key in keys:
            identities[key] = (identity, label)
    audit = []
    for key in sorted(names):
        for other in sorted(aliases[key] & names.keys()):
            if key == other:
                continue
            merged = identities[key][0] == identities[other][0]
            audit.append({'name': min(names[key]), 'alias_name': min(names[other]), 'decision': 'merged' if merged else 'review_required', 'canonical': identities[key][1] if merged else None})
    return (identities, audit)

def real_fuel_names():
    config = json.loads((ROOT / 'real_fuel_names.json').read_text(encoding='utf-8'))
    return {normalize(name): group['canonical'] for group in config['groups'] for name in group['names']}

def real_fuel_name_key(name):
    text = normalize(name).translate(str.maketrans({'‐': '-', '‑': '-', '–': '-', '—': '-'}))
    text = re.sub('\\bjet[ -]*a(?:[ -]*(\\d+))?(?=$|\\s)', lambda m: 'jet a' + ('-' + m[1] if m[1] else ''), text)
    text = re.sub('\\b(jp|rp|posf)[ -]*(\\d+)\\b', '\\1-\\2', text)
    return re.sub('\\s+', ' ', text).strip()

def real_fuel_identity(node, reviewed):
    name = clean_text(node['name'])
    text = real_fuel_name_key(name)
    label = reviewed.get(text, name)
    identity = real_fuel_name_key(label)
    local = bool(re.fullmatch('(?:test fuel|fuel|coal|gasoline|diesel|biodiesel)[ -]+(?:[a-z]|\\d+|[a-z]{1,2}\\d+)', text) or re.search('\\b(?:sample|batch|lot)\\s*#?\\s*(?:\\d+|[a-z])\\b', text) or re.fullmatch('(?:hefa|cn)[ -]*\\d+', text) or (re.fullmatch('jet a(?:[ -]+a?\\d+)', text) and text not in reviewed))
    scope = clean_text(recorded_value(node, 'doi') or node['id']) if local else ''
    return ([identity, scope], label)

def measurement_text(value):
    text = clean_text(value).replace('−', '-')
    if re.fullmatch('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?', text):
        try:
            number = Decimal(text)
            if number == 0:
                return '0'
            if -4 <= number.adjusted() <= 6:
                fixed = format(number, 'f')
                if '.' in fixed:
                    fixed = fixed.rstrip('0').rstrip('.')
                return fixed.lstrip('+')
            scientific = format(number, 'E')
            mantissa, exponent = scientific.split('E')
            return mantissa.rstrip('0').rstrip('.') + 'E' + str(int(exponent)) if '.' in mantissa else mantissa + 'E' + str(int(exponent))
        except InvalidOperation:
            pass
    return text

def unit_text(value):
    if not has_attribute_value(value):
        return ''
    text = clean_text(value).replace('⋅', '·').replace('*', '·')
    text = re.sub('\\^(?=[23]\\b)', '', text)
    text = re.sub('\\s*([/()·])\\s*', '\\1', text)
    text = text.replace(' ', '·')
    text = {'mPa.s': 'mPa·s', 'Pa.s': 'Pa·s', 'W/m/K': 'W/(m·K)', 'J/kg/K': 'J/(kg·K)'}.get(text, text)
    return re.sub('(?<=[A-Za-z])([23])\\b', lambda m: {'2': '²', '3': '³'}[m[1]], text)

def recorded_value(node, key):
    for section in (node['attributes'], node['fields']):
        if has_attribute_value(section.get(key)):
            return section[key]
    return None

def property_identity(node):
    kind, value = (recorded_value(node, 'property_type'), recorded_value(node, 'value'))
    if kind is None or value is None:
        return None

    def condition_text(value):
        text = clean_text(value)
        if re.fullmatch('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?', text):
            return measurement_text(value)
        match = re.fullmatch('([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*(\\D.*)', text)
        return measurement_text(match[1]) + ' ' + unit_text(match[2]) if match else measurement_text(value)
    conditions = {k: condition_text(recorded_value(node, k)) for k in ('temperature', 'pressure') if recorded_value(node, k) is not None}
    return [clean_text(kind).casefold(), measurement_text(value), unit_text(recorded_value(node, 'unit')), conditions]

def component_name(name):
    text = clean_text(name)
    labels = {'ch4': 'CH4', 'methane': 'CH4', 'h2': 'H2', 'hydrogen': 'H2', 'co2': 'CO2', 'carbon dioxide': 'CO2', 'n2': 'N2', 'nitrogen': 'N2', 'o2': 'O2', 'oxygen': 'O2', 'ar': 'Ar', 'argon': 'Ar', 'n2o': 'N2O', 'c2h4': 'C2H4', 'ethylene': 'C2H4', 'c2h6': 'C2H6', 'ethane': 'C2H6', 'c3h8': 'C3H8', 'propane': 'C3H8', 'nh3': 'NH3', 'ammonia': 'NH3'}
    return labels.get(text.casefold(), text)

def composition_signature(rows):
    return sorted({json.dumps([r['component'], r['role'], measurement_text(r['value']) if r['value'] is not None else None, clean_text(r['basis']).casefold() if r['basis'] else None], ensure_ascii=False) for r in rows})

def display_name(name):
    label = re.sub('\\s+', ' ', name).strip()
    if len(label) <= 72:
        return label
    match = re.search('\\s+\\(([^()]*)\\)\\s*$', label)
    if match and len(match.group(1)) > 80 and (';' in match.group(1) or re.match('(?:Table|Figure|present experiment|experiment conditions)\\b', match.group(1), re.I)):
        label = label[:match.start()].strip()
    label = label.replace(' versus ', ' / ')
    temperature = re.search('(?:coflow control temperature|temperature|\\bT\\s*[:=])\\s*(\\d+(?:\\.\\d+)?\\s*K)\\b', name, re.I)
    suffix = ' · ' + temperature.group(1) if temperature and temperature.group(1) not in label else ''
    available = 72 - len(suffix)
    if len(label) > available:
        label = label[:available - 1].rstrip() + '…'
    return label + suffix

class Graph:

    def __init__(self, data_dir: Path):
        csv.field_size_limit(100000000)
        self.nodes = {}
        self.search_rows = []
        self.edges = []
        self.adjacency = defaultdict(list)
        self.hidden_nodes = set()
        with (data_dir / 'merged_entities.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                identifier = row.get('entity_id', '')
                if identifier in HIDDEN_NODE_IDS and row.get('entity_type') == 'UnresolvedEntity':
                    self.hidden_nodes.add(identifier)
                    continue
                if not identifier or identifier in self.nodes:
                    raise ValueError(f'Missing or duplicate entity_id: {identifier}')
                try:
                    attributes = json.loads(row.get('attributes_json') or '{}')
                except json.JSONDecodeError:
                    attributes = {'raw_attributes': row.get('attributes_json', '')}
                if not isinstance(attributes, dict):
                    attributes = {'raw_attributes': attributes}
                original_type = row.get('entity_type') or 'Unknown'
                if original_type in ('MixedFuel', 'Mixture'):
                    attributes = {**attributes, 'version': 1 if original_type == 'MixedFuel' else 2}
                    if 'entity_type' in attributes:
                        attributes['entity_type'] = 'Mixture'
                node = {'id': identifier, 'type': 'Mixture' if original_type == 'MixedFuel' else original_type, 'name': row.get('name') or row.get('property_type') or identifier, 'fields': {k: v for k, v in row.items() if v and v.strip().casefold() not in ('null', 'none', 'nan') and (k not in ('attributes_json', 'entity_id', 'entity_type', 'name'))}, 'attributes': attributes}
                node['display_name'] = display_name(node['name'])
                self.nodes[identifier] = node
                aliases = attributes.get('aliases', [])
                if not isinstance(aliases, list):
                    aliases = [aliases]
                names = [normalize(node['name']), normalize(node['display_name']), normalize(row.get('normalized name', ''))] + [normalize(a) for a in aliases]
                text = normalize(' '.join((str(v) for v in row.values())) + ' ' + node['display_name'])
                self.search_rows.append((identifier, [n for n in names if n], text))
        missing = 0
        hidden_edges = 0
        with (data_dir / 'merged_relations.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                source, target = (row['source_id'], row['target'])
                if source in self.hidden_nodes or target in self.hidden_nodes:
                    hidden_edges += 1
                    continue
                if source not in self.nodes or target not in self.nodes:
                    missing += 1
                    continue
                edge = {'id': len(self.edges), 'source': source, 'target': target, 'type': row['relation_type'], 'fields': {k: v for k, v in row.items() if v and k not in ('source_id', 'target', 'relation_type')}}
                index = len(self.edges)
                self.edges.append(edge)
                self.adjacency[source].append(index)
                if target != source:
                    self.adjacency[target].append(index)
        self.missing_edges = missing
        self.hidden_edges = hidden_edges
        self._prepare_compositions()
        self._merge_entities()
        counts = Counter((n['type'] for n in self.nodes.values()))
        self.groups = []
        for index, (kind, count) in enumerate(counts.most_common()):
            angle = index * 2 * math.pi / len(counts)
            self.groups.append({'type': kind, 'count': count, 'color': TYPE_COLORS.get(kind, COLORS[index % len(COLORS)]), 'position': [math.cos(angle) * 780, math.sin(angle * 2) * 120, math.sin(angle) * 780]})
        self.group_map = {g['type']: g for g in self.groups}

    def _prepare_compositions(self):
        self.record_compositions = {}
        self.composition_labels = {}
        roles = {'HAS_COMPONENT': 'fuel', 'CONTAINS_FUEL': 'component', 'HAS_OXIDIZER': 'oxidizer', 'HAS_DILUENT': 'diluent'}
        for identifier, node in self.nodes.items():
            if node['type'] != 'Mixture':
                continue
            rows = []
            original = node['attributes'].get('composition')
            if isinstance(original, dict):
                for name, value in original.items():
                    if has_attribute_value(name):
                        rows.append({'component': component_name(name), 'role': 'component', 'value': value if has_attribute_value(value) else None, 'basis': None, 'source': 'original_composition', 'target_id': None, 'edge_ids': []})
            original_names = {r['component'] for r in rows}
            for index in self.adjacency[identifier]:
                edge = self.edges[index]
                if edge['source'] != identifier or edge['type'] not in roles:
                    continue
                target = self.nodes[edge['target']]
                name = component_name(target['name'])
                value = edge['fields'].get('fraction')
                value = value if has_attribute_value(value) else None
                basis = edge['fields'].get('fraction_basis')
                basis = basis if has_attribute_value(basis) else None
                if name in original_names and value is None:
                    for row in rows:
                        if row['component'] == name:
                            row['target_id'] = edge['target']
                            row['edge_ids'].append(edge['id'])
                    continue
                rows.append({'component': name, 'role': roles[edge['type']], 'value': value, 'basis': basis, 'source': 'relationship', 'target_id': edge['target'], 'edge_ids': [edge['id']]})
            unique = {}
            for row in rows:
                key = json.dumps([row['component'], row['role'], row['value'], row['basis'], row['source']], sort_keys=True)
                if key in unique:
                    unique[key]['edge_ids'].extend(row['edge_ids'])
                else:
                    unique[key] = row
            rows = list(unique.values())
            self.record_compositions[identifier] = rows
            original_rows = [r for r in rows if r['source'] == 'original_composition']
            if len(original_rows) > 1 and component_name(node['name']) in {r['component'] for r in original_rows}:
                self.composition_labels[identifier] = display_name(' / '.join((r['component'] + (' ' + str(r['value']) if r['value'] is not None else '') for r in original_rows)))

    def composition_groups(self, identifier):
        groups = {}
        for record_id in self.members[self.resolve(identifier)]:
            rows = self.record_compositions.get(record_id, [])
            if not rows:
                continue
            key = json.dumps(composition_signature(rows))
            group = groups.setdefault(key, {'rows': rows, 'record_count': 0, 'sources': []})
            group['record_count'] += 1
            group['sources'].append({'id': record_id, 'doi': self.records[record_id]['fields'].get('doi', ''), 'version': self.records[record_id]['attributes'].get('version')})
        return list(groups.values())

    def _merge_entities(self):
        self.records = self.nodes
        raw_edges = self.edges
        self.raw_edge_count = len(raw_edges)
        fuel_names, self.fuel_alias_audit = fuel_identities(self.records)
        reviewed_real = real_fuel_names()
        reviewed_real.update({real_fuel_name_key(name): label for name, label in list(reviewed_real.items())})
        self.real_fuel_normalization = {}
        name_key = lambda n: re.sub('\\s+', ' ', unicodedata.normalize('NFKC', n['name'])).strip()
        grouped = defaultdict(list)
        for identifier, node in self.records.items():
            kind, attrs = (node['type'], node['attributes'])
            key = [kind, name_key(node)]
            if kind == 'Fuel':
                key = [kind, fuel_names[fuel_name_key(node['name'])][0]]
            elif kind == 'RealFuel':
                identity, label = real_fuel_identity(node, reviewed_real)
                self.real_fuel_normalization[identifier] = {'identity': identity, 'display_name': label, 'name': node['name'], 'proposed_name': node['fields'].get('normalized name', '')}
                key = [kind, identity]
            elif kind == 'Condition':
                values = {k: attrs[k] for k in ('condition_type', 'value', 'unit') if has_attribute_value(attrs.get(k))}
                key.append(values if has_attribute_value(values.get('value')) else identifier)
            elif kind == 'Property':
                key = [kind, property_identity(node) or identifier]
            elif kind == 'Experiment':
                key = [kind, clean_text(recorded_value(node, 'exp_type') or node['name']).casefold()]
            elif kind == 'Mixture':
                rows = self.record_compositions.get(identifier, [])
                if rows:
                    key.append(composition_signature(rows))
            elif kind == 'FuelSample':
                key.extend([attrs.get('sample_id', identifier), attrs.get('source_or_batch', '')])
            elif kind not in ('Fuel', 'RealFuel', 'Mixture'):
                key.append(identifier)
            grouped[json.dumps(key, sort_keys=True, ensure_ascii=False)].append(identifier)
        self.members, self.canonical_ids, self.nodes = ({}, {}, {})
        for key, ids in grouped.items():
            canonical = ids[0] if len(ids) == 1 else 'KG:' + hashlib.sha256(key.encode()).hexdigest()[:24]
            self.members[canonical] = ids
            for identifier in ids:
                self.canonical_ids[identifier] = canonical
            records = [self.records[i] for i in ids]
            first = records[0]
            variants = defaultdict(dict)
            for record in records:
                for attr, value in record['attributes'].items():
                    if not has_attribute_value(value) or attr in ('entity_id', 'entity_type', 'name', 'doi'):
                        continue
                    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
                    item = variants[attr].setdefault(encoded, {'value': value, 'record_count': 0})
                    item['record_count'] += 1
            attributes = {k: next(iter(v.values()))['value'] for k, v in variants.items() if len(v) == 1}
            aliases = list(dict.fromkeys((str(alias) for record in records for alias in (record['attributes'].get('aliases', []) if isinstance(record['attributes'].get('aliases', []), list) else [record['attributes']['aliases']]) if has_attribute_value(alias))))
            if aliases:
                attributes['aliases'] = aliases
            if first['type'] in ('Fuel', 'RealFuel') and len({r['name'] for r in records}) > 1:
                attributes['aliases'] = list(dict.fromkeys([*(r['name'] for r in records), *aliases]))
            differing = {k: list(v.values()) for k, v in variants.items() if len(v) > 1 and k != 'aliases'}
            if first['type'] == 'RealFuel' and len(ids) > 1:
                for attr in ('sample_id', 'source_or_batch'):
                    if attr in variants and sum((v['record_count'] for v in variants[attr].values())) < len(ids):
                        attributes.pop(attr, None)
                        differing[attr] = list(variants[attr].values())
            fields = {k: v for k, v in first['fields'].items() if all((r['fields'].get(k) == v for r in records))}
            label = self.composition_labels.get(ids[0], first['display_name'])
            if first['type'] == 'Fuel':
                label = display_name(fuel_names[fuel_name_key(first['name'])][1])
            elif first['type'] == 'RealFuel':
                label = display_name(self.real_fuel_normalization[ids[0]]['display_name'])
            elif first['type'] == 'Condition' and has_attribute_value(attributes.get('value')):
                value = attributes['value']
                text = '; '.join(map(str, value)) if isinstance(value, list) else str(value)
                unit = attributes.get('unit') if has_attribute_value(attributes.get('unit')) else ''
                label = display_name(f"{first['name']} · {text} {unit}")
            elif first['type'] == 'Experiment':
                label = clean_text(recorded_value(first, 'exp_type') or first['name'])
            elif first['type'] == 'Property':
                identity = property_identity(first)
                if identity:
                    kind = clean_text(recorded_value(first, 'property_type'))
                    label = f'{kind} · {identity[1]} {identity[2]}'.rstrip()
                    if identity[3]:
                        context = '; '.join((('T' if k == 'temperature' else 'P') + '=' + v for k, v in identity[3].items()))
                        label += ' · ' + context
                    label = display_name(label)
                    if len(ids) > 1:
                        attributes.update({'property_type': kind, 'value': identity[1]})
                        if identity[2]:
                            attributes['unit'] = identity[2]
                        attributes.update(identity[3])
                        for k in ('property_type', 'value', 'unit', 'temperature', 'pressure'):
                            differing.pop(k, None)
            if len(ids) == 1:
                attributes = first['attributes']
            self.nodes[canonical] = {'id': canonical, 'type': first['type'], 'name': first['name'], 'display_name': label, 'fields': fields, 'attributes': attributes, 'attribute_variants': differing, 'record_count': len(ids), 'source_count': len({r['fields'].get('doi') for r in records if r['fields'].get('doi')})}
        text_rows = {identifier: (names, text) for identifier, names, text in self.search_rows}
        self.search_rows = []
        for canonical, ids in self.members.items():
            names = list(dict.fromkeys((name for i in ids for name in text_rows[i][0])))
            names.append(normalize(self.nodes[canonical]['display_name']))
            text = normalize(canonical + ' ' + self.nodes[canonical]['display_name']) + ' ' + ' '.join((text_rows[i][1] for i in ids))
            if self.nodes[canonical]['type'] == 'Mixture':
                text += ' ' + normalize(' '.join((str(value) for i in ids for row in self.record_compositions.get(i, []) for value in (row['component'], row['value'], row['basis']) if value is not None)))
            self.search_rows.append((canonical, names, text))
        grouped_edges = defaultdict(list)
        for edge in raw_edges:
            grouped_edges[self.canonical_ids[edge['source']], self.canonical_ids[edge['target']], edge['type']].append(edge)
        self.edges, self.adjacency, self.edge_evidence = ([], defaultdict(list), {})
        for (source, target, kind), evidence in grouped_edges.items():
            index = len(self.edges)
            fields = {k: v for k, v in evidence[0]['fields'].items() if all((e['fields'].get(k) == v for e in evidence))}
            self.edges.append({'id': index, 'source': source, 'target': target, 'type': kind, 'fields': fields, 'evidence_count': len(evidence)})
            self.edge_evidence[index] = evidence
            self.adjacency[source].append(index)
            if source != target:
                self.adjacency[target].append(index)

    def resolve(self, identifier):
        return self.canonical_ids.get(identifier, identifier)

    def record_page(self, identifier, offset=0, limit=20):
        identifier = self.resolve(identifier)
        ids = self.members[identifier]
        return {'records': [{**self.records[i], 'composition_rows': self.record_compositions.get(i, []), 'display_attributes': {k: v for k, v in self.records[i]['attributes'].items() if has_attribute_value(v)}} for i in ids[offset:offset + limit]], 'total': len(ids), 'offset': offset, 'has_more': offset + limit < len(ids)}

    def evidence_page(self, identifier, offset=0, limit=20):
        evidence = self.edge_evidence[int(identifier)]
        return {'evidence': evidence[offset:offset + limit], 'total': len(evidence), 'offset': offset, 'has_more': offset + limit < len(evidence)}

    def compact(self, identifier):
        identifier = self.resolve(identifier)
        node = self.nodes[identifier]
        digest = hashlib.sha256(identifier.encode()).digest()
        values = [int.from_bytes(digest[i:i + 4], 'big') / (2 ** 32 - 1) for i in (0, 4, 8)]
        group = self.group_map[node['type']]
        angle, radius, elevation = (values[0] * math.tau, values[1] ** 0.5 * 270, (values[2] - 0.5) * 280)
        x, y, z = group['position']
        return {'id': identifier, 'name': node['name'], 'display_name': node['display_name'], 'type': node['type'], 'doi': node['fields'].get('doi', ''), 'degree': len(self.adjacency[identifier]), 'record_count': node['record_count'], 'source_count': node['source_count'], 'color': group['color'], 'position': [x + math.cos(angle) * radius, y + elevation, z + math.sin(angle) * radius]}

    def search(self, query, kind='', offset=0, limit=30):
        query = normalize(query)
        if not query:
            return {'results': [], 'total': 0, 'offset': offset, 'limit': limit}
        tokens = query.split()
        hits = []
        for identifier, names, text in self.search_rows:
            if kind and self.nodes[identifier]['type'] != kind:
                continue
            if not all((t in text for t in tokens)):
                continue
            rank = 0 if query in names else 1 if any((n.startswith(query) for n in names)) else 2 if any((query in n for n in names)) else 3
            hits.append((rank, -len(self.adjacency[identifier]), identifier))
        hits.sort()
        return {'results': [self.compact(hit[2]) for hit in hits[offset:offset + limit]], 'total': len(hits), 'offset': offset, 'limit': limit}

    def schema(self):
        connections = {}
        for edge in self.edges:
            key = (self.nodes[edge['source']]['type'], edge['type'], self.nodes[edge['target']]['type'])
            row = connections.setdefault(key, {'source_type': key[0], 'relation': key[1], 'target_type': key[2], 'count': 0, 'evidence_count': 0})
            row['count'] += 1
            row['evidence_count'] += edge.get('evidence_count', 1)
        return {'types': [{'type': g['type'], 'color': g['color']} for g in self.groups], 'connections': sorted(connections.values(), key=lambda row: (row['source_type'], row['target_type'], row['relation']))}

    def overview(self):
        grouped = defaultdict(list)
        for identifier, node in self.nodes.items():
            grouped[node['type']].append(identifier)
        selected = set()
        for identifiers in grouped.values():
            ordered = sorted(identifiers, key=lambda i: (-len(self.adjacency[i]), i))
            selected.update(ordered[:25])
            selected.update(sorted(identifiers, key=lambda i: hashlib.sha256(i.encode()).digest())[:155])
        selected_edges = [e for e in self.edges if e['source'] in selected and e['target'] in selected][:2200]
        return {'nodes': [self.compact(i) for i in sorted(selected)], 'links': selected_edges, 'groups': self.groups, 'schema': self.schema(), 'stats': {'nodes': len(self.nodes), 'edges': len(self.edges), 'source_records': len(self.records), 'source_relations': self.raw_edge_count, 'excluded_edges': self.missing_edges, 'hidden_nodes': len(self.hidden_nodes), 'hidden_edges': self.hidden_edges, 'overview_nodes': len(selected)}}

    def neighborhood(self, identifier, offset=0, limit=80, kind=''):
        identifier = self.resolve(identifier)
        if identifier not in self.nodes:
            raise KeyError(identifier)
        neighbors = defaultdict(list)
        for index in self.adjacency[identifier]:
            edge = self.edges[index]
            other = edge['target'] if edge['source'] == identifier else edge['source']
            if other != identifier:
                neighbors[other].append(index)
        type_counts = Counter((self.nodes[i]['type'] for i in neighbors))
        buckets = defaultdict(list)
        for i in sorted(neighbors, key=lambda i: (normalize(self.nodes[i]['name']), i)):
            if not kind or self.nodes[i]['type'] == kind:
                buckets[self.nodes[i]['type']].append(i)
        active = [iter(buckets[t]) for t in sorted(buckets)]
        ordered = []
        while active:
            remaining = []
            for bucket in active:
                item = next(bucket, None)
                if item is not None:
                    ordered.append(item)
                    remaining.append(bucket)
            active = remaining
        chosen = ordered[offset:offset + limit]
        selected = {identifier, *chosen}
        edges = {index for i in selected for index in self.adjacency[i] if self.edges[index]['source'] in selected and self.edges[index]['target'] in selected}
        return {'nodes': [self.compact(identifier)] + [self.compact(i) for i in chosen], 'links': [self.edges[i] for i in sorted(edges)], 'total_neighbors': len(ordered), 'offset': offset, 'limit': limit, 'has_more': offset + len(chosen) < len(ordered), 'center': identifier, 'neighbor_type_counts': dict(type_counts)}

    def detail(self, identifier, offset=0, limit=30, kind=''):
        identifier = self.resolve(identifier)
        node = self.nodes[identifier]
        neighbors = self.neighborhood(identifier, offset, limit, kind)
        relations = []
        for other in neighbors['nodes'][1:]:
            edges = [e for e in neighbors['links'] if {e['source'], e['target']} == {identifier, other['id']}]
            relations.append({'node': other, 'edges': edges})
        return {**node, 'composition_groups': self.composition_groups(identifier) if node['type'] == 'Mixture' else [], 'display_attributes': {k: v for k, v in node['attributes'].items() if has_attribute_value(v)}, 'degree': len(self.adjacency[identifier]), 'total_neighbors': neighbors['total_neighbors'], 'records': self.record_page(identifier)['records'], 'has_more_records': len(self.members[identifier]) > 20, 'neighbor_type_counts': neighbors['neighbor_type_counts'], 'neighbors': relations, 'offset': offset, 'has_more': neighbors['has_more'], 'limit': limit}

    def induced_links(self, identifiers):
        if not isinstance(identifiers, list) or len(identifiers) > 600 or (not all((isinstance(i, str) for i in identifiers))):
            raise ValueError('ids must be a list containing at most 600 node IDs')
        selected = {self.resolve(i) for i in identifiers}
        if any((i not in self.nodes for i in selected)):
            raise KeyError('Unknown node')
        indices = {index for i in selected for index in self.adjacency[i] if self.edges[index]['source'] in selected and self.edges[index]['target'] in selected}
        return {'links': [self.edges[i] for i in sorted(indices)]}
