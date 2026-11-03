"""Build a fuel dictionary using PubChem CID and synonym matches."""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import quote
import requests
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path('data/inference')
OUTPUT_DIR = Path('results/normalization')
OUTPUT_FILE = OUTPUT_DIR / 'fuel_dictionary_base.json'
OUTPUT_LLM_CANDIDATES_FILE = OUTPUT_DIR / 'fuel_candidates.json'
CACHE_FILE = OUTPUT_DIR / 'pubchem_cache.json'
PUBCHEM_SLEEP = 0.2
PUBCHEM_TIMEOUT = 20
MAX_SYNONYMS_PER_QUERY = 300
SUBSCRIPT_MAP = str.maketrans({'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'})
SUPERSCRIPT_MAP = str.maketrans({'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'})

def clean_string(text):
    if text is None:
        return None
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    text = re.sub('[‐-‒–—−]', '-', text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = re.sub('[\\u200b\\u200c\\u200d\\ufeff]', '', text)
    text = re.sub('\\s+', ' ', text).strip()
    text = re.sub('\\s*-\\s*', '-', text)
    text = re.sub('\\s*/\\s*', '/', text)
    return text.strip() if text.strip() else None

def norm_key(text):
    text = clean_string(text)
    if not text:
        return None
    text = text.lower()
    text = re.sub('\\s+', ' ', text).strip()
    return text if text else None

def safe_parse_json(text):
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub('\\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and (end > start):
        return json.loads(text[start:end + 1])
    raise ValueError('Cannot parse prediction as JSON')

def load_fkg_objects_from_jsonl(path):
    objects = []
    parse_failed = 0
    with path.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                if 'prediction' in row:
                    obj = safe_parse_json(row['prediction'])
                elif 'nodes' in row and 'relations' in row:
                    obj = row
                else:
                    raise ValueError('No prediction or FKG structure')
                obj['_source_meta'] = {'jsonl_line': line_idx, 'id': row.get('id'), 'xml_file': row.get('xml_file') or obj.get('xml_file'), 'doi': row.get('doi') or obj.get('doi')}
                objects.append(obj)
            except Exception as e:
                parse_failed += 1
                print(f'[PARSE FAILED] line={line_idx}, error={e}')
    print(f'Loaded FKG objects: {len(objects)}')
    print(f'Parse failed:       {parse_failed}')
    return objects

def is_useful_pubchem_name(name):
    if not name:
        return False
    name = clean_string(name)
    if not name:
        return False
    low = name.lower()
    if len(name) > 120:
        return False
    banned_prefixes = ['inchi', 'inchikey', 'smiles']
    for prefix in banned_prefixes:
        if low.startswith(prefix):
            return False
    banned_patterns = ['^cid\\s*\\d+$', '^chebi[:\\s-]?\\d+$', '^dtxsid', '^dtxcid', '^unii[-\\s]', '^sc-?\\d+$', '^sc\\d+$', '^schembl\\d+$', '^akos', '^zinc\\d+', '^mfcd\\d+', '^nsc\\s*\\d+', '^brn\\s*\\d+', '^\\d{2,7}-\\d{2}-\\d$']
    for pattern in banned_patterns:
        if re.search(pattern, low):
            return False
    return True

def query_pubchem_by_name(name):
    encoded = quote(name, safe='')
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/synonyms/JSON'
    try:
        r = requests.get(url, timeout=PUBCHEM_TIMEOUT)
    except Exception:
        return {'status': 'failed', 'cid': None, 'synonyms': []}
    if r.status_code != 200:
        return {'status': 'failed', 'cid': None, 'synonyms': []}
    try:
        data = r.json()
    except Exception:
        return {'status': 'failed', 'cid': None, 'synonyms': []}
    infos = data.get('InformationList', {}).get('Information', [])
    if len(infos) != 1:
        return {'status': 'failed', 'cid': None, 'synonyms': []}
    info = infos[0]
    cid = info.get('CID')
    raw_synonyms = info.get('Synonym', []) or []
    synonyms = []
    for synonym in raw_synonyms[:MAX_SYNONYMS_PER_QUERY]:
        synonym = clean_string(synonym)
        if synonym and is_useful_pubchem_name(synonym):
            synonyms.append(synonym)
    return {'status': 'success', 'cid': cid, 'synonyms': sorted(set(synonyms), key=lambda x: norm_key(x) or x)}

def query_with_cache(name, query_cache):
    key = norm_key(name)
    if not key:
        return {'status': 'failed', 'cid': None, 'synonyms': []}
    if key in query_cache:
        return query_cache[key]
    result = query_pubchem_by_name(name)
    query_cache[key] = result
    time.sleep(PUBCHEM_SLEEP)
    return result

def extract_fuel_nodes(fkg_objects):
    fuel_nodes = []
    source_name_counter = Counter()
    for obj in fkg_objects:
        source_meta = obj.get('_source_meta', {}) or {}
        for node in obj.get('nodes', []):
            if node.get('node_type') != 'Fuel':
                continue
            attrs = node.get('attributes', {}) or {}
            name = clean_string(attrs.get('name'))
            if not name:
                continue
            aliases = attrs.get('aliases', [])
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list):
                aliases = [aliases]
            aliases = [clean_string(alias) for alias in aliases if clean_string(alias)]
            fuel_nodes.append({'name': name, 'aliases': aliases, 'node_id': node.get('node_id'), 'doi': source_meta.get('doi'), 'xml_file': source_meta.get('xml_file')})
            source_name_counter[name] += 1
    print(f'Fuel nodes with name: {len(fuel_nodes)}')
    print(f'Unique Fuel.name:     {len(source_name_counter)}')
    return (fuel_nodes, source_name_counter)

def build_set_from_fuel_node(fuel_node, query_cache):
    anchor_name = fuel_node['name']
    aliases = fuel_node.get('aliases', [])
    anchor_result = query_with_cache(anchor_name, query_cache)
    names = {anchor_name}
    source_names = {anchor_name}
    cids = set()
    if anchor_result.get('status') == 'success' and anchor_result.get('cid') is not None:
        anchor_cid = anchor_result['cid']
        cids.add(anchor_cid)
        for synonym in anchor_result.get('synonyms', []):
            names.add(synonym)
        for alias in aliases:
            alias_result = query_with_cache(alias, query_cache)
            if alias_result.get('status') == 'success' and alias_result.get('cid') == anchor_cid:
                names.add(alias)
                for synonym in alias_result.get('synonyms', []):
                    names.add(synonym)
        return {'names': names, 'source_names': source_names, 'cids': cids}
    cid_to_aliases = defaultdict(set)
    cid_to_synonyms = defaultdict(set)
    for alias in aliases:
        alias_result = query_with_cache(alias, query_cache)
        if alias_result.get('status') == 'success' and alias_result.get('cid') is not None:
            alias_cid = alias_result['cid']
            cid_to_aliases[alias_cid].add(alias)
            for synonym in alias_result.get('synonyms', []):
                cid_to_synonyms[alias_cid].add(synonym)
    valid_alias_cids = set(cid_to_aliases.keys())
    if len(valid_alias_cids) == 1:
        cid = next(iter(valid_alias_cids))
        cids.add(cid)
        names.update(cid_to_aliases[cid])
        names.update(cid_to_synonyms[cid])
        return {'names': names, 'source_names': source_names, 'cids': cids}
    return {'names': names, 'source_names': source_names, 'cids': set()}

def build_anchor_based_fuel_sets(fuel_nodes, query_cache=None, cache_file=None):
    if query_cache is None:
        query_cache = {}
    fuel_sets = []
    iterator = fuel_nodes
    if tqdm is not None:
        iterator = tqdm(fuel_nodes, desc='Building anchor-based Fuel sets')
    for i, fuel_node in enumerate(iterator):
        item = build_set_from_fuel_node(fuel_node, query_cache)
        fuel_sets.append(item)
        if cache_file is not None and (i + 1) % 500 == 0:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(query_cache, f, ensure_ascii=False, indent=2)
    print(f'Anchor-based Fuel sets: {len(fuel_sets)}')
    return fuel_sets

def looks_like_context_dependent_fuel(name, aliases=None):
    aliases = aliases or []
    texts = [name] + aliases
    texts = [clean_string(x) for x in texts if clean_string(x)]
    phrase_keywords = ['surrogate', 'blend', 'mixture', 'diesel', 'gasoline', 'kerosene', 'jet', 'biodiesel', 'biofuel', 'reference fuel', 'commercial fuel', 'real fuel', 'target fuel']
    exact_codes = {'prf', 'trf', 'face', 'dme', 'rd', 'hrj', 'hvo', 'fame'}
    for text in texts:
        low = text.lower()
        if any((k in low for k in phrase_keywords)):
            return True
        if low in exact_codes:
            return True
        if re.search('\\b[A-Za-z0-9][A-Za-z0-9-]*[/+&][A-Za-z0-9][A-Za-z0-9-]*\\b', text):
            return True
        if re.fullmatch('(?:[A-Z][a-z]?\\d*){2,}', text):
            return True
    return False

def build_llm_candidate_items(fuel_nodes, query_cache):
    candidates = []
    seen = set()
    for fuel_node in fuel_nodes:
        name = fuel_node['name']
        aliases = fuel_node.get('aliases', [])
        anchor_result = query_with_cache(name, query_cache)
        valid_alias_cids = set()
        conflict_aliases = []
        for alias in aliases:
            alias_result = query_with_cache(alias, query_cache)
            if alias_result.get('status') == 'success' and alias_result.get('cid') is not None:
                alias_cid = alias_result['cid']
                valid_alias_cids.add(alias_cid)
                if anchor_result.get('status') == 'success' and anchor_result.get('cid') is not None and (alias_cid != anchor_result.get('cid')):
                    conflict_aliases.append(alias)
        reason = None
        if conflict_aliases:
            reason = 'anchor_alias_cid_conflict'
        elif (anchor_result.get('status') != 'success' or anchor_result.get('cid') is None) and len(valid_alias_cids) > 1:
            reason = 'multiple_alias_cids'
        elif (anchor_result.get('status') != 'success' or anchor_result.get('cid') is None) and len(valid_alias_cids) == 0 and looks_like_context_dependent_fuel(name, aliases):
            reason = 'pubchem_failed_context_dependent_name'
        if reason is None:
            continue
        key = json.dumps({'name': norm_key(name), 'aliases': sorted([norm_key(a) for a in aliases if norm_key(a)]), 'doi': fuel_node.get('doi'), 'xml_file': fuel_node.get('xml_file'), 'reason': reason}, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({'raw_fuel_name': name, 'aliases': aliases, 'doi': fuel_node.get('doi'), 'xml_file': fuel_node.get('xml_file'), 'node_id': fuel_node.get('node_id'), 'reason': reason})
    return candidates

class DSU:

    def __init__(self, n, fuel_sets):
        self.parent = list(range(n))
        self.cids = [set(item.get('cids', set())) for item in fuel_sets]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def compatible(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        cids_a = self.cids[ra]
        cids_b = self.cids[rb]
        if not cids_a or not cids_b:
            return True
        return len(cids_a & cids_b) > 0

    def union_if_compatible(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True
        if not self.compatible(ra, rb):
            return False
        self.parent[rb] = ra
        self.cids[ra].update(self.cids[rb])
        return True

def merge_sets_by_intersection_with_cid_check(fuel_sets):
    dsu = DSU(len(fuel_sets), fuel_sets)
    name_to_indices = defaultdict(list)
    for i, item in enumerate(fuel_sets):
        for name in item.get('names', set()):
            key = norm_key(name)
            if key:
                name_to_indices[key].append(i)
    for indices in name_to_indices.values():
        if len(indices) <= 1:
            continue
        first = indices[0]
        for idx in indices[1:]:
            dsu.union_if_compatible(first, idx)
    groups = defaultdict(list)
    for i in range(len(fuel_sets)):
        groups[dsu.find(i)].append(i)
    merged_sets = []
    for indices in groups.values():
        names = set()
        source_names = set()
        cids = set()
        for idx in indices:
            item = fuel_sets[idx]
            names.update(item.get('names', set()))
            source_names.update(item.get('source_names', set()))
            cids.update(item.get('cids', set()))
        if len(cids) <= 1:
            merged_sets.append({'names': names, 'source_names': source_names, 'cids': cids})
        else:
            cid_to_names = defaultdict(set)
            no_cid_names = set()
            for idx in indices:
                item = fuel_sets[idx]
                item_cids = item.get('cids', set())
                if len(item_cids) == 1:
                    cid = next(iter(item_cids))
                    cid_to_names[cid].update(item.get('names', set()))
                else:
                    no_cid_names.update(item.get('names', set()))
            for cid, cid_names in cid_to_names.items():
                merged_sets.append({'names': cid_names, 'source_names': source_names & cid_names, 'cids': {cid}})
            for name in no_cid_names:
                merged_sets.append({'names': {name}, 'source_names': {name} if name in source_names else set(), 'cids': set()})
    print(f'Final merged Fuel sets: {len(merged_sets)}')
    return merged_sets

def choose_canonical_name(names, source_names, source_name_counter):
    source_candidates = [name for name in names if name in source_names]
    if source_candidates:
        source_candidates = sorted(source_candidates, key=lambda x: (-source_name_counter.get(x, 0), len(x), norm_key(x) or x))
        return source_candidates[0]
    names = sorted(names, key=lambda x: (len(x), norm_key(x) or x))
    return names[0] if names else None

def deduplicate_names_by_norm_key(names):
    key_to_name = {}
    for name in names:
        name = clean_string(name)
        if not name:
            continue
        key = norm_key(name)
        if not key:
            continue
        if key not in key_to_name:
            key_to_name[key] = name
        else:
            old = key_to_name[key]
            if len(name) < len(old):
                key_to_name[key] = name
    return set(key_to_name.values())

def build_minimal_dictionary(fuel_sets, source_name_counter):
    dictionary = []
    for item in fuel_sets:
        names = deduplicate_names_by_norm_key(item.get('names', set()))
        if not names:
            continue
        canonical_name = choose_canonical_name(names=names, source_names=item.get('source_names', set()), source_name_counter=source_name_counter)
        if not canonical_name:
            continue
        dictionary.append({'canonical_name': canonical_name, 'names': sorted(names, key=lambda x: norm_key(x) or x)})
    dictionary = sorted(dictionary, key=lambda x: norm_key(x['canonical_name']) or x['canonical_name'])
    return dictionary

def main():
    global INPUT_DIR, OUTPUT_DIR, OUTPUT_FILE, OUTPUT_LLM_CANDIDATES_FILE, CACHE_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--pattern', default='*.jsonl')
    args = parser.parse_args()
    INPUT_DIR, OUTPUT_DIR = (args.input_dir, args.output_dir)
    OUTPUT_FILE = OUTPUT_DIR / 'fuel_dictionary_base.json'
    OUTPUT_LLM_CANDIDATES_FILE = OUTPUT_DIR / 'fuel_candidates.json'
    CACHE_FILE = OUTPUT_DIR / 'pubchem_cache.json'
    if not INPUT_DIR.is_dir() or not list(INPUT_DIR.glob(args.pattern)):
        parser.error('No input JSONL files match the specified directory and pattern.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Input dir:            {INPUT_DIR}')
    print(f'Output dictionary:    {OUTPUT_FILE}')
    print(f'Output LLM candidates:{OUTPUT_LLM_CANDIDATES_FILE}')
    batch_files = sorted(INPUT_DIR.glob(args.pattern))
    print(f'Found {len(batch_files)} batch files')
    fkg_objects = []
    for bf in batch_files:
        print(f'  Loading {bf.name}...')
        objs = load_fkg_objects_from_jsonl(bf)
        fkg_objects.extend(objs)
    print(f'Total FKG objects loaded: {len(fkg_objects)}')
    fuel_nodes, source_name_counter = extract_fuel_nodes(fkg_objects)
    if CACHE_FILE.exists():
        with CACHE_FILE.open('r', encoding='utf-8') as f:
            query_cache = json.load(f)
        print(f'Loaded {len(query_cache)} cached PubChem queries')
    else:
        query_cache = {}
    fuel_sets = build_anchor_based_fuel_sets(fuel_nodes, query_cache=query_cache, cache_file=CACHE_FILE)
    llm_candidates = build_llm_candidate_items(fuel_nodes, query_cache=query_cache)
    with OUTPUT_LLM_CANDIDATES_FILE.open('w', encoding='utf-8') as f:
        json.dump(llm_candidates, f, ensure_ascii=False, indent=2)
    with CACHE_FILE.open('w', encoding='utf-8') as f:
        json.dump(query_cache, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(query_cache)} PubChem queries to cache')
    fuel_sets = merge_sets_by_intersection_with_cid_check(fuel_sets)
    dictionary = build_minimal_dictionary(fuel_sets, source_name_counter)
    with OUTPUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print('\n================ DONE ================')
    print(f'Dictionary entries: {len(dictionary)}')
    print(f'LLM candidates:     {len(llm_candidates)}')
    print(f'Saved dictionary to:{OUTPUT_FILE}')
    print(f'Saved LLM file to:  {OUTPUT_LLM_CANDIDATES_FILE}')
if __name__ == '__main__':
    main()
