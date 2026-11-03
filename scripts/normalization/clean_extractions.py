import argparse
import copy
import json
import os
import re
import signal
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple
from difflib import SequenceMatcher
INPUT_DIR = Path('data/inference')
OUTPUT_DIR = Path('results/normalized')
FUEL_DICTIONARY = Path(os.environ.get('FKG_FUEL_DICTIONARY', 'data/fuel_dictionary.json'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
SUBSCRIPT_MAP = str.maketrans({'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'})
SUPERSCRIPT_MAP = str.maketrans({'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'})
GREEK_MAP = str.maketrans({'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega', 'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta', 'Ε': 'Epsilon', 'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta', 'Ι': 'Iota', 'Κ': 'Kappa', 'Λ': 'Lambda', 'Μ': 'Mu', 'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron', 'Π': 'Pi', 'Ρ': 'Rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon', 'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega'})
ALLOWED_EXPERIMENT_TYPES = ['Shock Tube', 'Rapid Compression Machine', 'Advanced Fuel Ignition Delay Analyzer', 'Counterflow', 'Spherically Expanding Flames', 'Tube Channel Propagation', 'Bunsen Burner', 'Flat Flame Burner', 'Laminar Premixed Flame', 'Coflow', 'Jet-stirred Reactor', 'Plug Flow Reactor']
ALLOWED_RESULT_TYPES = ['Ignition delay time', 'Autoignition temperature', 'Extinction strain rate', 'Laminar flame speed', 'Species mole fraction vs. position', 'Species mole fraction vs. temperature', 'Species mole fraction vs. residence time', 'Species mole fraction vs. time', 'Temperature vs. position', 'Soot volume fraction', 'Soot number density', 'Particle size distribution', 'Primary particle diameter']
ALLOWED_CONDITION_TYPES = ['Pressure', 'Temperature', 'Fuel composition', 'Oxidizer composition', 'Equivalence ratio', 'Dilution gas', 'Separation distance', 'Inlet velocity', 'Inlet volumetric flow rate', 'Inlet mass flow rate', 'Reactor volume', 'Residence time']
TAXONOMY_ALIAS_MAP = {'shocktube': 'Shock Tube', 'shock tube': 'Shock Tube', 'shock-tube': 'Shock Tube', 'st': 'Shock Tube', 'rcm': 'Rapid Compression Machine', 'afida': 'Advanced Fuel Ignition Delay Analyzer', 'ignition delay analyzer': 'Advanced Fuel Ignition Delay Analyzer', 'jsr': 'Jet-stirred Reactor', 'jet stirred reactor': 'Jet-stirred Reactor', 'perfectly stirred reactor': 'Jet-stirred Reactor', 'psr': 'Jet-stirred Reactor', 'pfr': 'Plug Flow Reactor', 'plug flow reactor': 'Plug Flow Reactor', 'plug-flow reactor': 'Plug Flow Reactor', 'flow reactor': 'Plug Flow Reactor', 'idt': 'Ignition delay time', 'ignition delay': 'Ignition delay time', 'ignition delay times': 'Ignition delay time', 'autoignition delay time': 'Ignition delay time', 'ait': 'Autoignition temperature', 'lbv': 'Laminar flame speed', 'laminar burning velocity': 'Laminar flame speed', 'burning velocity': 'Laminar flame speed', 'laminar flame speeds': 'Laminar flame speed'}

def clean_string(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(GREEK_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    text = re.sub('[‐-‒–—−]', '-', text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = re.sub('[\\u200b\\u200c\\u200d\\ufeff]', '', text)
    text = re.sub('\\s+', ' ', text).strip()
    text = re.sub('\\s*-\\s*', '-', text)
    text = re.sub('\\s*/\\s*', '/', text)
    text = re.sub(',{2,}', ',', text)
    text = re.sub(';{2,}', ';', text)
    text = text.strip()
    return text

def clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_string(value)
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            clean_k = clean_string(k) if isinstance(k, str) else k
            cleaned[clean_k] = clean_value(v)
        return cleaned
    return value

def make_hashable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def node_key_without_id(node: Dict[str, Any]) -> str:
    node_copy = copy.deepcopy(node)
    node_copy.pop('node_id', None)
    return make_hashable(node_copy)

def relation_key(relation: Dict[str, Any]) -> str:
    return make_hashable(relation)

def deduplicate_nodes(nodes):
    seen = {}
    new_nodes = []
    node_id_mapping = {}
    removed_count = 0
    for node in nodes:
        node_id = node.get('node_id')
        key = node_key_without_id(node)
        if key in seen:
            kept_node_id = seen[key]
            if node_id is not None:
                node_id_mapping[node_id] = kept_node_id
            removed_count += 1
        else:
            seen[key] = node_id
            new_nodes.append(node)
    return (new_nodes, node_id_mapping, removed_count)
SOURCE_KEYS = ('source_id', 'source')
TARGET_KEYS = ('target_id', 'target')

def get_endpoint(relation, keys):
    for key in keys:
        if key in relation:
            return (key, relation[key])
    return (None, None)

def remap_relation_node_ids(relations, node_id_mapping):
    new_relations = []
    for rel in relations:
        rel = copy.deepcopy(rel)
        for key in SOURCE_KEYS:
            if key in rel and rel[key] in node_id_mapping:
                rel[key] = node_id_mapping[rel[key]]
        for key in TARGET_KEYS:
            if key in rel and rel[key] in node_id_mapping:
                rel[key] = node_id_mapping[rel[key]]
        new_relations.append(rel)
    return new_relations

def remove_invalid_relations(relations, valid_node_ids):
    valid_relations = []
    removed_count = 0
    for rel in relations:
        _, source_id = get_endpoint(rel, SOURCE_KEYS)
        _, target_id = get_endpoint(rel, TARGET_KEYS)
        if source_id in valid_node_ids and target_id in valid_node_ids:
            valid_relations.append(rel)
        else:
            removed_count += 1
    return (valid_relations, removed_count)

def deduplicate_relations(relations):
    seen = set()
    new_relations = []
    removed_count = 0
    for rel in relations:
        key = relation_key(rel)
        if key not in seen:
            seen.add(key)
            new_relations.append(rel)
        else:
            removed_count += 1
    return (new_relations, removed_count)

def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub('\\s*```$', '', text)
    return text.strip()

def parse_prediction(prediction: Any) -> Dict[str, Any]:
    if isinstance(prediction, dict):
        return prediction
    if not isinstance(prediction, str):
        raise ValueError(f'prediction is {type(prediction).__name__}, not str/dict')
    text = strip_code_fence(prediction)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and (end > start):
            return json.loads(text[start:end + 1])
        raise

def normalize_label_key(text: str) -> str:
    text = clean_string(text)
    text = text.lower()
    text = text.replace('vs.', 'vs')
    text = re.sub('[^a-z0-9]+', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    tokens = []
    for tok in text.split():
        if tok not in {'species', 'vs', 'gas'} and len(tok) > 3 and tok.endswith('s'):
            tok = tok[:-1]
        tokens.append(tok)
    return ' '.join(tokens)

def build_allowed_index(allowed_values):
    index = {}
    for value in allowed_values:
        index[normalize_label_key(value)] = value
    return index

def fuzzy_canonicalize_label(value, allowed_values, threshold=0.88, min_gap=0.04):
    if not isinstance(value, str):
        return value
    key = normalize_label_key(value)
    index = build_allowed_index(allowed_values)
    if key in index:
        return index[key]
    if key in TAXONOMY_ALIAS_MAP:
        candidate = TAXONOMY_ALIAS_MAP[key]
        if candidate in allowed_values:
            return candidate
    scored = []
    for allowed in allowed_values:
        allowed_key = normalize_label_key(allowed)
        ratio = SequenceMatcher(None, key, allowed_key).ratio()
        q_tokens = set(key.split())
        a_tokens = set(allowed_key.split())
        if q_tokens and a_tokens:
            overlap = len(q_tokens & a_tokens)
            coverage = overlap / len(q_tokens)
            jaccard = overlap / len(q_tokens | a_tokens)
        else:
            coverage = 0.0
            jaccard = 0.0
        score = max(ratio, 0.55 * coverage + 0.45 * jaccard)
        scored.append((score, allowed))
    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_value = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= threshold and best_score - second_score >= min_gap:
        return best_value
    return value

def normalize_taxonomy_fields(data):
    for node in data.get('nodes', []):
        if not isinstance(node, dict):
            continue
        node_type = node.get('node_type')
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        if node_type == 'Experiment':
            attrs['exp_type'] = fuzzy_canonicalize_label(attrs.get('exp_type'), ALLOWED_EXPERIMENT_TYPES)
        elif node_type == 'ExpResult':
            attrs['result_type'] = fuzzy_canonicalize_label(attrs.get('result_type'), ALLOWED_RESULT_TYPES)
        elif node_type == 'Condition':
            attrs['condition_type'] = fuzzy_canonicalize_label(attrs.get('condition_type'), ALLOWED_CONDITION_TYPES)
    return data
UNIT_MAP = {'K': 'K', 'k': 'K', 'kelvin': 'K', 'Kelvin': 'K', '°C': '°C', 'C': '°C', 'c': '°C', 'Celsius': '°C', 'celsius': '°C', 'bar': 'bar', 'atm': 'atm', 'MPa': 'MPa', 'mpa': 'MPa', 'kPa': 'kPa', 'kpa': 'kPa', 'mbar': 'mbar', 's': 's', 'sec': 's', 'second': 's', 'seconds': 's', 'ms': 'ms', 'm/s': 'm/s', 'cm/s': 'cm/s', 'mm/s': 'mm/s', 'mm': 'mm', 'cm': 'cm', 'm': 'm', 'L': 'L', 'l': 'L', 'liter': 'L', 'litre': 'L', 'L/min': 'L/min', 'l/min': 'L/min', 'lpm': 'L/min', 'LPM': 'L/min', 'SLM': 'SLM', 'slm': 'SLM', 'NLPH': 'NLPH', 'nlph': 'NLPH', 'g/min': 'g/min', 'g/h': 'g/h', 'Nl/h': 'NLPH', 'nl/h': 'NLPH', 'Nlh': 'NLPH', 'mL/min': 'mL/min', 'ml/min': 'mL/min', 'mL (STP)/min': 'mL(STP)/min', 'ml (STP)/min': 'mL(STP)/min', 'cm3/s': 'cm3/s', 'sccm': 'sccm', 'standard liter per minute': 'SLM', 'standard litre per minute': 'SLM', 'L·min-1 (STP)': 'SLM', 'L/min (STP)': 'SLM', 'slpm': 'SLM', 'Nm3·h-1': 'Nm3/h', 'cm3': 'cm3', 'vol%': 'vol%', 'vol.%': 'vol%', 'vol %': 'vol%', 'mol%': 'mol%', 'mol.%': 'mol%', 'mole fraction': 'mole fraction', '%': '%', 'rpm': 'rpm', 'CAD': 'CAD', 'cad': 'CAD', '1/mus': '1/μs', '1/mu s': '1/μs', '1/μs': '1/μs', '1/us': '1/μs'}

def normalize_unit_string(unit):
    if not isinstance(unit, str):
        return unit
    unit = clean_string(unit)
    unit = re.sub('\\s+', ' ', unit).strip()
    return UNIT_MAP.get(unit, unit)

def normalize_unit_value(value):
    if isinstance(value, str):
        return normalize_unit_string(value)
    if isinstance(value, list):
        return [normalize_unit_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_unit_value(v) for k, v in value.items()}
    return value

def normalize_units_in_obj(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            key_norm = str(k).strip().lower()
            if key_norm in {'unit', 'units'}:
                new_obj[k] = normalize_unit_value(v)
            else:
                new_obj[k] = normalize_units_in_obj(v)
        return new_obj
    if isinstance(obj, list):
        return [normalize_units_in_obj(v) for v in obj]
    return obj

def normalize_fuel_key(x):
    if x is None:
        return None
    x = clean_string(str(x))
    x = x.strip().lower()
    x = re.sub('\\s+', ' ', x)
    return x if x else None

def build_fuel_alias_map(dictionary_path):
    if not dictionary_path.exists():
        raise FileNotFoundError('Supply a fuel dictionary with --fuel_dictionary PATH.')
    with dictionary_path.open('r', encoding='utf-8') as f:
        dictionary = json.load(f)
    alias_map = {}
    conflict_keys = set()
    for entry in dictionary:
        canonical = normalize_fuel_key(entry.get('canonical_name'))
        if not canonical:
            continue
        names = set(entry.get('names', []))
        names.add(entry.get('canonical_name'))
        for name in names:
            key = normalize_fuel_key(name)
            if not key:
                continue
            if key not in alias_map:
                alias_map[key] = canonical
            elif alias_map[key] != canonical:
                conflict_keys.add(key)
    for key in conflict_keys:
        alias_map.pop(key, None)
    return alias_map

def canonicalize_fuel_name(name, fuel_alias_map=None):
    key = normalize_fuel_key(name)
    if not key:
        return None
    if fuel_alias_map and key in fuel_alias_map:
        return fuel_alias_map[key]
    return key

def format_percentage(number):
    if abs(number - round(number)) < 1e-08:
        return f'{int(round(number))}%'
    return f'{number:.6g}%'

def normalize_percentage_value(value, force_ratio=False):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = clean_string(str(value))
        text = text.replace('％', '%')
        m = re.search('[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', text)
        if not m:
            return None
        number = float(m.group(0))
        if '%' in text or 'percent' in text.lower():
            force_ratio = False
    if not force_ratio and 0 < number <= 1:
        number *= 100
    return format_percentage(number)

def split_composition_text(text):
    text = clean_string(text)
    text = text.replace('；', ';')
    text = re.sub('\\s+(and|with)\\s+', ',', text, flags=re.IGNORECASE)
    return [p.strip() for p in re.split('[,;]', text) if p.strip()]

def parse_name_value_pair(part):
    part = clean_string(part)
    m = re.match('^(.+?)\\s*[:=]\\s*(.+)$', part)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    m = re.match('^([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*%?\\s+(.+)$', part)
    if m:
        return (m.group(2).strip(), m.group(1).strip() + ('%' if '%' in part else ''))
    m = re.match('^(.+?)\\s+([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*%?$', part)
    if m:
        return (m.group(1).strip(), m.group(2).strip() + ('%' if '%' in part else ''))
    return (None, None)

def normalize_composition_mapping(items, fuel_alias_map=None):
    cleaned = []
    for raw_name, raw_value in items:
        name = canonicalize_fuel_name(raw_name, fuel_alias_map)
        if not name:
            continue
        if raw_value is None:
            continue
        raw_text = str(raw_value)
        m = re.search('[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', raw_text)
        if not m:
            continue
        number = float(m.group(0))
        has_percent = '%' in raw_text or 'percent' in raw_text.lower()
        cleaned.append((name, number, has_percent))
    if not cleaned:
        return {}
    total = sum((x[1] for x in cleaned))
    has_any_percent = any((x[2] for x in cleaned))
    output = {}
    for name, number, has_percent in cleaned:
        if has_any_percent:
            percentage = number
        elif 0 < total <= 1.000001:
            percentage = number * 100
        elif abs(total - 100) < 1e-06:
            percentage = number
        else:
            percentage = number / total * 100 if total else number
        output[name] = format_percentage(percentage)
    return output

def parse_slash_composition_string(text, fuel_alias_map=None):
    text = clean_string(text)
    m = re.search('(.+?)(?:=|\\(|\\s)\\s*((?:\\d*\\.?\\d+\\s*/\\s*)+\\d*\\.?\\d+)\\)?$', text)
    if not m:
        return {}
    name_part = m.group(1).strip()
    number_part = m.group(2).strip()
    names = [x.strip() for x in name_part.split('/') if x.strip()]
    numbers = [x.strip() for x in number_part.split('/') if x.strip()]
    if len(names) != len(numbers) or len(names) < 2:
        return {}
    return normalize_composition_mapping(list(zip(names, numbers)), fuel_alias_map)

def normalize_composition(composition, fuel_alias_map=None):
    if composition is None:
        return composition
    if isinstance(composition, dict):
        if 'name' in composition and any((k in composition for k in ['percentage', 'percent', 'fraction', 'ratio', 'value'])):
            name = composition.get('name')
            value = None
            for key in ['percentage', 'percent', 'fraction', 'ratio', 'value']:
                if key in composition:
                    value = composition.get(key)
                    break
            parsed = normalize_composition_mapping([(name, value)], fuel_alias_map)
            return parsed if parsed else composition
        items = []
        for k, v in composition.items():
            if k in {'unit', 'units'}:
                continue
            items.append((k, v))
        parsed = normalize_composition_mapping(items, fuel_alias_map)
        return parsed if parsed else composition
    if isinstance(composition, list):
        items = []
        for item in composition:
            if isinstance(item, dict):
                if 'name' in item:
                    name = item.get('name')
                    value = None
                    for key in ['percentage', 'percent', 'fraction', 'ratio', 'value']:
                        if key in item:
                            value = item.get(key)
                            break
                    items.append((name, value))
                else:
                    for k, v in item.items():
                        if k not in {'unit', 'units'}:
                            items.append((k, v))
            elif isinstance(item, str):
                name, value = parse_name_value_pair(item)
                if name is not None:
                    items.append((name, value))
        parsed = normalize_composition_mapping(items, fuel_alias_map)
        return parsed if parsed else composition
    if isinstance(composition, str):
        parsed = parse_slash_composition_string(composition, fuel_alias_map)
        if parsed:
            return parsed
        items = []
        for part in split_composition_text(composition):
            name, value = parse_name_value_pair(part)
            if name is not None:
                items.append((name, value))
        parsed = normalize_composition_mapping(items, fuel_alias_map)
        return parsed if parsed else composition
    return composition

def normalize_mixedfuel_compositions(data, fuel_alias_map=None):
    for node in data.get('nodes', []):
        if not isinstance(node, dict):
            continue
        if node.get('node_type') != 'MixedFuel':
            continue
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        if 'composition' in attrs:
            attrs['composition'] = normalize_composition(attrs.get('composition'), fuel_alias_map=fuel_alias_map)
    return data
AIR_PATTERN = re.compile('\\b(?:air|compressed\\s+air|dry\\s+air|preheated\\s+air|standard\\s+air|synthetic\\s+air|zero\\s+air|oxygen[-\\s]*enriched\\s+air|vitiated\\s+air)\\b', re.IGNORECASE)
GAS_KEYWORD_PATTERNS = [('O2', re.compile('(?<![A-Za-z0-9])(?:O2|O₂|oxygen|dioxygen)(?![A-Za-z0-9])', re.IGNORECASE)), ('N2O', re.compile('(?<![A-Za-z0-9])(?:N2O|N₂O|nitrous\\s+oxide|dinitrogen\\s+monoxide|laughing\\s+gas)(?![A-Za-z0-9])', re.IGNORECASE)), ('O3', re.compile('(?<![A-Za-z0-9])(?:O3|O₃|ozone)(?![A-Za-z0-9])', re.IGNORECASE)), ('H2O2', re.compile('(?<![A-Za-z0-9])(?:H2O2|H₂O₂|hydrogen\\s+peroxide)(?![A-Za-z0-9])', re.IGNORECASE)), ('NO2', re.compile('(?<![A-Za-z0-9])(?:NO2|NO₂|nitrogen\\s+dioxide)(?![A-Za-z0-9])', re.IGNORECASE)), ('NO', re.compile('(?<![A-Za-z0-9])NO(?![A-Za-z0-9])|(?i:(?<![A-Za-z0-9])nitric\\s+oxide(?![A-Za-z0-9]))')), ('NOx', re.compile('(?<![A-Za-z0-9])(?:NOx|NOₓ|nitrogen\\s+oxides?)(?![A-Za-z0-9])', re.IGNORECASE)), ('Cl2', re.compile('(?<![A-Za-z0-9])(?:Cl2|Cl₂|chlorine)(?![A-Za-z0-9])', re.IGNORECASE)), ('F2', re.compile('(?<![A-Za-z0-9])(?:F2|F₂|fluorine)(?![A-Za-z0-9])', re.IGNORECASE)), ('N2', re.compile('(?<![A-Za-z0-9])(?:N2|N₂|nitrogen)(?![A-Za-z0-9])', re.IGNORECASE)), ('Ar', re.compile('(?<![A-Za-z0-9])(?:Ar|argon)(?![A-Za-z0-9])', re.IGNORECASE)), ('He', re.compile('(?<![A-Za-z0-9])(?:He|helium)(?![A-Za-z0-9])', re.IGNORECASE)), ('Ne', re.compile('(?<![A-Za-z0-9])(?:Ne|neon)(?![A-Za-z0-9])', re.IGNORECASE)), ('Kr', re.compile('(?<![A-Za-z0-9])(?:Kr|krypton)(?![A-Za-z0-9])', re.IGNORECASE)), ('Xe', re.compile('(?<![A-Za-z0-9])(?:Xe|xenon)(?![A-Za-z0-9])', re.IGNORECASE)), ('CO2', re.compile('(?<![A-Za-z0-9])(?:CO2|CO₂|carbon\\s+dioxide)(?![A-Za-z0-9])', re.IGNORECASE)), ('CO', re.compile('(?<![A-Za-z0-9])(?:CO|carbon\\s+monoxide)(?![A-Za-z0-9])', re.IGNORECASE)), ('H2O', re.compile('(?<![A-Za-z0-9])(?:H2O|H₂O|water|steam|water\\s+vapor|water\\s+vapou?r)(?![A-Za-z0-9])', re.IGNORECASE)), ('SO2', re.compile('(?<![A-Za-z0-9])(?:SO2|SO₂|sulfur\\s+dioxide|sulphur\\s+dioxide)(?![A-Za-z0-9])', re.IGNORECASE)), ('SF6', re.compile('(?<![A-Za-z0-9])(?:SF6|SF₆|sulfur\\s+hexafluoride|sulphur\\s+hexafluoride)(?![A-Za-z0-9])', re.IGNORECASE))]
ALLOWED_OXIDIZER_GASES = {'air', 'O2', 'N2O', 'O3', 'H2O2', 'NO', 'NO2', 'NOx', 'Cl2', 'F2', 'N2', 'Ar', 'He', 'Ne', 'Kr', 'Xe', 'CO2', 'CO', 'H2O', 'SO2', 'SF6'}
ALLOWED_DILUTION_GASES = {'N2', 'Ar', 'He', 'Ne', 'Kr', 'Xe', 'CO2', 'CO', 'H2O', 'SO2', 'SF6'}
GAS_OUTPUT_ORDER = ['air', 'O2', 'N2O', 'O3', 'H2O2', 'NO', 'NO2', 'NOx', 'Cl2', 'F2', 'N2', 'Ar', 'He', 'Ne', 'Kr', 'Xe', 'CO2', 'CO', 'H2O', 'SO2', 'SF6']
FUEL_LIKE_GAS_PATTERNS = [re.compile('(?<![A-Za-z0-9])(?:H2|hydrogen)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:CH4|methane)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:C2H4|ethylene)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:C2H6|ethane)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:C3H8|propane)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:NH3|ammonia)(?![A-Za-z0-9])', re.IGNORECASE), re.compile('(?<![A-Za-z0-9])(?:DME|DMM)(?![A-Za-z0-9])', re.IGNORECASE)]
VALUE_UNIT_PATTERNS = [('mL(STP)/min', re.compile('\\bml\\s*\\(\\s*STP\\s*\\)\\s*/\\s*min\\b', re.IGNORECASE)), ('SLM', re.compile('\\b(?:standard\\s+lit(?:er|re)\\s+per\\s+minute|L\\s*[·.]?\\s*min-?1\\s*\\(\\s*STP\\s*\\)|slpm|SLM)\\b', re.IGNORECASE)), ('Nm3/h', re.compile('\\bNm\\s*3\\s*[·/]?\\s*h-?1\\b', re.IGNORECASE)), ('NLPH', re.compile('\\b(?:Nl|NL|nl)\\s*/\\s*h\\b', re.IGNORECASE)), ('L/min', re.compile('\\b(?:L\\s*/\\s*min|l\\s*/\\s*min|lpm|LPM)\\b', re.IGNORECASE)), ('mL/min', re.compile('\\b(?:mL|ml)\\s*/\\s*min\\b', re.IGNORECASE)), ('cm3/s', re.compile('\\bcm\\s*3\\s*/\\s*s\\b', re.IGNORECASE)), ('sccm', re.compile('\\bsccm\\b', re.IGNORECASE)), ('g/min', re.compile('\\bg\\s*/\\s*min\\b', re.IGNORECASE)), ('g/h', re.compile('\\bg\\s*/\\s*h\\b', re.IGNORECASE)), ('MPa', re.compile('\\bMPa\\b', re.IGNORECASE)), ('kPa', re.compile('\\bkPa\\b', re.IGNORECASE)), ('mbar', re.compile('\\bmbar\\b', re.IGNORECASE)), ('bar', re.compile('\\bbar\\b', re.IGNORECASE)), ('atm', re.compile('\\batm\\b', re.IGNORECASE)), ('cm/s', re.compile('\\bcm\\s*/\\s*s\\b', re.IGNORECASE)), ('mm/s', re.compile('\\bmm\\s*/\\s*s\\b', re.IGNORECASE)), ('m/s', re.compile('\\bm\\s*/\\s*s\\b', re.IGNORECASE)), ('°C', re.compile('(?:°\\s*C|\\bCelsius\\b|(?<=\\d)\\s*C\\b)', re.IGNORECASE)), ('K', re.compile('(?<=\\d)\\s*K\\b|\\bKelvin\\b', re.IGNORECASE)), ('ms', re.compile('\\bms\\b', re.IGNORECASE)), ('s', re.compile('\\bs\\b|\\bsec(?:ond|onds)?\\b', re.IGNORECASE)), ('cm3', re.compile('\\bcm\\s*3\\b', re.IGNORECASE)), ('mm', re.compile('\\bmm\\b', re.IGNORECASE)), ('cm', re.compile('\\bcm\\b', re.IGNORECASE)), ('L', re.compile('\\b(?:L|liter|litre)\\b', re.IGNORECASE))]
EXPECTED_UNITS_BY_CONDITION = {'Pressure': {'bar', 'atm', 'MPa', 'kPa', 'mbar'}, 'Temperature': {'K', '°C'}, 'Separation distance': {'mm', 'cm', 'm'}, 'Inlet velocity': {'m/s', 'cm/s', 'mm/s'}, 'Inlet volumetric flow rate': {'L/min', 'SLM', 'NLPH', 'mL/min', 'mL(STP)/min', 'cm3/s', 'sccm', 'Nm3/h'}, 'Inlet mass flow rate': {'g/min', 'g/h'}, 'Reactor volume': {'L', 'cm3'}, 'Residence time': {'s', 'ms'}}

def ensure_string_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    out = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, (dict, list)):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = clean_string(str(item))
        text = re.sub('\\s+', ' ', text).strip()
        if text:
            out.append(text)
    return out

def looks_like_numeric_or_range(text):
    text = clean_string(text)
    text = text.replace('~', '').replace('∼', '').replace('≈', '')
    text = text.replace('±', '')
    if re.search('[A-Za-z]', text):
        return False
    return bool(re.search('[-+]?\\d*\\.?\\d+', text))

def normalize_numeric_text(text):
    text = clean_string(text)
    text = text.replace('∼', '~')
    text = text.replace('−', '-')
    text = re.sub('\\s*[-–—]\\s*', '-', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text if text else None

def detect_units_in_value(text):
    detected = []
    for unit, pattern in VALUE_UNIT_PATTERNS:
        if pattern.search(text):
            detected.append((normalize_unit_string(unit), pattern))
    return detected

def strip_detected_unit(text, pattern):
    text = pattern.sub('', text)
    text = re.sub('\\s+', ' ', text)
    text = re.sub('\\s*,\\s*', ', ', text)
    text = re.sub('\\s*/\\s*$', '', text)
    return text.strip(' ,;:=')

def remove_units_from_condition_values(values, current_unit, condition_type, stats):
    unit = normalize_unit_string(current_unit) if current_unit else None
    expected_units = EXPECTED_UNITS_BY_CONDITION.get(condition_type)
    new_values = []
    if expected_units is None:
        return (values, unit)
    for value in values:
        text = clean_string(value)
        detected = detect_units_in_value(text)
        if not detected:
            new_values.append(text)
            continue
        compatible = []
        for detected_unit, pattern in detected:
            if expected_units is None or detected_unit in expected_units:
                compatible.append((detected_unit, pattern))
        if not compatible:
            stats['condition_values_removed_unit_mismatch'] += 1
            continue
        detected_unit, pattern = compatible[0]
        if unit is None:
            unit = detected_unit
        elif normalize_unit_string(unit) != detected_unit:
            stats['condition_values_removed_unit_mismatch'] += 1
            continue
        stripped = strip_detected_unit(text, pattern)
        if stripped:
            new_values.append(stripped)
            stats['condition_units_moved_from_value'] += 1
    return (new_values, unit)

def contains_fuel_like_gas(text):
    return any((pattern.search(text) for pattern in FUEL_LIKE_GAS_PATTERNS))

def detect_standard_gases(text):
    text = clean_string(text)
    gases = []
    if AIR_PATTERN.search(text):
        gases.append('air')
    for gas, pattern in GAS_KEYWORD_PATTERNS:
        if pattern.search(text):
            gases.append(gas)
    seen = set()
    out = []
    for gas in GAS_OUTPUT_ORDER:
        if gas in gases and gas not in seen:
            seen.add(gas)
            out.append(gas)
    return out

def normalize_oxidizer_value(text):
    text = clean_string(text)
    if AIR_PATTERN.search(text):
        return 'air'
    gases = [g for g in detect_standard_gases(text) if g in ALLOWED_OXIDIZER_GASES and g != 'air']
    if not gases:
        return None
    gases = [g for g in GAS_OUTPUT_ORDER if g in gases and g != 'air']
    return '/'.join(gases) if gases else None

def normalize_dilution_value(text):
    gases = [g for g in detect_standard_gases(text) if g in ALLOWED_DILUTION_GASES]
    gases = [g for g in GAS_OUTPUT_ORDER if g in gases and g in ALLOWED_DILUTION_GASES]
    return '/'.join(gases) if gases else None

def clean_condition_values_for_node(attrs, stats):
    condition_type = attrs.get('condition_type')
    values = ensure_string_list(attrs.get('value'))
    unit = normalize_unit_string(attrs.get('unit')) if attrs.get('unit') else None
    expected_units = EXPECTED_UNITS_BY_CONDITION.get(condition_type)
    if unit is not None and expected_units is not None and (unit not in expected_units):
        stats['condition_values_removed_unit_mismatch'] += len(values) if values else 1
        return False
    if condition_type != 'Fuel composition':
        values, unit = remove_units_from_condition_values(values, unit, condition_type, stats)
    cleaned_values = []
    for value in values:
        text = clean_string(value)
        lower = text.lower()
        if condition_type == 'Pressure':
            if 'atmospheric' in lower or 'near-atmospheric' in lower or lower in {'ambient', 'ambient pressure', 'atmospheric pressure'}:
                cleaned_values.append('atmospheric')
                unit = None
            elif looks_like_numeric_or_range(text):
                v = normalize_numeric_text(text)
                if v:
                    cleaned_values.append(v)
        elif condition_type == 'Temperature':
            if lower in {'room', 'room temperature', 'ambient', 'ambient temperature'}:
                cleaned_values.append('room temperature')
                unit = None
            elif looks_like_numeric_or_range(text):
                v = normalize_numeric_text(text)
                if v:
                    cleaned_values.append(v)
        elif condition_type == 'Equivalence ratio':
            unit = None
            m = re.search('(?:phi|Φ|φ)\\s*=\\s*([-+]?\\d*\\.?\\d+)', text, flags=re.IGNORECASE)
            if m:
                cleaned_values.append(m.group(1))
            elif lower == 'varied':
                stats['condition_values_removed_invalid'] += 1
            elif looks_like_numeric_or_range(text):
                v = normalize_numeric_text(text)
                if v:
                    cleaned_values.append(v)
        elif condition_type == 'Oxidizer composition':
            unit = None
            v = normalize_oxidizer_value(text)
            if v:
                cleaned_values.append(v)
            else:
                stats['condition_values_removed_invalid'] += 1
        elif condition_type == 'Dilution gas':
            unit = None
            v = normalize_dilution_value(text)
            if v:
                cleaned_values.append(v)
            else:
                stats['condition_values_removed_invalid'] += 1
        elif condition_type in {'Separation distance', 'Inlet velocity', 'Inlet volumetric flow rate', 'Inlet mass flow rate', 'Reactor volume', 'Residence time'}:
            if re.search('[A-Za-z]', text):
                stats['condition_values_removed_invalid'] += 1
                continue
            if looks_like_numeric_or_range(text):
                v = normalize_numeric_text(text)
                if v:
                    cleaned_values.append(v)
        elif condition_type == 'Fuel composition':
            if text:
                cleaned_values.append(text)
        elif text:
            cleaned_values.append(text)
    deduped = []
    seen = set()
    for value in cleaned_values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    attrs['value'] = deduped
    attrs['unit'] = unit
    return bool(deduped)

def clean_condition_nodes(nodes, stats):
    new_nodes = []
    removed_count = 0
    for node in nodes:
        if not isinstance(node, dict) or node.get('node_type') != 'Condition':
            new_nodes.append(node)
            continue
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            removed_count += 1
            continue
        keep = clean_condition_values_for_node(attrs, stats)
        if keep:
            new_nodes.append(node)
        else:
            removed_count += 1
    return (new_nodes, removed_count)

def remove_off_taxonomy_nodes(nodes):
    allowed_map = {'Experiment': ('exp_type', set(ALLOWED_EXPERIMENT_TYPES)), 'Condition': ('condition_type', set(ALLOWED_CONDITION_TYPES)), 'ExpResult': ('result_type', set(ALLOWED_RESULT_TYPES))}
    new_nodes = []
    removed_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            removed_count += 1
            continue
        node_type = node.get('node_type')
        if node_type not in allowed_map:
            new_nodes.append(node)
            continue
        attrs = node.get('attributes', {}) or {}
        field_name, allowed_values = allowed_map[node_type]
        value = attrs.get(field_name)
        if value in allowed_values:
            new_nodes.append(node)
        else:
            removed_count += 1
    return (new_nodes, removed_count)

def clean_fkg_extraction(data, fuel_alias_map=None):
    stats = {'nodes_removed_duplicate': 0, 'nodes_removed_off_taxonomy': 0, 'relations_removed_invalid': 0, 'relations_removed_duplicate': 0, 'nodes_removed_empty_condition_value': 0, 'condition_units_moved_from_value': 0, 'condition_values_removed_unit_mismatch': 0, 'condition_values_removed_invalid': 0}
    data = copy.deepcopy(data)
    data = clean_value(data)
    data = normalize_units_in_obj(data)
    data = normalize_taxonomy_fields(data)
    data = normalize_mixedfuel_compositions(data, fuel_alias_map=fuel_alias_map)
    nodes = data.get('nodes', [])
    relations = data.get('relations', [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(relations, list):
        relations = []
    nodes, removed_empty_condition = clean_condition_nodes(nodes, stats)
    stats['nodes_removed_empty_condition_value'] = removed_empty_condition
    nodes, removed_off_taxonomy = remove_off_taxonomy_nodes(nodes)
    stats['nodes_removed_off_taxonomy'] = removed_off_taxonomy
    nodes, node_id_mapping, removed_nodes = deduplicate_nodes(nodes)
    stats['nodes_removed_duplicate'] = removed_nodes
    relations = remap_relation_node_ids(relations, node_id_mapping)
    valid_node_ids = {node.get('node_id') for node in nodes if isinstance(node, dict) and node.get('node_id') is not None}
    relations, removed_invalid_relations = remove_invalid_relations(relations, valid_node_ids)
    stats['relations_removed_invalid'] = removed_invalid_relations
    relations, removed_duplicate_relations = deduplicate_relations(relations)
    stats['relations_removed_duplicate'] = removed_duplicate_relations
    data['nodes'] = nodes
    data['relations'] = relations
    return (data, stats)

def load_done_keys(out_file):
    done = set()
    if not out_file.exists():
        return done
    with out_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get('xml_file')
            if k:
                done.add(k)
    return done

def append_jsonl(path, item):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch_file', help='e.g. inference_000.jsonl')
    ap.add_argument('--input_dir', default=str(INPUT_DIR))
    ap.add_argument('--output_dir', default=str(OUTPUT_DIR))
    ap.add_argument('--fuel_dictionary', default=str(FUEL_DICTIONARY))
    return ap.parse_args()

def resolve_batch_path(arg, input_dir):
    p = Path(arg)
    if p.exists():
        return p.resolve()
    p2 = input_dir / arg
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f'Cannot find input file: {arg}')

def get_batch_suffix(batch_path):
    m = re.search('inference_(\\d+)', batch_path.stem)
    return m.group(1) if m else batch_path.stem

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fuel_dict_path = Path(args.fuel_dictionary)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = resolve_batch_path(args.batch_file, input_dir)
    batch_suffix = get_batch_suffix(input_path)
    out_file = output_dir / f'clean_{batch_suffix}.jsonl'
    print(f'Input file       : {input_path}')
    print(f'Output file      : {out_file}')
    print(f'Fuel dictionary  : {fuel_dict_path}')
    fuel_alias_map = build_fuel_alias_map(fuel_dict_path)
    print(f'Fuel alias entries: {len(fuel_alias_map)}')
    done_keys = load_done_keys(out_file)
    print(f'Already done     : {len(done_keys)}')

    def handle_signal(signum, frame):
        print(f'\nReceived signal {signum}, exiting cleanly.', flush=True)
        os._exit(128 + signum)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    with input_path.open('r', encoding='utf-8') as f:
        total_lines = sum((1 for line in f if line.strip()))
    print(f'Total records    : {total_lines}')
    total_stats = {'nodes_removed_duplicate': 0, 'nodes_removed_off_taxonomy': 0, 'relations_removed_invalid': 0, 'relations_removed_duplicate': 0, 'nodes_removed_empty_condition_value': 0, 'condition_units_moved_from_value': 0, 'condition_values_removed_unit_mismatch': 0, 'condition_values_removed_invalid': 0}
    n_skipped = 0
    n_success = 0
    n_failed = 0
    fh = open(input_path, 'r', encoding='utf-8')
    iterator = fh
    if tqdm is not None:
        iterator = tqdm(fh, total=total_lines, desc=f'Cleaning batch_{batch_suffix}')
    try:
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_failed += 1
                continue
            xml_file = row.get('xml_file')
            if xml_file and xml_file in done_keys:
                n_skipped += 1
                continue
            try:
                prediction_json = parse_prediction(row.get('prediction'))
                cleaned_prediction, stats = clean_fkg_extraction(prediction_json, fuel_alias_map=fuel_alias_map)
                row['prediction'] = json.dumps(cleaned_prediction, ensure_ascii=False, separators=(',', ':'))
                for key in total_stats:
                    total_stats[key] += stats[key]
                n_success += 1
            except Exception as e:
                row['cleaning_error'] = str(e)
                n_failed += 1
            append_jsonl(out_file, row)
    finally:
        fh.close()
    print(f'\nDone.')
    print(f'  Total    : {total_lines}')
    print(f'  Skipped  : {n_skipped} (resume)')
    print(f'  Success  : {n_success}')
    print(f'  Failed   : {n_failed}')
    print(f'  Output   : {out_file}')
    print('Cleaning stats:')
    for key, value in total_stats.items():
        print(f'  {key}: {value}')
if __name__ == '__main__':
    main()
