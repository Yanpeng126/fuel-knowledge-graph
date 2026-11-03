"""Normalize extracted graph names and conditions."""
import argparse
import copy
import json
import os
import re
import signal
from pathlib import Path
from typing import Any, Dict, Optional
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
INPUT_DIR = Path('data/inference')
OUTPUT_DIR = Path('results/normalized')
FUEL_DICTIONARY = Path(os.environ.get('FKG_FUEL_DICTIONARY', 'data/fuel_dictionary.json'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_ALIAS_MAP = {'rcm': 'Rapid Compression Machine', 'rapid compression machine': 'Rapid Compression Machine', 'rapid compression machine experiment': 'Rapid Compression Machine', 'shock tube': 'Shock Tube', 'shock tube experiment': 'Shock Tube', 'st': 'Shock Tube', 'jet stirred reactor': 'Jet-stirred Reactor', 'jet-stirred reactor': 'Jet-stirred Reactor', 'jsr': 'Jet-stirred Reactor', 'perfectly stirred reactor': 'Jet-stirred Reactor', 'psr': 'Jet-stirred Reactor', 'plug flow reactor': 'Plug Flow Reactor', 'flow reactor': 'Plug Flow Reactor', 'pfr': 'Plug Flow Reactor', 'counterflow': 'Counterflow', 'counterflow burner': 'Counterflow', 'spherically expanding flames': 'Spherically Expanding Flames', 'spherical flame': 'Spherically Expanding Flames', 'bunsen burner': 'Bunsen Burner', 'flat flame burner': 'Flat Flame Burner', 'laminar premixed flame': 'Laminar Premixed Flame', 'coflow': 'Coflow', 'afida': 'Advanced Fuel Ignition Delay Analyzer', 'advanced fuel ignition delay analyzer': 'Advanced Fuel Ignition Delay Analyzer', 'ignition delay analyzer': 'Advanced Fuel Ignition Delay Analyzer', 'tube channel propagation': 'Tube Channel Propagation'}
CONDITION_ALIAS_MAP = {'temp': 'Temperature', 'temperature': 'Temperature', 'initial temperature': 'Temperature', 'inlet temperature': 'Temperature', 'reactor temperature': 'Temperature', 'gas temperature': 'Temperature', 'bath temperature': 'Temperature', 'pressure': 'Pressure', 'initial pressure': 'Pressure', 'inlet pressure': 'Pressure', 'reactor pressure': 'Pressure', 'phi': 'Equivalence ratio', 'φ': 'Equivalence ratio', 'equivalence ratio': 'Equivalence ratio', 'fuel-air equivalence ratio': 'Equivalence ratio', 'fuel/air equivalence ratio': 'Equivalence ratio', 'residence time': 'Residence time', 'reaction time': 'Residence time', 'inlet velocity': 'Inlet velocity', 'flow velocity': 'Inlet velocity', 'inlet volumetric flow rate': 'Inlet volumetric flow rate', 'volumetric flow rate': 'Inlet volumetric flow rate', 'flow rate': 'Inlet volumetric flow rate', 'inlet mass flow rate': 'Inlet mass flow rate', 'mass flow rate': 'Inlet mass flow rate', 'fuel composition': 'Fuel composition', 'fuel mixture composition': 'Fuel composition', 'fuel mole fraction': 'Fuel composition', 'oxidizer composition': 'Oxidizer composition', 'oxidiser composition': 'Oxidizer composition', 'oxidant composition': 'Oxidizer composition', 'air composition': 'Oxidizer composition', 'dilution gas': 'Dilution gas', 'diluent gas': 'Dilution gas', 'diluent': 'Dilution gas', 'bath gas': 'Dilution gas', 'separation distance': 'Separation distance', 'reactor volume': 'Reactor volume'}
RESULT_ALIAS_MAP = {'idt': 'Ignition delay time', 'ignition delay': 'Ignition delay time', 'ignition delay time': 'Ignition delay time', 'ignition delay times': 'Ignition delay time', 'autoignition delay time': 'Ignition delay time', 'ait': 'Autoignition temperature', 'autoignition temperature': 'Autoignition temperature', 'auto-ignition temperature': 'Autoignition temperature', 'lbv': 'Laminar flame speed', 'laminar burning velocity': 'Laminar flame speed', 'laminar flame speed': 'Laminar flame speed', 'laminar flame speeds': 'Laminar flame speed', 'burning velocity': 'Laminar flame speed', 'flame speed': 'Laminar flame speed', 'extinction strain rate': 'Extinction strain rate', 'species mole fraction': 'Species mole fraction vs. time', 'mole fraction': 'Species mole fraction vs. time', 'species concentration': 'Species mole fraction vs. time', 'soot volume fraction': 'Soot volume fraction', 'soot number density': 'Soot number density', 'particle size distribution': 'Particle size distribution', 'primary particle diameter': 'Primary particle diameter'}

def clean_string(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    s = re.sub('\\s+', ' ', s)
    return s if s else None

def normalize_key(text: Any) -> Optional[str]:
    s = clean_string(text)
    if not s:
        return None
    s = s.lower()
    return s

def build_fuel_alias_map(dictionary_path: Path) -> Dict[str, str]:
    if not dictionary_path.exists():
        print(f'[WARNING] Fuel dictionary not found: {dictionary_path}')
        raise FileNotFoundError('Supply a fuel dictionary with --fuel_dictionary PATH.')
    with dictionary_path.open('r', encoding='utf-8') as f:
        dictionary = json.load(f)
    alias_map = {}
    conflict_keys = set()
    for entry in dictionary:
        canonical = entry.get('canonical_name')
        if not canonical:
            continue
        names = set(entry.get('names', []))
        names.add(canonical)
        for name in names:
            key = normalize_key(name)
            if not key:
                continue
            if key not in alias_map:
                alias_map[key] = canonical
            elif alias_map[key] != canonical:
                conflict_keys.add(key)
    for key in conflict_keys:
        alias_map.pop(key, None)
    return alias_map

def canonicalize_fuel_name(name: Any, fuel_alias_map: Dict[str, str]) -> Optional[str]:
    key = normalize_key(name)
    if not key:
        return None
    return fuel_alias_map.get(key, clean_string(name))

def normalize_taxonomy_label(value: Any, alias_map: Dict[str, str]) -> Optional[str]:
    key = normalize_key(value)
    if not key:
        return clean_string(value)
    return alias_map.get(key, clean_string(value))

def normalize_taxonomy_in_nodes(data: Dict) -> Dict:
    for node in data.get('nodes', []):
        if not isinstance(node, dict):
            continue
        node_type = node.get('node_type')
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        if node_type == 'Experiment':
            attrs['exp_type'] = normalize_taxonomy_label(attrs.get('exp_type'), EXPERIMENT_ALIAS_MAP)
        elif node_type == 'Condition':
            attrs['condition_type'] = normalize_taxonomy_label(attrs.get('condition_type'), CONDITION_ALIAS_MAP)
        elif node_type == 'ExpResult':
            attrs['result_type'] = normalize_taxonomy_label(attrs.get('result_type'), RESULT_ALIAS_MAP)
    return data

def normalize_fuel_names(data: Dict, fuel_alias_map: Dict[str, str]) -> Dict:
    for node in data.get('nodes', []):
        if not isinstance(node, dict):
            continue
        node_type = node.get('node_type')
        if node_type not in {'Fuel', 'MixedFuel'}:
            continue
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        original_name = attrs.get('name')
        canonical = canonicalize_fuel_name(original_name, fuel_alias_map)
        if canonical:
            attrs['name'] = canonical
        original_clean = clean_string(original_name)
        if original_clean and canonical and (original_clean != canonical):
            aliases = attrs.get('aliases', []) or []
            if not isinstance(aliases, list):
                aliases = [aliases]
            if original_clean not in aliases:
                aliases.append(original_clean)
            attrs['aliases'] = aliases
    return data
VALUE_UNIT_STRIP_PATTERNS = ['\\s*(standard liter per minute)$', '\\s*(ml \\(stp\\)/min)$', '\\s*(l·min-1 \\(stp\\))$', '\\s*(l/min|lpm|slpm|slm|nlph|nl/h|sccm|ml/min|cm3/s|nm3·h-1)$', '\\s*(mm/s|cm/s|m/s)$', '\\s*(g/h|g/min|g/s)$', '\\s*(cm3|ml|l)$', '\\s*(mpa|kpa|mbar|bar|atm)$', '\\s*(°c|kelvin|celsius|k)$', '\\s*(ms|s)$', '\\s*(mm|cm|m)$', '\\s*(mol%|vol%|mole fraction)$']

def strip_value_unit(text: str) -> str:
    x = text
    for pat in VALUE_UNIT_STRIP_PATTERNS:
        x = re.sub(pat, '', x, flags=re.IGNORECASE)
    return x.strip()

def normalize_numeric_string(text: str) -> str:
    x = text.strip()
    if re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', x):
        try:
            val = float(x)
            if abs(val - round(val)) < 1e-09:
                return str(int(round(val)))
            return f'{val:g}'
        except Exception:
            return x
    return x

def normalize_condition_value_string(condition_type: Optional[str], value: Any) -> Optional[str]:
    if value is None:
        return None
    x = str(value).strip()
    if not x:
        return None
    x = re.sub('^(~|∼|approximately|approx\\.?|about)\\s*', '', x, flags=re.IGNORECASE)
    x = re.sub('^(phi|φ|Φ)\\s*=\\s*', '', x, flags=re.IGNORECASE).strip()
    x_before = x
    x = strip_value_unit(x)
    x = normalize_numeric_string(x)
    return x if x else None

def normalize_condition_values_in_nodes(data: Dict) -> Dict:
    for node in data.get('nodes', []):
        if not isinstance(node, dict) or node.get('node_type') != 'Condition':
            continue
        attrs = node.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        condition_type = attrs.get('condition_type')
        if normalize_key(condition_type) == normalize_key('Fuel composition'):
            continue
        values = attrs.get('value', [])
        if not isinstance(values, list):
            values = [values]
        new_values = []
        for v in values:
            nv = normalize_condition_value_string(condition_type, v)
            if nv:
                new_values.append(nv)
        seen = set()
        deduped = []
        for v in new_values:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        attrs['value'] = deduped
    return data

def normalize_fkg(data: Dict, fuel_alias_map: Dict[str, str]) -> Dict:
    data = copy.deepcopy(data)
    data = normalize_taxonomy_in_nodes(data)
    data = normalize_fuel_names(data, fuel_alias_map)
    data = normalize_condition_values_in_nodes(data)
    return data

def parse_prediction(text: Any) -> Dict:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError(f'prediction is {type(text).__name__}')
    text = text.strip()
    text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub('\\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and (end > start):
            return json.loads(text[start:end + 1])
        raise

def load_done_keys(out_file: Path) -> set:
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

def append_jsonl(path: Path, item: Dict) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch_file', help='e.g. clean_000.jsonl')
    ap.add_argument('--input_dir', default=str(INPUT_DIR))
    ap.add_argument('--output_dir', default=str(OUTPUT_DIR))
    ap.add_argument('--fuel_dictionary', default=str(FUEL_DICTIONARY))
    return ap.parse_args()

def resolve_batch_path(arg: str, input_dir: Path) -> Path:
    p = Path(arg)
    if p.exists():
        return p.resolve()
    p2 = input_dir / arg
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f'Cannot find input: {arg}')

def get_batch_suffix(batch_path: Path) -> str:
    m = re.search('(?:clean|inference)_(\\d+)', batch_path.stem)
    return m.group(1) if m else batch_path.stem

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fuel_dict_path = Path(args.fuel_dictionary)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = resolve_batch_path(args.batch_file, input_dir)
    batch_suffix = get_batch_suffix(input_path)
    out_file = output_dir / f'norm_{batch_suffix}.jsonl'
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
    n_skipped = 0
    n_success = 0
    n_failed = 0
    fh = open(input_path, 'r', encoding='utf-8')
    iterator = fh
    if tqdm is not None:
        iterator = tqdm(fh, total=total_lines, desc=f'Norm batch_{batch_suffix}')
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
                pred = parse_prediction(row.get('prediction'))
                normed = normalize_fkg(pred, fuel_alias_map)
                row['prediction'] = json.dumps(normed, ensure_ascii=False, separators=(',', ':'))
                n_success += 1
            except Exception as e:
                row['normalize_error'] = str(e)
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
if __name__ == '__main__':
    main()
