"""Component features and fold-local preprocessing for property models."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
import numpy as np
SEED = 20261007
DATA = REPO_ROOT / 'data/prediction/property'
ALLOW_MISSING = '--allow-missing' in sys.argv
KG_CONTEXT = '--kg-context' in sys.argv
ADVANCED = '--advanced' in sys.argv
FAMILY_IMPUTATION = False
REFINE = False
OUT = REPO_ROOT / 'results/property' / 'advanced_kg' if KG_CONTEXT else REPO_ROOT / 'results/property'
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_DIR = Path(os.environ.get('FKG_FEATURE_DIR', str(REPO_ROOT / 'data/component_inputs')))
feature_file = FEATURE_DIR / 'component_features.json'
if not feature_file.is_file():
    raise FileNotFoundError('Component descriptors are required for property prediction. Set FKG_FEATURE_DIR to a directory containing component_features.json, block_component_features.json and block_component_features_v3.json.')
FEATURES = {k.casefold(): v for k, v in json.loads(feature_file.read_text(encoding='utf-8')).items()}
ALIASES = {'n-heptane': 'n-c7h16', 'ammonia': 'nh3', 'hydrogen': 'h2', 'carbon monoxide': 'co', 'dimethyl ether': 'dme', 'isocetane': 'iso-cetane'}
if FAMILY_IMPUTATION:
    ALIASES.update({'n2': 'nitrogen', 'dimethyl ether': 'di-methyl ether', 'decahydronaphthalene': 'decalin', 'tetrahydronaphthalene': 'tetralin'})
if ADVANCED:
    ALIASES.update({'n2': 'nitrogen', 'dimethyl ether': 'di-methyl ether', 'decahydronaphthalene': 'decalin', 'tetrahydronaphthalene': 'tetralin'})
VOCAB = sorted({str(v['CID']) for v in FEATURES.values()})
CID_INDEX = {c: i for i, c in enumerate(VOCAB)}
BASIC = ['MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'FractionCSP3', 'HeavyAtomCount', 'BertzCT', 'NumAliphaticRings', 'MolMR', 'NOCount', 'NHOHCount', 'LabuteASA']
RICH = sorted(set.intersection(*(set(v['rdkit']) for v in FEATURES.values())))
TARGETS = {'Research octane number (RON)': 'RON', 'Motor octane number (MON)': 'MON', 'Derived cetane number (DCN)': 'DCN', 'Net heat of combustion (NHC)': 'NHC'}

def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def feature(n):
    return FEATURES.get(n.casefold()) or FEATURES.get(ALIASES.get(n.casefold(), ''))
if KG_CONTEXT:
    from graph_context import GraphContext
    GRAPH = GraphContext(Path(os.environ.get('FKG_GRAPH_RECORD_DIR', str(REPO_ROOT / 'data/training_graph_records'))), feature)
primary, auxiliaries = (defaultdict(list), defaultdict(list))
for filename, destination in [('training_catalog.csv', primary), ('auxiliary_records.csv', auxiliaries)]:
    for row in read(filename):
        p = row['property_type']
        if p not in TARGETS:
            continue
        r = dict(row)
        names = json.loads(r['components_json'])
        ff = [feature(n) for n in names]
        if not names or (not ALLOW_MISSING and (not all((v and v.get('CID') for v in ff)))):
            continue
        r['feature_missing'] = [not bool(v and v.get('CID')) for v in ff]
        if ALLOW_MISSING:
            ff = [v if v and v.get('CID') else {'CID': 'unresolved_name:' + n.casefold(), 'imputed': True} for n, v in zip(names, ff)]
        r['ff'], r['names'] = (ff, names)
        r['fraction_values'] = json.loads(r['fractions_json'])
        r['set_key'] = json.dumps(sorted({str(v['CID']) for v in ff}))
        r['y'] = float(r['value'])
        if not math.isfinite(r['y']):
            raise ValueError(r['property_id'])
        r['group'] = r['recipe_group_id'] or 'partial:' + r['set_key'] + ':' + r['doi']
        r['recipe'] = ' / '.join(names) + ' | ' + r['fractions_json'] + ' | ' + r['fraction_basis']
        if FAMILY_IMPUTATION and destination is primary:
            identity = defaultdict(float)
            total = sum(r['fraction_values'])
            for f, fraction in zip(ff, r['fraction_values']):
                identity[str(f['CID'])] += fraction / total
            r['group'] = json.dumps([r['fraction_basis'], sorted(((c, round(w, 8)) for c, w in identity.items()))])
        destination[p].append(r)
if ALLOW_MISSING:
    VOCAB = sorted(set(VOCAB) | {str(f['CID']) for rr in list(primary.values()) + list(auxiliaries.values()) for r in rr for f in r['ff']})
    CID_INDEX = {c: i for i, c in enumerate(VOCAB)}

def matrix_row(r, mode):
    ff, raw = (r['ff'], r['fraction_values'])
    complete = len(raw) == len(ff) and all((v is not None for v in raw))
    w = np.array(raw, dtype=float) if complete else np.ones(len(ff))
    w = w / w.sum()
    onehot = np.zeros(len(VOCAB))
    present = np.zeros(len(VOCAB))
    for f, fraction in zip(ff, w):
        idx = CID_INDEX[str(f['CID'])]
        onehot[idx] += fraction
        present[idx] = 1
    keys = RICH if mode == 'rich' else BASIC
    vectors = np.nan_to_num([[f['rdkit'].get(k, 0) for k in keys] for f in ff], nan=0, posinf=0, neginf=0)
    avg = np.sum(vectors * w[:, None], axis=0)
    sd = np.sqrt(np.maximum(0, np.sum((vectors - avg) ** 2 * w[:, None], axis=0)))
    mw = np.array([f['molecular_weight'] for f in ff])
    c = np.array([f['carbon'] for f in ff])
    h = np.array([f['hydrogen'] for f in ff])
    mass_fraction = w * mw / (w @ mw) if r['fraction_basis'] == 'mole' else w
    mol_fraction = w / mw / np.sum(w / mw) if r['fraction_basis'] == 'mass' else w
    extra = [len(ff), float(complete), float(bool(r['fraction_basis'])), -np.sum(w * np.log(w)), w @ mw, 1 / np.sum(w / mw), np.sum(mol_fraction * h) / max(np.sum(mol_fraction * c), 1e-09), np.sum(mass_fraction * c / mw), np.sum(mass_fraction * h / mw)]
    basis = np.array([r['fraction_basis'] == b for b in ['mole', 'mass', 'volume', 'pure', '']], dtype=float)
    parts = [onehot, onehot ** 2, present, avg, sd, np.array(extra), basis]
    if ALLOW_MISSING:
        missing = np.array(r['feature_missing'], dtype=float)
        parts.extend([np.array([missing.sum(), missing.mean(), w @ missing]), present * np.array([c.startswith('unresolved_name:') for c in VOCAB])])
    if REFINE:
        mass_cid = np.zeros(len(VOCAB))
        mol_cid = np.zeros(len(VOCAB))
        physical_basis_known = complete and r['fraction_basis'] in {'mole', 'mass', 'pure'}
        if physical_basis_known:
            for f, mass, mol in zip(ff, mass_fraction, mol_fraction):
                mass_cid[CID_INDEX[str(f['CID'])]] += mass
                mol_cid[CID_INDEX[str(f['CID'])]] += mol
        parts.extend([mass_cid, mol_cid, np.array([physical_basis_known])])
        if mode == 'blend':
            i, j = np.triu_indices(len(VOCAB), 1)
            parts.append(onehot[i] * onehot[j])
    if mode == 'rich':
        parts.append(np.sum(np.array([f['fingerprint'] for f in ff]) * w[:, None], axis=0))
    if mode == 'blend':
        for active in basis[:3]:
            parts.extend([active * onehot, active * onehot ** 2])
    return np.concatenate(parts)

def allowed_aux(aux, validation, mode):
    if mode == 'none':
        return []
    dois = {r['doi'] for r in validation}
    groups = {r['group'] for r in validation}
    sets = {r['set_key'] for r in validation}
    result = [r for r in aux if r['doi'] not in dois and r['group'] not in groups and (r['set_key'] not in sets)]
    if mode == 'pure':
        result = [r for r in result if len(r['names']) == 1]
    assert not {r['doi'] for r in result} & dois
    assert not {r['set_key'] for r in result} & sets
    return result
MATRIX_CACHE = {}

def fold_matrices(fit_rows, valid, mode):
    key = (mode, tuple(((r['owner_id'], r['property_id']) for r in fit_rows)), tuple(((r['owner_id'], r['property_id']) for r in valid)))
    if key in MATRIX_CACHE:
        return MATRIX_CACHE[key]
    known = {str(f['CID']): f for r in fit_rows for f in r['ff'] if not f.get('imputed')}
    assert known, 'No known training components for fold-local imputation'
    known_rows = list(known.values())

    def mean_feature(pool):
        return {'rdkit': {k: float(np.mean([f['rdkit'].get(k, 0) for f in pool])) for k in RICH}, 'fingerprint': np.mean([f['fingerprint'] for f in pool], axis=0), **{k: float(np.mean([f[k] for f in pool])) for k in ['molecular_weight', 'carbon', 'hydrogen']}}
    fallback = mean_feature(known_rows)
    family_cache = {}

    def for_name(name):
        if not FAMILY_IMPUTATION:
            return fallback
        n = name.casefold()
        if any((t in n for t in ['biodiesel', 'methyl ester', 'ethyl ester', 'palm oil', 'hvo'])):
            family = 'long_chain_oxygenated'
        elif any((t in n for t in ['furan', '2-mf', 'ome', 'oxymethylene', 'bio-oil', 'surfactant', 'ethanol'])):
            family = 'oxygenated'
        elif 'trimethylbenzene' in n:
            family = 'aromatic'
        elif any((t in n for t in ['gasoline', 'pentene', 'isoparaffin', 'methylheptane'])):
            family = 'light_hydrocarbon'
        elif any((t in n for t in ['diesel', 'jet', 'rp-3', 'spk', 'atj', 'sasol', 'hefa', 'hrj', 'sip', 'jp-8', 'ulsd'])):
            family = 'middle_distillate_hydrocarbon'
        else:
            family = 'global'
        if family in family_cache:
            return family_cache[family]

        def match(f):
            carbon, hetero = (f['carbon'], f['rdkit'].get('NOCount', 0))
            return {'long_chain_oxygenated': carbon >= 8 and hetero > 0, 'oxygenated': carbon > 0 and hetero > 0, 'aromatic': f['rdkit'].get('NumAromaticRings', 0) > 0, 'light_hydrocarbon': 4 <= carbon <= 12 and hetero == 0, 'middle_distillate_hydrocarbon': 8 <= carbon <= 22 and hetero == 0, 'global': True}[family]
        selected = [f for f in known_rows if match(f)]
        family_cache[family] = mean_feature(selected) if len(selected) >= 2 else fallback
        return family_cache[family]

    def make(rows):
        result = []
        for r in rows:
            rr = dict(r)
            rr['ff'] = [dict(for_name(n), CID=f['CID']) if f.get('imputed') else f for n, f in zip(r['names'], r['ff'])]
            result.append(matrix_row(rr, mode))
        return np.array(result)
    result = (make(fit_rows), make(valid))
    assert np.isfinite(result[0]).all() and np.isfinite(result[1]).all()
    MATRIX_CACHE[key] = result
    return result
