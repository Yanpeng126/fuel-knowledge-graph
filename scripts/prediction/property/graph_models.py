"""Fixed-fold nested search of graph kernels and molecular/graph ensembles."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import runpy
import csv
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
COMPOSITION_ONLY = '--composition-ablation' in sys.argv
FUEL_LAYER = '--fuel-layer' in sys.argv
NUMERIC_CONTEXT = '--numeric-context' in sys.argv
FUEL_LAYER = FUEL_LAYER or NUMERIC_CONTEXT
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
source = HERE / 'feature_pipeline.py'
sys.argv = [str(source), '--allow-missing', '--kg-context', '--advanced']
core = runpy.run_path(str(source), run_name='feature_pipeline')
OUT, GRAPH, SEED = (core['OUT'], core['GRAPH'], core['SEED'])
if FUEL_LAYER:
    OUT = OUT / ('numeric_context' if NUMERIC_CONTEXT else 'fuel_layer')
    OUT.mkdir(parents=True, exist_ok=True)
    GRAPH.add_fuel_layer(Path(os.environ.get('FKG_GRAPH_RECORD_DIR', str(REPO_ROOT / 'data/training_graph_records'))), numeric=NUMERIC_CONTEXT)
if COMPOSITION_ONLY:
    OUT = OUT / 'composition_ablation'
    OUT.mkdir(parents=True, exist_ok=True)
core['FAMILY_IMPUTATION'] = True
core['REFINE'] = True

def write(name, rows):
    if rows:
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

def diffusion(excluded):
    key = frozenset(excluded)
    if key in DIFFUSION:
        return DIFFUSION[key]
    fuels = sorted(GRAPH.events)
    index = {f: i for i, f in enumerate(fuels)}
    paper = defaultdict(set)
    for fuel, events in GRAPH.events.items():
        for doi, _, co in events:
            if doi not in key:
                paper[doi].update(co)
    ii, jj, values = ([], [], [])
    for members in paper.values():
        members = sorted(members & set(index))
        if len(members) < 2:
            continue
        weight = 1 / (len(members) - 1)
        for a in members:
            for b in members:
                if a != b:
                    ii.append(index[a])
                    jj.append(index[b])
                    values.append(weight)
    adjacency = csr_matrix((values, (ii, jj)), shape=(len(fuels), len(fuels)))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = diags(1 / np.maximum(degree, 1)) @ adjacency
    anchors = sorted({str(f['CID']) for f in core['FEATURES'].values()})
    restart = np.zeros((len(fuels), len(anchors)))
    for j, fuel in enumerate(anchors):
        if fuel in index:
            restart[index[fuel], j] = 1
    state = restart.copy()
    pieces = []
    for step in range(20):
        state = 0.15 * restart + 0.85 * (transition @ state)
        if step in {0, 2, 7, 19}:
            pieces.append(state.copy())
    embedding = np.column_stack(pieces)
    result = {fuel: embedding[i] for fuel, i in index.items()}
    DIFFUSION[key] = result
    return result

def diffusion_matrix(rows, excluded):
    vectors = diffusion(excluded)
    zero = np.zeros(4 * len({str(f['CID']) for f in core['FEATURES'].values()}))
    matrix = []
    for r in rows:
        vv = np.array([vectors.get(GRAPH.identity(n), zero) for n in r['names']])
        raw = r['fraction_values']
        weights = np.array(raw, dtype=float) if len(raw) == len(vv) and all((x is not None for x in raw)) else np.ones(len(vv))
        weights /= weights.sum()
        matrix.append(weights @ vv)
    return np.array(matrix)

def prepare(train, valid, aux, spec, outer_dois):
    extra = core['allowed_aux'](aux, valid, spec['aux'])
    seen, dedup = (set(), [])
    for r in extra:
        k = (r['group'], r['y'])
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    rows = train + dedup
    key = (tuple(((r['owner_id'], r['property_id']) for r in rows)), tuple(((r['owner_id'], r['property_id']) for r in valid)), spec['features'], frozenset(outer_dois))
    if key in MATRICES:
        return MATRICES[key]
    x, v = core['fold_matrices'](rows, valid, spec['features'])
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    v = np.nan_to_num(v, nan=0, posinf=0, neginf=0)
    keep = np.var(x, axis=0) > 1e-12
    x, v = (x[:, keep], v[:, keep])
    scale = StandardScaler().fit(x)
    x, v = (scale.transform(x), scale.transform(v))
    excluded = outer_dois | {r['doi'] for r in valid}
    gx, gv = GRAPH.matrices(rows, valid, excluded)
    dx, dv = (diffusion_matrix(rows, excluded), diffusion_matrix(valid, excluded))
    graph_scaler = StandardScaler().fit(gx)
    gx, gv = (graph_scaler.transform(gx), graph_scaler.transform(gv))
    dx /= np.maximum(np.linalg.norm(dx, axis=1, keepdims=True), 1e-12)
    dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-12)
    y = np.array([r['y'] for r in rows])
    result = (x, v, gx, gv, dx, dv, y, len(dedup))
    MATRICES[key] = result
    return result

def squared_distance(a, b):
    key = (id(a), id(b))
    if key not in DISTANCES:
        DISTANCES[key] = np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T, 0) / max(a.shape[1], 1)
    return DISTANCES[key]

def predict(train, valid, aux, spec, outer_dois):
    x, v, gx, gv, dx, dv, y, na = prepare(train, valid, aux, spec, outer_dois)
    mean, sd = (y.mean(), max(y.std(), 1e-09))
    yy = (y - mean) / sd
    weight = spec['weight']
    if spec['model'] in {'krr', 'svr'}:
        molecular = np.exp(-spec['gamma'] * squared_distance(x, x))
        molecular_test = np.exp(-spec['gamma'] * squared_distance(v, x))
        if spec['graph'] == 'diffusion':
            graph = dx @ dx.T
            graph_test = dv @ dx.T
        else:
            graph = np.exp(-squared_distance(gx, gx))
            graph_test = np.exp(-squared_distance(gv, gx))
        kernel = (1 - weight) * molecular + weight * graph
        test = (1 - weight) * molecular_test + weight * graph_test
        if spec['model'] == 'krr':
            prediction = test @ np.linalg.solve(kernel + spec['alpha'] * np.eye(len(y)), yy)
        else:
            m = SVR(kernel='precomputed', C=spec['C'], epsilon=0.02)
            m.fit(kernel, yy)
            prediction = m.predict(test)
    else:
        xx = np.column_stack([x, weight * gx, weight * dx]) if weight else x
        vv = np.column_stack([v, weight * gv, weight * dv]) if weight else v
        if spec['model'] == 'extra':
            m = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=spec['leaf'], max_features=spec['max_features'], random_state=SEED, n_jobs=2)
        else:
            m = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=spec['alpha'], learning_rate=0.06, random_state=SEED)
        m.fit(xx, yy)
        prediction = m.predict(vv)
    return (prediction * sd + mean, na)
specs = []
for features in ['blend', 'rich']:
    for aux in ['none', 'all']:
        for graph in ['context', 'diffusion']:
            for gamma in [0.01, 0.1, 1, 10]:
                for alpha in [0.0001, 0.01, 0.1]:
                    for weight in [0.05, 0.3, 0.7]:
                        specs.append(dict(features=features, aux=aux, graph=graph, gamma=gamma, alpha=alpha, weight=weight, model='krr'))
rng = np.random.default_rng(SEED)
specs = [specs[i] for i in rng.choice(len(specs), 84, replace=False)]
for features in ['blend', 'rich']:
    for aux in ['none', 'all']:
        for weight in [0, 0.3, 1]:
            specs.append(dict(features=features, aux=aux, graph='context', gamma=0.1, alpha=0.001, weight=weight, model='krr'))
            specs.append(dict(features=features, aux=aux, graph='context', gamma=1, C=100, weight=weight, model='svr'))
        for leaf in [1, 2]:
            specs.append(dict(features=features, aux=aux, graph='context', leaf=leaf, max_features=1.0, weight=1, model='extra'))
        specs.append(dict(features=features, aux=aux, graph='context', alpha=1, weight=1, model='hist'))
if COMPOSITION_ONLY:
    unique = {}
    for s in specs:
        s = dict(s, weight=0, graph='context')
        unique[json.dumps(s, sort_keys=True)] = s
    specs = list(unique.values())
predictions, summary, folds = ([], [], [])
DIFFUSION, MATRICES = ({}, {})
DISTANCES = {}
for prop, short in core['TARGETS'].items():
    rows, aux = (core['primary'][prop], core['auxiliaries'][prop])
    y = np.array([r['y'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    oof = np.full(len(rows), np.nan)
    fold_ids = np.full(len(rows), -1)
    print('START', short, len(rows), 'candidates', len(specs), flush=True)
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(outer.split(y, y, groups)):
        MATRICES.clear()
        DIFFUSION.clear()
        DISTANCES.clear()
        GRAPH.cache.clear()
        core['MATRIX_CACHE'].clear()
        train, valid = ([rows[i] for i in tr], [rows[i] for i in va])
        excluded = {r['doi'] for r in valid}
        available = core['allowed_aux'](aux, valid, 'all')
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold).split(tr, y[tr], groups[tr]))
        candidate_predictions, scores = ([], [])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for spec in specs:
                ip = np.full(len(tr), np.nan)
                try:
                    for it, iv in inner:
                        ip[iv], _ = predict([train[i] for i in it], [train[i] for i in iv], available, spec, excluded)
                    score = mean_squared_error(y[tr], ip)
                except (ValueError, np.linalg.LinAlgError):
                    score = np.inf
                candidate_predictions.append(ip)
                scores.append(score)
        order = np.argsort(scores)
        chosen, best = ([int(order[0])], scores[order[0]])
        for n in [2, 3, 5]:
            indices = [int(i) for i in order[:n]]
            score = mean_squared_error(y[tr], np.mean([candidate_predictions[i] for i in indices], axis=0))
            if score < best:
                chosen, best = (indices, score)
        pp = [predict(train, valid, available, specs[i], excluded)[0] for i in chosen]
        oof[va] = np.mean(pp, axis=0)
        fold_ids[va] = fold
        folds.append({'property': short, 'fold': fold, 'selected': json.dumps([specs[i] for i in chosen]), 'inner_MSE': best})
        print('FOLD', short, fold, 'ensemble', len(chosen), 'inner_MSE', best, flush=True)
    assert np.isfinite(oof).all()
    for i, r in enumerate(rows):
        predictions.append({'property': short, 'owner_id': r['owner_id'], 'property_id': r['property_id'], 'doi': r['doi'], 'recipe': r['recipe'], 'recipe_group': r['group'], 'fold': int(fold_ids[i]), 'reference_value': r['y'], 'oof_prediction': float(oof[i]), 'unit': r['unit']})
    result = {'property': short, 'records': len(rows), 'R2': r2_score(y, oof), 'MAE': mean_absolute_error(y, oof)}
    summary.append(result)
    for name, values in [('summary.csv', summary), ('oof_predictions.csv', predictions), ('outer_folds.csv', folds)]:
        write(name, values)
    print('RESULT', json.dumps(result), flush=True)
print('DONE', flush=True)
