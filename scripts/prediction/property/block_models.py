"""Nested fixed-fold OOF for block kernels and mixing-rule residual models."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import runpy
import csv
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXTENDED = '--extended' in sys.argv
ENRICHED = '--enriched' in sys.argv
SELECTED = '--selected' in sys.argv
ENRICHED = ENRICHED or SELECTED
WORKERS = 4 if '--workers4' in sys.argv else 1
CALIBRATED = '--calibrated' in sys.argv
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression
source = HERE / 'feature_pipeline.py'
sys.argv = [str(source), '--allow-missing', '--advanced']
core = runpy.run_path(str(source), run_name='feature_pipeline')
core['FAMILY_IMPUTATION'] = True
core['REFINE'] = True
OUT = REPO_ROOT / 'results/property/composition_blocks'
if CALIBRATED:
    OUT = OUT / 'calibrated'
if EXTENDED or ENRICHED:
    OUT = OUT / ('extended' if EXTENDED else 'selected' if SELECTED else 'enriched')
    additions = json.loads((core['FEATURE_DIR'] / ('block_component_features.json' if EXTENDED else 'block_component_features_v3.json')).read_text())
    core['FEATURES'].update({k.casefold(): v for k, v in additions.items()})
    for pool in [core['primary'], core['auxiliaries']]:
        for rows in pool.values():
            for r in rows:
                for i, name in enumerate(r['names']):
                    if name in additions:
                        r['ff'][i] = additions[name]
                        r['feature_missing'][i] = False
                r['set_key'] = json.dumps(sorted({str(f['CID']) for f in r['ff']}))
    core['VOCAB'] = sorted(set(core['VOCAB']) | {str(v['CID']) for v in additions.values()})
    core['CID_INDEX'] = {c: i for i, c in enumerate(core['VOCAB'])}
OUT.mkdir(parents=True, exist_ok=True)
SEED = core['SEED']
CACHE = {}

def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def prepare(train, valid, aux, scope):
    key = (tuple(((r['owner_id'], r['property_id']) for r in train)), tuple(((r['owner_id'], r['property_id']) for r in valid)), scope)
    if key in CACHE:
        return CACHE[key]
    extra = core['allowed_aux'](aux, valid, scope)
    seen, dedup = (set(), [])
    for r in extra:
        k = (r['group'], r['y'])
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    rows = train + dedup
    assert {r['group'] for r in rows}.isdisjoint({r['group'] for r in valid})
    basic, vb = core['fold_matrices'](rows, valid, 'basic')
    rich, vr = core['fold_matrices'](rows, valid, 'rich')
    n = len(core['VOCAB'])
    fp = len(next(iter(core['FEATURES'].values()))['fingerprint'])
    parts = {'composition': (basic[:, :n], vb[:, :n]), 'compact': (basic[:, 3 * n:3 * n + 2 * len(core['BASIC']) + 14], vb[:, 3 * n:3 * n + 2 * len(core['BASIC']) + 14]), 'rich': (rich[:, 3 * n:3 * n + 2 * len(core['RICH']) + 14], vr[:, 3 * n:3 * n + 2 * len(core['RICH']) + 14]), 'fingerprint': (rich[:, -fp:], vr[:, -fp:]), 'full': (rich, vr)}
    if SELECTED:
        yy = np.array([r['y'] for r in rows])
        for block in ['rich', 'full']:
            xx, vv = parts[block]
            keep = xx.var(axis=0) > 1e-12
            xx, vv = (xx[:, keep], vv[:, keep])
            scores, _ = f_regression(xx, yy)
            order = np.argsort(np.nan_to_num(scores, nan=0, posinf=1e+30))[::-1]
            for count in [8, 24, 64]:
                indices = order[:count]
                parts[block + '_' + str(count)] = (xx[:, indices], vv[:, indices])
    distances, linear = ({}, {})
    for name, (x, v) in parts.items():
        x, v = (np.nan_to_num(x), np.nan_to_num(v))
        scaled = name in {'compact', 'rich', 'full'} or '_' in name
        if scaled:
            keep = x.var(axis=0) > 1e-12
            scaler = StandardScaler().fit(x[:, keep])
            x, v = (scaler.transform(x[:, keep]), scaler.transform(v[:, keep]))
        width = max(x.shape[1], 1)
        for metric in ['sqeuclidean', 'cityblock']:
            norm = width if scaled else 1
            distances[name, metric] = (cdist(x, x, metric) / norm, cdist(v, x, metric) / norm)
        linear[name] = (x @ x.T, v @ x.T)
    cx, cv = parts['composition']
    i, j = np.triu_indices(n)
    prior_x = np.column_stack([cx, cx[:, i] * cx[:, j]])
    prior_v = np.column_stack([cv, cv[:, i] * cv[:, j]])
    keep = prior_x.var(axis=0) > 1e-12
    y = np.array([r['y'] for r in rows])
    result = (distances, linear, prior_x[:, keep], prior_v[:, keep], y, parts)
    CACHE[key] = result
    return result

def kernel(d, linear, name, kind, gamma):
    if name == 'hybrid':
        a = kernel(d, linear, 'composition', kind, gamma)
        b = kernel(d, linear, 'compact', kind, gamma)
        return tuple((0.5 * x + 0.5 * y for x, y in zip(a, b)))
    if kind == 'tanimoto':
        a, b = linear[name]
        norm = np.diag(a)
        dv = d[name, 'sqeuclidean'][1]
        vn = np.maximum(dv[:, 0] - norm[0] + 2 * b[:, 0], 0)
        return (a / np.maximum(norm[:, None] + norm[None, :] - a, 1e-12), b / np.maximum(vn[:, None] + norm[None, :] - b, 1e-12))
    metric = 'cityblock' if kind == 'laplace' else 'sqeuclidean'
    a, b = d[name, metric]

    def transform(x):
        if kind in {'rbf', 'laplace'}:
            return np.exp(-gamma * x)
        if kind == 'rq':
            return 1 / (1 + gamma * x)
        z = np.sqrt(np.maximum(3 * gamma * x, 0))
        return (1 + z) * np.exp(-z)
    return (transform(a), transform(b))

def predict(train, valid, aux, spec):
    d, linear, x, v, y, parts = prepare(train, valid, aux, spec['aux'])
    mean, sd = (y.mean(), max(y.std(), 1e-09))
    yy = (y - mean) / sd
    if spec['model'] in {'extra', 'boost'}:
        x, v = parts[spec['block']]
        if spec['model'] == 'extra':
            m = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=spec['leaf'], max_features=spec['max_features'], random_state=SEED, n_jobs=2)
        else:
            m = GradientBoostingRegressor(n_estimators=150, learning_rate=0.04, max_depth=spec['depth'], loss=spec['loss'], random_state=SEED)
        return m.fit(x, yy).predict(v) * sd + mean
    if spec['model'] == 'ridge':
        m = Ridge(alpha=spec['alpha']).fit(x, yy)
        return m.predict(v) * sd + mean
    a, b = kernel(d, linear, spec['block'], spec['kernel'], spec['gamma'])
    prior = np.zeros(len(y))
    pv = np.zeros(len(valid))
    if spec['model'] == 'residual':
        m = Ridge(alpha=0.1).fit(x, yy)
        prior, pv = (m.predict(x), m.predict(v))
    if spec['model'] == 'svr':
        m = SVR(kernel='precomputed', C=spec['alpha'], epsilon=0.01).fit(a, yy)
        pp = m.predict(b)
    else:
        pp = b @ solve(a + spec['alpha'] * np.eye(len(y)), yy - prior, assume_a='pos') + pv
    return pp * sd + mean
specs = []
for aux in ['none', 'pure', 'all']:
    for alpha in [0.001, 0.03, 1, 30]:
        specs.append(dict(model='ridge', aux=aux, alpha=alpha))
    for block in ['composition', 'compact', 'rich', 'fingerprint', 'hybrid']:
        for kind in ['rbf', 'laplace', 'rq', 'matern']:
            for gamma in [0.03, 0.3, 3]:
                for alpha in [0.001, 0.03]:
                    specs.append(dict(model='krr', aux=aux, block=block, kernel=kind, gamma=gamma, alpha=alpha))
    for block in ['fingerprint', 'composition']:
        for alpha in [0.001, 0.03]:
            specs.append(dict(model='krr', aux=aux, block=block, kernel='tanimoto', gamma=1, alpha=alpha))
    for block in ['compact', 'rich', 'hybrid']:
        for gamma in [0.03, 0.3, 3]:
            specs.append(dict(model='residual', aux=aux, block=block, kernel='rbf', gamma=gamma, alpha=0.03))
            specs.append(dict(model='svr', aux=aux, block=block, kernel='rbf', gamma=gamma, alpha=100))
if EXTENDED:
    for aux in ['none', 'pure', 'all']:
        for gamma in [0.01, 0.1, 1, 10]:
            for alpha in [0.0001, 0.001, 0.01, 0.1]:
                specs.append(dict(model='krr', aux=aux, block='full', kernel='rbf', gamma=gamma, alpha=alpha))
        for block in ['composition', 'compact', 'full', 'fingerprint']:
            for leaf in [1, 2]:
                for maximum in [0.5, 1.0]:
                    specs.append(dict(model='extra', aux=aux, block=block, leaf=leaf, max_features=maximum))
            for depth in [1, 2, 3]:
                for loss in ['squared_error', 'huber']:
                    specs.append(dict(model='boost', aux=aux, block=block, depth=depth, loss=loss))
if SELECTED:
    specs = []
    for aux in ['none', 'all']:
        for block in ['rich_8', 'rich_24', 'rich_64', 'full_8', 'full_24', 'full_64']:
            for kind in ['rbf', 'laplace']:
                for gamma in [0.03, 0.3, 3]:
                    for alpha in [0.001, 0.03, 0.3]:
                        specs.append(dict(model='krr', aux=aux, block=block, kernel=kind, gamma=gamma, alpha=alpha))
            for gamma in [0.03, 0.3, 3]:
                for penalty in [10, 100]:
                    specs.append(dict(model='svr', aux=aux, block=block, kernel='rbf', gamma=gamma, alpha=penalty))
summary, predictions, folds = ([], [], [])
for prop, short in core['TARGETS'].items():
    rows, aux = (core['primary'][prop], core['auxiliaries'][prop])
    y = np.array([r['y'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(rows), np.nan)
    fold_ids = np.full(len(rows), -1)
    print('START', short, len(rows), len(specs), flush=True)
    for fold, (tr, va) in enumerate(outer.split(y, y, groups)):
        CACHE.clear()
        core['MATRIX_CACHE'].clear()
        train, valid = ([rows[i] for i in tr], [rows[i] for i in va])
        available = core['allowed_aux'](aux, valid, 'all')
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold).split(tr, y[tr], groups[tr]))
        if WORKERS > 1:
            for it, iv in inner:
                for scope in sorted({s['aux'] for s in specs}):
                    prepare([train[i] for i in it], [train[i] for i in iv], available, scope)

        def evaluate(spec):
            ip = np.full(len(tr), np.nan)
            try:
                for it, iv in inner:
                    ip[iv] = predict([train[i] for i in it], [train[i] for i in iv], available, spec)
                score = mean_squared_error(y[tr], ip)
            except (ValueError, np.linalg.LinAlgError):
                score = float('inf')
            return (ip, score)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            evaluated = list(executor.map(evaluate, specs))
        ips, scores = ([], [])
        for spec, (ip, score) in zip(specs, evaluated):
            ips.append(ip)
            scores.append(score)
        order = np.argsort(scores)
        chosen, best = ([int(order[0])], scores[order[0]])
        for count in [2, 3, 5, 10]:
            indices = [int(i) for i in order[:count]]
            mse = mean_squared_error(y[tr], np.mean([ips[i] for i in indices], axis=0))
            if mse < best:
                chosen, best = (indices, mse)
        weights = np.ones(len(chosen)) / len(chosen)
        if EXTENDED:
            for count in [10, 30]:
                indices = [int(i) for i in order[:count]]
                matrix = np.column_stack([ips[i] for i in indices])
                target = y[tr]
                norm = max(target.var(), 1e-09)
                for penalty in [0.01, 0.1]:
                    objective = lambda w: np.mean((matrix @ w - target) ** 2) / norm + penalty * np.sum((w - 1 / count) ** 2)
                    fitted = minimize(objective, np.ones(count) / count, method='SLSQP', bounds=[(0, 1)] * count, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}, options={'maxiter': 200, 'ftol': 1e-09})
                    mse = mean_squared_error(target, matrix @ fitted.x)
                    if fitted.success and mse < best:
                        chosen, best, weights = (indices, mse, fitted.x)
        final = np.column_stack([predict(train, valid, available, specs[i]) for i in chosen]) @ weights
        calibration = {}
        if CALIBRATED:
            inner_prediction = np.column_stack([ips[i] for i in chosen]) @ weights
            m = Ridge(alpha=1).fit(inner_prediction[:, None], y[tr])
            final = m.predict(final[:, None])
            calibration = {'slope': float(m.coef_[0]), 'intercept': float(m.intercept_)}
        oof[va] = final
        fold_ids[va] = fold
        folds.append({'property': short, 'fold': fold, 'selected': json.dumps([specs[i] for i in chosen]), 'weights': json.dumps(weights.tolist()), 'inner_MSE': best})
        if CALIBRATED:
            folds[-1]['calibration'] = json.dumps(calibration)
        print('FOLD', short, fold, 'inner_MSE', best, flush=True)
    assert np.isfinite(oof).all()
    for i, r in enumerate(rows):
        predictions.append({'property': short, 'owner_id': r['owner_id'], 'property_id': r['property_id'], 'doi': r['doi'], 'recipe': r['recipe'], 'recipe_group': r['group'], 'fold': int(fold_ids[i]), 'reference_value': r['y'], 'oof_prediction': float(oof[i]), 'unit': r['unit']})
    result = {'property': short, 'records': len(rows), 'R2': r2_score(y, oof), 'MAE': mean_absolute_error(y, oof)}
    summary.append(result)
    for name, data in [('summary.csv', summary), ('oof_predictions.csv', predictions), ('outer_folds.csv', folds)]:
        write(name, data)
    print('RESULT', json.dumps(result), flush=True)
print('DONE', flush=True)
