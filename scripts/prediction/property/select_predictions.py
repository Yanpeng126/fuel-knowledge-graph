"""Assemble recipe-group OOF predictions using training-side model selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
PREDICTION_FIELDS = ['property', 'owner_id', 'property_id', 'doi', 'recipe', 'recipe_group', 'fold', 'reference_value', 'oof_prediction', 'unit']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def assemble(root, output):
    output.mkdir(parents=True, exist_ok=True)
    variants = ['composition_blocks', 'composition_blocks/extended', 'composition_blocks/enriched', 'composition_blocks/selected']
    predictions = {v: read(root / v / 'oof_predictions.csv') for v in variants}
    folds = {v: {(r['property'], int(r['fold'])): r for r in read(root / v / 'outer_folds.csv')} for v in variants}
    identity = lambda rows: {(r['property'], r['owner_id'], r['property_id'], r['reference_value'], r['recipe_group'], r['fold']) for r in rows}
    assert all((identity(rows) == identity(predictions[variants[0]]) for rows in predictions.values()))
    selected, decisions = ([], [])
    for prop in ('RON', 'MON', 'DCN'):
        for fold in range(5):
            chosen = min(variants, key=lambda v: float(folds[v][prop, fold]['inner_MSE']))
            decisions.append({'property': prop, 'fold': fold, 'variant': chosen, 'inner_MSE': folds[chosen][prop, fold]['inner_MSE']})
            selected.extend((dict(r) for r in predictions[chosen] if r['property'] == prop and int(r['fold']) == fold))
    numeric_context = read(root / 'advanced_kg/numeric_context/oof_predictions.csv')
    selected.extend((dict(r) for r in numeric_context if r['property'] == 'NHC'))
    groups = {}
    for r in selected:
        groups.setdefault((r['property'], r['recipe_group']), set()).add(r['fold'])
    assert all((len(v) == 1 for v in groups.values())), 'A recipe crosses evaluation folds'
    summary = []
    for prop in ('RON', 'MON', 'DCN', 'NHC'):
        rows = [r for r in selected if r['property'] == prop]
        y = np.array([float(r['reference_value']) for r in rows])
        p = np.array([float(r['oof_prediction']) for r in rows])
        summary.append({'property': prop, 'records': len(rows), 'R2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'MAE': float(np.abs(y - p).mean())})
    prediction_rows = [{key: row[key] for key in PREDICTION_FIELDS} for row in selected]
    write(output / 'oof_predictions.csv', prediction_rows)
    write(output / 'model_selection.csv', decisions)
    write(output / 'metrics.csv', summary)
    return summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=REPO / 'results/property')
    parser.add_argument('--output', type=Path, default=REPO / 'results/property/reported')
    args = parser.parse_args()
    print(json.dumps(assemble(args.runs_root, args.output), indent=2))
