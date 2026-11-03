# Prediction methods

## Property models

### Inputs

`data/prediction/property/training_catalog.csv` contains **464 complete-recipe
records**. `auxiliary_records.csv` contains **95 auxiliary records**: 78 pure-component
records and 17 records with incomplete mixture proportions.

| Column | Meaning |
|---|---|
| `owner_id`, `property_id` | Source fuel and property identifiers |
| `property_type` | Prediction target |
| `components_json` | Component names |
| `fractions_json` | Reported fractions, with null for unspecified entries |
| `fraction_basis` | Mole, mass or volume basis |
| `value`, `unit` | Reference property value and unit |
| `doi` | Source DOI |
| `recipe_group_id` | Recipe identity used for cross-validation grouping |

Model metrics compare predictions with the reference property values in the dataset.

### Features

Set `FKG_FEATURE_DIR` to the descriptor directory containing:

- `component_features.json`
- `block_component_features.json`
- `block_component_features_v3.json`

These files map component names to molecular identifiers, descriptors and fingerprints.
Features combine component identity, fractions, composition basis, weighted molecular
aggregates, fingerprints, interactions and fold-local category proxies.

Set `FKG_GRAPH_RECORD_DIR` to the source-graph directory containing:

- `original_entities.csv`
- `original_relations.csv`
- `merged_entities.csv`
- `merged_relations.csv`

Graph-context features use article-level neighborhoods and non-target properties,
with validation DOIs excluded from feature construction.

### Training

- Five shuffled outer recipe-group folds, seed 20261007.
- Three training-only recipe-group folds for model and parameter selection.
- Complete recipes remain within one outer fold.
- Auxiliary records enter fitting after validation DOI, recipe-group and component-set exclusion.
- Auxiliary observations are deduplicated by recipe group and value.
- Related papers and component combinations can occur across outer folds.

`block_models.py` trains component-block kernels, mixing-rule residual models,
SVR and tree models. `graph_models.py` trains graph-context variants. RON, MON
and DCN configurations are selected by inner-fold MSE; NHC uses numeric context.
`select_predictions.py` combines the selected fold predictions.

```bash
export FKG_FEATURE_DIR=/path/to/component_descriptors
export FKG_GRAPH_RECORD_DIR=/path/to/source_graph_records
python scripts/prediction/property/run_models.py
```

Fold predictions, selected configurations and numerical metrics are written to
`results/property/`. Prediction outputs use `reference_value` for the target label.
