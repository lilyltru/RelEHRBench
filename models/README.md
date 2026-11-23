# RelEHRBench models

This directory contains the LightGBM, GraphSAGE, GAT, RelGNN, StageNet, and
Transformer baselines.

## Model stacks

The benchmark uses [two Python environments](../README.md#requirements-and-access).

| Environment | Models | Training entry point |
|---|---|---|
| RelBench | LightGBM | `models.lightgbm.train` |
| RelBench | GraphSAGE and GAT | `models.gnn.train` |
| RelBench | RelGNN | `models.relgnn.train` |
| PyHealth | StageNet | `models.pyhealth.train_stagenet` |
| PyHealth | Transformer | `models.pyhealth.train_transformer` |

All models reuse the same cached cohort and patient partition.

## Tasks

| Clinical task | RelBench name | PyHealth name | Type |
|---|---|---|---|
| 30-day readmission | `patient-hosp-admit-30d` | `readmission_30d` | Binary |
| High-risk abnormal lab in 24 hours | `patient-high-risk-lab-abnormal-24h` | `high_risk_lab_24h` | Binary |
| Phenotyping | `patient-phenotyping` | `phenotyping` | Multilabel |
| Cardiovascular subtype | `patient-cardio-subtype` | `cardiovascular_subtype` | Multilabel |
| Digestive subtype | `patient-digestive-subtype` | `digestive_subtype` | Multilabel |

Two protocols are supported:

- `patient`: disjoint patient cohorts with complete patient histories.
- `joint`: the same disjoint patients plus cumulative temporal windows. Train
  history ends at 2123-01-01, validation history ends at 2125-01-01, and test
  history ends at the database end. A future-label sample is retained only if
  its complete label horizon is observable inside the split.

## Run the included models individually

Run the commands from the repository root after
[preparing the dataset and patient split](../data_pipeline/README.md#prepare-the-dataset-and-patient-split).
Each command trains one model on the task passed through `--task "$TASK"`.
Set `TASK` using the appropriate environment column in the [task table](#tasks).
The examples use 30-day readmission, the shared 80/10/10 split, the `joint`
protocol, and the default model seed of 42. Use `--split-protocol patient` to
run the patient-only protocol.

### RelBench models

Set `--gnn gat` in the GraphSAGE command to train GAT. LightGBM uses the
complete training table with `--sample_size 0` and builds features in
`models/lightgbm/features_allowed.py`.

```bash
source .venv-relbench/bin/activate
SPLIT_ID=v2_2_80_10_10_seed42_numpy
TASK=patient-hosp-admit-30d

# LightGBM
python -m models.lightgbm.train --task "$TASK" --split-id "$SPLIT_ID" --split-protocol joint \
  --num_trials 50 --sample_size 0

# GraphSAGE or GAT
python -m models.gnn.train --task "$TASK" --split-id "$SPLIT_ID" --split-protocol joint \
  --gnn sage --stypes-file stypes_matched.json \
  --node_tables patients admissions diagnoses_icd procedures_icd prescriptions labevents \
  --epochs 20 --lr 1e-4 --batch_size 128 --encoder mlp

# RelGNN
python -m models.relgnn.train --task "$TASK" --split-id "$SPLIT_ID" --split-protocol joint \
  --stypes-file stypes_matched.json \
  --node_tables patients admissions diagnoses_icd procedures_icd prescriptions labevents \
  --epochs 20 --lr 1e-4 --batch_size 128 --encoder mlp --num_heads 4
```

### PyHealth models

```bash
source .venv-pyhealth/bin/activate
SPLIT_ID=v2_2_80_10_10_seed42_numpy
TASK=readmission_30d

python -m models.pyhealth.train_stagenet --task "$TASK" --split-id "$SPLIT_ID" --split-protocol joint --lr 1e-4
python -m models.pyhealth.train_transformer --task "$TASK" --split-id "$SPLIT_ID" --split-protocol joint --lr 1e-3
```

StageNet and Transformer print their test metrics to the terminal. Model
caches and run outputs use these paths, where `<model>` is `lightgbm`, `gnn`,
`relgnn`, `stagenet`, or `transformer`.

```text
cache/<model>/<protocol>/<split_id>/<task>/...
outputs/<model>/<protocol>/<split_id>/<task>/seed_<seed>/<run_id>/...
```

## Customize GNN input data

GraphSAGE, GAT, and RelGNN can change input tables and feature columns without
rebuilding the dataset.

- `--node_tables TABLE` selects model input tables that are retained
  and passed to the encoder and GNN. Use
  `--node_tables all` for all available tables. The task entity table must
  remain.
- `--stypes-file FILE` defines which feature columns are encoded for each
  table and assigns a semantic type to every selected column.
  - `--stypes-file stypes.json` declares the full schema.
- The reproduction commands select six input tables and encode the columns in
  `stypes_matched.json` to align the GNN inputs with the PyHealth feature
  configuration.
