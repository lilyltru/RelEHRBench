# RelEHRBench dataset extraction

This directory contains the MIMIC-IV 2.2 extraction and RelBench adaptation
pipeline. Run all commands from the RelEHRBench repository root.

For installation and the complete benchmark reproduction, see the
[main README](../README.md). For model training and custom-model integration,
see the [model README](../models/README.md).

## Pipeline overview

The pipeline has three manifest-tracked stages:

| Stage | Implementation | Input | Output |
|---|---|---|---|
| Raw extraction | `raw.py` | MIMIC-IV 2.2 in BigQuery | Full `patients` and `admissions` Parquet files |
| Cohort and table construction | `subset.py` | Raw cache plus cohort-filtered BigQuery queries | Selected, cleaned MIMIC-Hosp tables |
| RelBench adaptation | `adapter.py` | Subset Parquet files | RelBench database, ID mappings, schema, and manifest |

The default cohort contains the first 30,000 eligible adult patients ordered
by their first valid hospital admission.
The available RelBench tables are declared in `schema.py`:

```text
patients                 admissions
admission_discharges     d_labitems
labevents                d_icd_diagnoses
diagnoses_icd            prescriptions
prescription_stops       d_icd_procedures
procedures_icd           drgcodes
omr
```
The adapter additionally writes
`table_schema.json` and original-to-RelBench key mappings.

Calendar-date values without a clock time, such as `procedures_icd.chartdate`
and `omr.chartdate`, are stored at the final nanosecond of that day. Full
timestamps retain their original time. This prevents date-only events from
appearing available at midnight when their within-day availability is unknown.

## Prepare the dataset and patient split

Activate the RelBench environment and set `PROJECT_ID` to your Google Cloud
project ID. Extract the benchmark dataset, then generate the shared 80/10/10
train/validation/test split with NumPy seed 42.

```bash
source .venv-relbench/bin/activate
export PROJECT_ID=YOUR_GCP_PROJECT
SPLIT_ID=v2_2_80_10_10_seed42_numpy

python -m data_pipeline.cli --project-id "$PROJECT_ID"
python -m tasks.register --split-id "$SPLIT_ID" --split-seed 42 --split-ratios 0.8 0.1 0.1
```

Both the `patient` and `joint` protocols use this partition. The dataset and
split files are stored under `cache/datasets/`.

```text
cache/datasets/
├── raw/mimic4_raw_v2_2_30k/
├── subsets/hosp_subset_v2_2_30k/tables/*.parquet
└── relbench/relehr-v2_2_30k/
    ├── db/*.parquet
    ├── mappings/*.parquet
    ├── manifest.json
    ├── table_schema.json
    └── db_patient/v2_2_80_10_10_seed42_numpy/
        ├── subject_ids/{train,val,test}.npy
        └── original_subject_ids/{train,val,test}.npy
```

## Table dependencies

Table selection must preserve the relational graph:

- Always retain `patients` and `admissions`.
- `labevents` requires `d_labitems`.
- `diagnoses_icd` requires `d_icd_diagnoses`.
- `procedures_icd` requires `d_icd_procedures`.
- A table containing `hadm_id` requires `admissions`; a table containing
  `subject_id` requires `patients`.
- The readmission task requires `admission_discharges`.
- The high-risk-lab task additionally requires `labevents` and `d_labitems`.
- The diagnosis multilabel tasks require `diagnoses_icd` and, under the
  `joint` protocol, `admission_discharges`.

`RelBenchAdapter` validates foreign-key targets and stops if a required target
table is absent.

## Add a new source table

To add a table that is not already supported:

The `raw` stage caches the complete MIMIC-IV `patients` and `admissions`
tables, while other hospital tables are loaded by `HospSubsetBuilder` through
cohort-filtered BigQuery queries. To modify tables' filtering conditions or add
a new MIMIC hospital table, you can modify `subset.py`.

1. Define its `RelBenchTableSpec` in `schema.py`.
2. Implement its cohort-filtered query and transformation in
   `HospSubsetBuilder`.
3. Include the table in both the subset and adapter `tables` tuples.
4. Declare its primary key, timestamp, and every foreign-key relationship.
5. Add feature semantics to the GNN stype file or the PyHealth YAML when the
   table should be a model input.
