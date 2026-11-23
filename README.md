# RelEHRBench: A Benchmark for Relational Learning on Longitudinal Electronic Health Records

> Official repository for RelEHRBench: A Benchmark for Relational Learning on Longitudinal Electronic Health Records, accepted as an Oral Presentation at the Fifth Learning on Graphs Conference (LoG 2026).

**Authors**: Tianru Li, Kamilia Zaripova, Chantal Pellegrini, Nassir Navab, Azade Farshad.

## Overview
RelEHRBench is a benchmark for **relational learning on longitudinal electronic health records**, derived from **MIMIC-IV**.

Electronic health records are inherently longitudinal relational databases, linking patients to encounters and clinical events distributed across multiple tables. Existing benchmarks often reduce EHRs to isolated encounters or flattened sequences, limiting comparisons between relational and conventional models.

RelEHRBench preserves longitudinal histories, temporal validity, and native relational structure. It includes:

* **30,000** adult patients
* **65,007** hospitalizations
* **13** interconnected tables
* **15M+** rows
* **5** clinical prediction tasks

We provide interoperable **RelBench** and **PyHealth** interfaces, enabling relational, sequential, and feature-engineered tabular models to be evaluated on identical cohorts, labels, and temporal cutoffs.

Our controlled experiments show that **no model family dominates across tasks**: relational models perform best on readmission, phenotyping, and digestive-disease prediction, while sequential and tabular models lead on other tasks.


## Requirements and access

- **MIMIC-IV access.** Permission to query MIMIC-IV 2.2 in Google
  BigQuery, Google Cloud Application Default Credentials, and a Google Cloud project ID is needed. Supply the ID as `--project-id` or set `PROJECT_ID`.
- **Separate environments.** We used two separate Python 3.12 environments.
  - One `relbench` environment for data extraction, patient split generation, LightGBM, GraphSAGE, GAT, and RelGNN.
  - One `pyhealth` environment for StageNet and Transformer.

### Installation

Please follow the upstream installation instructions of [RelBench](https://github.com/stanford-star/relbench) and [Pyhealth](https://pyhealth.readthedocs.io/en/latest/install.html). The
full package snapshots are provided for reference.

We used `relbench==2.1.1`, `google-cloud-bigquery==3.41.0`, and
PyTorch `2.5.0+cu121` with CUDA 12.1 for **RelBench**.
(Our [full package](requirements_relbench.txt) snapshot)

We used `pyhealth==2.0.1` and PyTorch `2.7.1` with CUDA 12.6 for **PyHealth**.
(Our [full package](requirements_pyhealth.txt) snapshot)

## Reproduce main experiment results

1. [Prepare the dataset and generate the shared patient split](data_pipeline/README.md#prepare-the-dataset-and-patient-split).
2. Pass a [task](models/README.md#tasks) and run the corresponding [model training command](models/README.md#run-the-included-models-individually).

