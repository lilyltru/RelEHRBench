# RelEHRBench: A Benchmark for Relational Learning on Longitudinal Electronic Health Records

> Official repository for RelEHRBench: A Benchmark for Relational Learning on Longitudinal Electronic Health Records, accepted as an Oral Presentation at the Fifth Learning on Graphs Conference (LoG 2026).

**Authors**: Tianru Li, Kamilia Zaripova, Chantal Pellegrini, Nassir Navab, Azade Farshad.

## Overview
RelEHRBench is a benchmark for **relational learning on longitudinal electronic health records**, derived from **MIMIC-IV**.

Electronic health records are inherently longitudinal relational databases, linking patients to encounters and clinical events distributed across multiple tables. Existing benchmarks often reduce EHRs to isolated encounters or flattened event sequences, limiting comparisons between relational and conventional models.

RelEHRBench preserves longitudinal histories, temporal validity, and native relational structure. It includes:

* **30,000** adult patients
* **65,007** hospitalizations
* **13** interconnected tables
* **15M+** temporally indexed records
* **5** clinical prediction tasks

We provide interoperable **RelBench** and **PyHealth** interfaces, enabling relational, sequential, and feature-engineered tabular models to be evaluated on identical cohorts, labels, and temporal cutoffs.

Our experiments show that **no model family dominates across tasks**: relational models perform best on readmission, phenotyping, and digestive-disease prediction, while sequential and tabular models lead on other tasks.

## Code

Code and benchmark construction scripts will be made public soon.
