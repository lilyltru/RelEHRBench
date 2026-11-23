"""Build the reproduction MIMIC-IV 2.2 dataset from BigQuery."""

import argparse

from .adapter import RelBenchAdaptConfig, RelBenchAdapter
from .raw import MimicRawExtractor, RawExtractConfig
from .subset import HospSubsetBuilder, HospSubsetConfig


RAW_CACHE_NAME = "mimic4_raw_v2_2_30k"
SUBSET_CACHE_NAME = "hosp_subset_v2_2_30k"
RELBENCH_CACHE_NAME = "relehr-v2_2_30k"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV 2.2, build the cohort, and adapt it to RelBench."
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Google Cloud project; defaults to the PROJECT_ID environment variable.",
    )
    parser.add_argument("--patients", type=int, default=30_000)
    parser.add_argument("--min-age", type=int, default=18)
    parser.add_argument(
        "--top-lab-items",
        type=int,
        default=0,
        help="Number of frequent lab items to keep; 0 keeps all items.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_dir = MimicRawExtractor(
        RawExtractConfig(name=RAW_CACHE_NAME),
        project_id=args.project_id,
    ).run(force=args.force)

    subset_dir = HospSubsetBuilder(
        HospSubsetConfig(
            name=SUBSET_CACHE_NAME,
            raw_dir=raw_dir,
            patients_limit=args.patients,
            min_age=args.min_age,
            top_lab_items=args.top_lab_items,
        ),
        project_id=args.project_id,
    ).run(force=args.force)

    RelBenchAdapter(
        RelBenchAdaptConfig(
            name=RELBENCH_CACHE_NAME,
            subset_dir=subset_dir,
        )
    ).run(force=args.force)


if __name__ == "__main__":
    main()
