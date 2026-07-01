"""
preprocess.py
-------------
Standalone preprocessing script for the 2026 FIFA World Cup Elo ratings dataset.

Downloads the raw Kaggle dataset, runs the full preprocessing pipeline,
and saves a clean ratings CSV ready for simulation.

Usage
-----
    # With Kaggle API configured:
    python preprocess.py --download

    # With dataset already in data/ directory:
    python preprocess.py --input data/elo_ratings_wc2026.csv

Output: data/ratings_preprocessed.csv

Author: Selma Marrakchi (2026)
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from utils import load_ratings, QUALIFIED_TEAMS_2026, TOURNAMENT_KICKOFF


KAGGLE_DATASET = 'afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings'
RAW_PATH       = Path('data/elo_ratings_wc2026.csv')
OUTPUT_PATH    = Path('data/ratings_preprocessed.csv')


def download_dataset() -> None:
    """Download the Elo ratings dataset from Kaggle."""
    print(f"Downloading: {KAGGLE_DATASET}")
    Path('data').mkdir(exist_ok=True)
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET,
         '--unzip', '-p', 'data/'],
        check=True
    )
    print(f"Downloaded to data/")


def run_preprocessing(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline and save the result.

    Pipeline:
        1. Load raw CSV
        2. Filter to pre-tournament snapshots (<= June 10, 2026)
        3. Retain most recent snapshot per team
        4. Filter to 48 qualified teams
        5. Derive offensive/defensive ratings from Elo
        6. Validate and save
    """
    print(f"\n{'='*55}")
    print("  Preprocessing: 2026 FIFA World Cup Elo Ratings")
    print(f"{'='*55}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    # Load raw
    raw = pd.read_csv(input_path)
    print(f"Raw dataset: {len(raw):,} rows, {len(raw.columns)} columns")
    print(f"Columns: {list(raw.columns)}\n")

    # Run pipeline
    df = load_ratings(str(input_path), mode='elo')

    # Show sample
    print("\nSample (top 8 by Elo):")
    print(df.nlargest(8, 'rating')[['country', 'rating', 'off', 'def']].to_string(index=False))

    # Save
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved preprocessed ratings: {len(df)} teams → {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 2026 FIFA World Cup Elo ratings for simulation"
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset from Kaggle before preprocessing'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(RAW_PATH),
        help=f'Path to raw Elo ratings CSV (default: {RAW_PATH})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_PATH),
        help=f'Output path for preprocessed CSV (default: {OUTPUT_PATH})'
    )
    args = parser.parse_args()

    if args.download:
        download_dataset()

    run_preprocessing(Path(args.input), Path(args.output))

    print("\nPreprocessing complete. Run simulation with:")
    print(f"  python simulate.py --data {args.output}")


if __name__ == '__main__':
    main()
