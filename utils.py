"""
utils.py
--------
Data loading, preprocessing, and output utilities for the 2026 FIFA World Cup
Poisson-Monte Carlo simulation.

Preprocessing pipeline:
    1. Load raw Elo ratings CSV (Cruz, 2026 — Kaggle)
    2. Normalize column names
    3. Extract pre-tournament snapshot (latest per team before June 11, 2026)
    4. Filter to 48 qualified teams
    5. Derive SPI-style offensive/defensive ratings from Elo
    6. Validate

Author: Selma Marrakchi (2026)
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_RATE = 1.35   # approximate mean goals per team per international match
SCALE     = 400.0  # standard Elo scale parameter
TOURNAMENT_KICKOFF = pd.Timestamp('2026-06-10')  # pre-tournament cutoff for reproducibility

QUALIFIED_TEAMS_2026 = [
    'Mexico', 'South Korea', 'South Africa', 'Czechia',
    'Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina',
    'Brazil', 'Morocco', 'Scotland', 'Haiti',
    'United States', 'Turkey', 'Australia', 'Paraguay',
    'Germany', 'Ivory Coast', 'Ecuador', 'Curaçao',
    'Netherlands', 'Japan', 'Sweden', 'Tunisia',
    'England', 'Croatia', 'Ghana', 'Panama',
    'Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay',
    'France', 'Senegal', 'Iraq', 'Norway',
    'Argentina', 'Algeria', 'Austria', 'Jordan',
    'Portugal', 'Colombia', 'DR Congo', 'Uzbekistan',
    'Belgium', 'Egypt', 'Iran', 'New Zealand',
]


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def load_ratings(path: str, mode: str = 'elo') -> pd.DataFrame:
    """
    Full preprocessing pipeline for team ratings.

    Steps:
        1. Load CSV
        2. Extract pre-tournament snapshot (most recent per team <= June 10 2026)
        3. Filter to 48 qualified teams
        4. Derive off/def ratings from Elo (if mode == 'elo')
        5. Validate

    Parameters
    ----------
    path : str
        Path to ratings CSV. Expected format: Cruz (2026) Kaggle Elo dataset
        with columns 'country' and 'rating'.
    mode : str
        'elo' — derive off/def from Elo ratings (default)
        'spi' — expect off/def columns directly in CSV

    Returns
    -------
    pd.DataFrame
        Preprocessed ratings with columns: country, rating, off, def
    """
    # Step 1: Load
    df = pd.read_csv(path)

    # Step 2: Extract most recent pre-tournament snapshot per team
    if 'snapshot_date' in df.columns:
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        df = df[df['snapshot_date'] <= TOURNAMENT_KICKOFF]
        df = df.sort_values('snapshot_date')
        df = df.groupby('country').last().reset_index()

    # Step 3: Filter to 48 qualified teams
    df = df[df['country'].isin(QUALIFIED_TEAMS_2026)].reset_index(drop=True)

    missing_teams = set(QUALIFIED_TEAMS_2026) - set(df['country'].tolist())
    if missing_teams:
        print(f"Warning: {len(missing_teams)} teams not found in dataset: "
              f"{sorted(missing_teams)}")

    # Step 4: Derive off/def from Elo if needed
    if mode == 'elo' and 'off' not in df.columns:
        df = _derive_goal_rates(df)
    elif mode == 'spi':
        required = ['off', 'def']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SPI mode requires columns: {missing}")

    # Step 5: Validate
    _validate(df)

    return df


def _derive_goal_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive offensive and defensive goal rates from Elo ratings.

    Uses logistic normalization relative to the field mean:
        off_i = BASE_RATE * (1 + 0.5 * (rating_i - mean_rating) / SCALE)
        def_i = BASE_RATE * (1 - 0.5 * (rating_i - mean_rating) / SCALE)

    Higher Elo → higher offensive rate, lower defensive concession rate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rating' column.

    Returns
    -------
    pd.DataFrame
        Input dataframe with 'off' and 'def' columns added.
    """
    df = df.copy()
    mean_rating = df['rating'].mean()

    df['off'] = BASE_RATE * (1 + 0.5 * (df['rating'] - mean_rating) / SCALE)
    df['def'] = BASE_RATE * (1 - 0.5 * (df['rating'] - mean_rating) / SCALE)

    df['off'] = df['off'].clip(lower=0.3, upper=3.5)
    df['def'] = df['def'].clip(lower=0.3, upper=3.5)

    return df


def _validate(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the preprocessed ratings dataframe."""
    required = ['country', 'rating', 'off', 'def']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after preprocessing: {missing_cols}")

    assert df[required].notna().all().all(), \
        "Missing values detected in ratings"
    assert (df['off'] > 0).all() and (df['def'] > 0).all(), \
        "Non-positive goal rates detected"

    print(f"Dataset validated: {len(df)} teams | "
          f"Elo range: {df['rating'].min():.0f}–{df['rating'].max():.0f} | "
          f"off range: {df['off'].min():.2f}–{df['off'].max():.2f}")


# ---------------------------------------------------------------------------
# Group loading
# ---------------------------------------------------------------------------

def load_groups(path: str) -> Dict[str, List[str]]:
    """
    Load group assignments from CSV.

    Expected columns: ['group', 'team']

    Returns
    -------
    Dict[str, List[str]]
        Mapping of group name to list of team names.
    """
    df = pd.read_csv(path)
    groups = {}
    for group_name, group_df in df.groupby('group'):
        groups[group_name] = group_df['team'].tolist()
    return groups


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, float], top_n: int = 15) -> None:
    """Print a formatted results table to stdout."""
    print(f"\n{'Rank':<6}{'Team':<30}{'Win Probability':>16}")
    print('-' * 55)
    for rank, (team, prob) in enumerate(
        sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_n],
        start=1
    ):
        print(f"{rank:<6}{team:<30}{prob:>15.1%}")
    print('-' * 55)


def save_results(results: Dict[str, float], path: Path) -> None:
    """Save win probabilities to CSV."""
    df = pd.DataFrame([
        {'team': team, 'win_probability': round(prob, 4)}
        for team, prob in sorted(
            results.items(), key=lambda x: x[1], reverse=True
        )
    ])
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} team probabilities to {path}")
