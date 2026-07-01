"""
tournament.py
-------------
Tournament bracket logic for the 2026 FIFA World Cup.
 
Implements group stage round-robin play, third-place qualification
(2026-specific), and a full five-round knockout bracket
(R32 → R16 → QF → SF → Final).
 
Author: Selma Marrakchi (2026)
"""
 
from __future__ import annotations
import itertools
from typing import Dict, List, Tuple
 
import pandas as pd
 
from model import spi_to_goal_rates, elo_to_goal_rates, simulate_match
 
 
# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------
 
GroupStandings = Dict[str, Dict[str, int]]  # team -> {pts, gd, gf}
 
 
def group_stage(group: List[str],
                ratings: pd.DataFrame,
                rating_mode: str = "spi",
                n: int = 10_000) -> Tuple[str, str, str, GroupStandings]:
    """
    Simulate round-robin group play.
 
    Each pair of teams meets once. Points are awarded as:
        Win  → 3 pts
        Draw → 1 pt each
        Loss → 0 pts
 
    Tiebreakers (in order): points → goal differential → goals scored.
 
    Parameters
    ----------
    group : List[str]
        Team names in the group (4 teams).
    ratings : pd.DataFrame
        Team ratings table. Must contain columns appropriate for rating_mode.
    rating_mode : str
        "spi"  — uses 'off' and 'def' columns
        "elo"  — uses 'elo' column
    n : int
        Monte Carlo iterations per match.
 
    Returns
    -------
    first, second, third : str
        Teams finishing 1st, 2nd, and 3rd.
    standings : GroupStandings
        Full standings dictionary for third-place comparison.
    """
    standings: GroupStandings = {
        team: {"pts": 0, "gd": 0, "gf": 0} for team in group
    }
 
    for team_a, team_b in itertools.combinations(group, 2):
        lambda_a, lambda_b = _get_lambdas(team_a, team_b, ratings, rating_mode)
        result = simulate_match(lambda_a, lambda_b, n=n, elimination=False)
 
        # Simulate scoreline for goal statistics
        goals_a, goals_b = _sample_scoreline(lambda_a, lambda_b)
 
        if result == "A":
            standings[team_a]["pts"] += 3
        elif result == "B":
            standings[team_b]["pts"] += 3
        else:  # draw
            standings[team_a]["pts"] += 1
            standings[team_b]["pts"] += 1
 
        standings[team_a]["gf"] += goals_a
        standings[team_b]["gf"] += goals_b
        standings[team_a]["gd"] += goals_a - goals_b
        standings[team_b]["gd"] += goals_b - goals_a
 
    ranked = sorted(
        group,
        key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]),
        reverse=True
    )
 
    return ranked[0], ranked[1], ranked[2], standings
 
 
# ---------------------------------------------------------------------------
# Third-place qualification (2026-specific)
# ---------------------------------------------------------------------------
 
def select_best_third_place(
        third_place_candidates: List[Tuple[str, GroupStandings]],
        n_advance: int = 8
) -> List[str]:
    """
    Select the N best third-place finishers across all 12 groups.
 
    Ranking criteria (in order):
        1. Points
        2. Goal differential
        3. Goals scored
 
    Parameters
    ----------
    third_place_candidates : List[Tuple[str, GroupStandings]]
        List of (team_name, standings_dict) for each group's third-place team.
    n_advance : int
        Number of third-place teams that advance (default 8).
 
    Returns
    -------
    List[str]
        Names of the advancing third-place teams.
    """
    ranked = sorted(
        third_place_candidates,
        key=lambda x: (x[1][x[0]]["pts"],
                       x[1][x[0]]["gd"],
                       x[1][x[0]]["gf"]),
        reverse=True
    )
    return [team for team, _ in ranked[:n_advance]]
 
 
# ---------------------------------------------------------------------------
# Knockout bracket
# ---------------------------------------------------------------------------
 
def run_knockout_bracket(teams: List[str],
                         ratings: pd.DataFrame,
                         rating_mode: str = "spi",
                         n: int = 10_000) -> str:
    """
    Run a single-elimination knockout bracket.
 
    Expects 32 teams (for R32 → R16 → QF → SF → Final).
    Draws are resolved via Bernoulli tie-break (penalty simulation).
 
    Parameters
    ----------
    teams : List[str]
        Ordered list of 32 teams entering the Round of 32.
    ratings : pd.DataFrame
        Team ratings table.
    rating_mode : str
        "spi" or "elo".
    n : int
        Monte Carlo iterations per match.
 
    Returns
    -------
    str
        Name of the tournament winner.
    """
    remaining = list(teams)
 
    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            team_a = remaining[i]
            team_b = remaining[i + 1]
            lambda_a, lambda_b = _get_lambdas(team_a, team_b, ratings, rating_mode)
            result = simulate_match(lambda_a, lambda_b, n=n, elimination=True)
            winner = team_a if result == "A" else team_b
            next_round.append(winner)
        remaining = next_round
 
    return remaining[0]
 
 
# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------
 
def simulate_tournament(groups: Dict[str, List[str]],
                        ratings: pd.DataFrame,
                        rating_mode: str = "spi",
                        n: int = 10_000) -> str:
    """
    Simulate the full 2026 FIFA World Cup tournament.
 
    Parameters
    ----------
    groups : Dict[str, List[str]]
        Mapping of group name (e.g. "A") to list of 4 team names.
    ratings : pd.DataFrame
        Team ratings.
    rating_mode : str
        "spi" or "elo".
    n : int
        Monte Carlo iterations per match.
 
    Returns
    -------
    str
        Name of the simulated tournament winner.
    """
    group_results: Dict[str, Tuple[str, str]] = {}
    third_place_candidates: List[Tuple[str, GroupStandings]] = []
 
    for group_name, group in groups.items():
        first, second, third, standings = group_stage(
            group, ratings, rating_mode=rating_mode, n=n
        )
        group_results[group_name] = (first, second)
        third_place_candidates.append((third, standings))
 
    advancing_thirds = select_best_third_place(third_place_candidates, n_advance=8)
 
    # Build Round of 32 bracket
    r32: List[str] = []
    for first, second in group_results.values():
        r32.append(first)
        r32.append(second)
    r32.extend(advancing_thirds)
 
    return run_knockout_bracket(r32, ratings, rating_mode=rating_mode, n=n)
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _get_lambdas(team_a: str, team_b: str,
                 ratings: pd.DataFrame,
                 rating_mode: str) -> Tuple[float, float]:
    if rating_mode == "spi":
        off_a = ratings.loc[ratings.name == team_a, "off"].values[0]
        def_a = ratings.loc[ratings.name == team_a, "def"].values[0]
        off_b = ratings.loc[ratings.name == team_b, "off"].values[0]
        def_b = ratings.loc[ratings.name == team_b, "def"].values[0]
        return spi_to_goal_rates(off_a, def_b, off_b, def_a)
    elif rating_mode == "elo":
        elo_a = ratings.loc[ratings.name == team_a, "elo"].values[0]
        elo_b = ratings.loc[ratings.name == team_b, "elo"].values[0]
        return elo_to_goal_rates(elo_a, elo_b)
    else:
        raise ValueError(f"Unknown rating_mode: {rating_mode}")
 
 
def _sample_scoreline(lambda_a: float, lambda_b: float) -> Tuple[int, int]:
    """Sample a single scoreline for goal differential tracking."""
    import numpy as np
    return int(np.random.poisson(lambda_a)), int(np.random.poisson(lambda_b))
