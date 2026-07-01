"""
bayesian.py
-----------
Bayesian updating of team offensive and defensive ratings after observed matches.
 
Uses conjugate Gamma priors on Poisson rates. After each observed match,
the posterior is updated analytically:
 
    lambda | k ~ Gamma(alpha + k, beta + 1)
    posterior mean = (alpha + k) / (beta + 1)
 
Usage
-----
    # After group stage — update priors with all 72 observed scorelines
    python bayesian.py --results data/group_stage_results.csv --round group
 
    # After Round of 32 — update with 16 observed scorelines
    python bayesian.py --results data/r32_results.csv --round r32
 
    # Output: results/win_probabilities_updated.csv
 
Author: Selma Marrakchi (2026)
"""
 
import argparse
import collections
import copy
import csv
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
 
import numpy as np
import pandas as pd
 
from simulate import GROUPS_2026
from utils import load_ratings, QUALIFIED_TEAMS_2026
 
FAC = [math.factorial(i) for i in range(15)]
 
 
# ---------------------------------------------------------------------------
# Gamma prior dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class TeamPrior:
    """Gamma prior on a team's offensive and defensive Poisson rates."""
    alpha_off: float
    beta_off:  float
    alpha_def: float
    beta_def:  float
 
    @property
    def off(self) -> float:
        """Posterior mean offensive rate."""
        return self.alpha_off / self.beta_off
 
    @property
    def def_(self) -> float:
        """Posterior mean defensive rate."""
        return self.alpha_def / self.beta_def
 
 
# ---------------------------------------------------------------------------
# Prior initialization
# ---------------------------------------------------------------------------
 
def initialize_priors(ratings: dict,
                      prior_strength: float = 5.0) -> Dict[str, TeamPrior]:
    """
    Initialize Gamma priors from pre-tournament Elo-derived ratings.
 
    Parameters
    ----------
    ratings : dict
        Team ratings in the format {team_name: {'off': float, 'def': float}}.
    prior_strength : float
        Number of 'virtual matches' the pre-tournament rating represents.
        Higher values make the prior more resistant to in-tournament updating.
        Recommended range: 3–10. Default: 5.
 
    Returns
    -------
    Dict[str, TeamPrior]
        Initialized priors for all teams.
    """
    return {
        team: TeamPrior(
            alpha_off = r['off'] * prior_strength,
            beta_off  = prior_strength,
            alpha_def = r['def'] * prior_strength,
            beta_def  = prior_strength,
        )
        for team, r in ratings.items()
    }
 
 
# ---------------------------------------------------------------------------
# Bayesian update
# ---------------------------------------------------------------------------
 
def update_after_match(priors: Dict[str, TeamPrior],
                       team_a: str, goals_a: int,
                       team_b: str, goals_b: int) -> Dict[str, TeamPrior]:
    """
    Update Gamma priors for both teams after an observed match.
 
    Goals scored update the offensive prior (alpha_off += goals, beta_off += 1).
    Goals conceded update the defensive prior (alpha_def += goals_conceded, beta_def += 1).
 
    Parameters
    ----------
    priors   : current priors for all teams (not mutated)
    team_a   : name of team A
    goals_a  : goals scored by team A
    team_b   : name of team B
    goals_b  : goals scored by team B
 
    Returns
    -------
    Dict[str, TeamPrior]
        Updated priors (deep copy — original is not mutated).
    """
    priors = copy.deepcopy(priors)
 
    # Team A
    priors[team_a].alpha_off += goals_a
    priors[team_a].beta_off  += 1
    priors[team_a].alpha_def += goals_b
    priors[team_a].beta_def  += 1
 
    # Team B
    priors[team_b].alpha_off += goals_b
    priors[team_b].beta_off  += 1
    priors[team_b].alpha_def += goals_a
    priors[team_b].beta_def  += 1
 
    return priors
 
 
def update_from_results(priors: Dict[str, TeamPrior],
                        results: List[Tuple[str, int, str, int]]) -> Dict[str, TeamPrior]:
    """
    Apply sequential Bayesian updates from a list of observed match results.
 
    Parameters
    ----------
    priors  : initial priors
    results : list of (team_a, goals_a, team_b, goals_b) tuples
 
    Returns
    -------
    Dict[str, TeamPrior]
        Fully updated priors after all observed matches.
    """
    for team_a, goals_a, team_b, goals_b in results:
        priors = update_after_match(priors, team_a, goals_a, team_b, goals_b)
    return priors
 
 
def priors_to_ratings(priors: Dict[str, TeamPrior]) -> dict:
    """Convert posterior means to the ratings dict format used by the simulator."""
    return {team: {'off': p.off, 'def': p.def_} for team, p in priors.items()}
 
 
# ---------------------------------------------------------------------------
# Pairwise probability computation
# ---------------------------------------------------------------------------
 
def poisson_probs(la: float, lb: float, nmax: int = 10):
    pa_win = p_draw = 0.0
    for i in range(nmax + 1):
        poi_a = math.exp(-la) * la**i / FAC[i]
        for j in range(nmax + 1):
            poi_b = math.exp(-lb) * lb**j / FAC[j]
            if i > j:    pa_win += poi_a * poi_b
            elif i == j: p_draw += poi_a * poi_b
    return pa_win, p_draw, 1 - pa_win - p_draw
 
 
def build_prob_cache(remaining_teams: List[str], ratings: dict) -> dict:
    cache = {}
    for a, b in itertools.combinations(remaining_teams, 2):
        la = 0.5 * (ratings[a]['off'] + ratings[b]['def'])
        lb = 0.5 * (ratings[b]['off'] + ratings[a]['def'])
        cache[(a, b)] = poisson_probs(la, lb)
        cache[(b, a)] = (cache[(a,b)][2], cache[(a,b)][1], cache[(a,b)][0])
    return cache
 
 
# ---------------------------------------------------------------------------
# Knockout simulation (self-contained, no dependency on simulate.py state)
# ---------------------------------------------------------------------------
 
def sim_match(a: str, b: str, prob_cache: dict) -> str:
    pa, pd, pb = prob_cache[(a, b)]
    r = np.random.random()
    if r < pa:      return a
    elif r < pa+pd: return a if np.random.random() < 0.5 else b
    else:           return b
 
 
def run_knockout(teams: List[str], prob_cache: dict) -> str:
    remaining = list(teams)
    while len(remaining) > 1:
        remaining = [sim_match(remaining[i], remaining[i+1], prob_cache)
                     for i in range(0, len(remaining), 2)]
    return remaining[0]
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def load_results_csv(path: str) -> List[Tuple[str, int, str, int]]:
    """
    Load observed match results from CSV.
    Expected columns: team_a, goals_a, team_b, goals_b
    """
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append((
                row['team_a'], int(row['goals_a']),
                row['team_b'], int(row['goals_b'])
            ))
    return results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Bayesian updating of win probabilities after observed matches"
    )
    parser.add_argument('--ratings',  default='data/ratings_preprocessed.csv',
                        help='Pre-tournament ratings CSV')
    parser.add_argument('--results',  required=True,
                        help='Observed match results CSV (team_a,goals_a,team_b,goals_b)')
    parser.add_argument('--remaining', default=None,
                        help='CSV of teams still in tournament (one per row). '
                             'If omitted, all 48 teams are used.')
    parser.add_argument('--prior_strength', type=float, default=5.0,
                        help='Gamma prior strength (default: 5.0)')
    parser.add_argument('--n_simulations', type=int, default=100_000)
    parser.add_argument('--output', default='results/win_probabilities_updated.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
 
    np.random.seed(args.seed)
 
    # Load pre-tournament ratings
    df = load_ratings(args.ratings)
    ratings = df.set_index('name')[['off', 'def']].to_dict('index')
 
    # Initialize priors
    priors = initialize_priors(ratings, prior_strength=args.prior_strength)
 
    # Load and apply observed results
    results = load_results_csv(args.results)
    print(f"Applying {len(results)} observed match results...")
    priors = update_from_results(priors, results)
 
    # Updated ratings
    updated_ratings = priors_to_ratings(priors)
 
    # Remaining teams
    if args.remaining:
        with open(args.remaining) as f:
            remaining_teams = [line.strip() for line in f if line.strip()]
    else:
        remaining_teams = list(ratings.keys())
 
    print(f"Simulating over {len(remaining_teams)} remaining teams...")
    prob_cache = build_prob_cache(remaining_teams, updated_ratings)
 
    # Run simulation
    win_counts = collections.defaultdict(int)
    t0 = time.time()
    for i in range(args.n_simulations):
        win_counts[run_knockout(remaining_teams, prob_cache)] += 1
        if (i+1) % 20_000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,}/{args.n_simulations:,} | {(i+1)/el:.0f} sims/s")
 
    results_out = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
 
    print(f"\n{'Rank':<6}{'Team':<30}{'Win %':>10}")
    print('─' * 48)
    for rank, (team, wins) in enumerate(results_out, 1):
        print(f"{rank:<6}{team:<30}{wins/args.n_simulations:>9.1%}")
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(t, w, round(w/args.n_simulations, 4)) for t, w in results_out],
        columns=['team', 'wins', 'win_probability']
    ).to_csv(args.output, index=False)
    print(f"\nSaved to {args.output} | {time.time()-t0:.1f}s | seed={args.seed}")
 
 
if __name__ == '__main__':
    main()
 
