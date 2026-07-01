"""
model.py
--------
Poisson regression model for match outcome prediction.
 
Given offensive and defensive ratings for two teams, estimates expected goal
rates (lambda) and derives match outcome probabilities via Monte Carlo sampling
from independent Poisson distributions.
 
Author: Selma Marrakchi (2026)
"""
 
import numpy as np
import collections
from typing import Union, Tuple
 
 
# ---------------------------------------------------------------------------
# Rating derivation
# ---------------------------------------------------------------------------
 
def elo_to_goal_rates(elo_home: float, elo_away: float,
                      base_rate: float = 1.35,
                      scale: float = 400.0) -> Tuple[float, float]:
    """
    Convert Elo ratings to expected goal rates using a logistic transformation.
 
    The expected goal rate for each team is adjusted from a baseline (mean
    goals per match in international football) by the Elo rating differential.
    Higher-rated teams are assigned higher offensive rates against lower-rated
    defensive opponents.
 
    Parameters
    ----------
    elo_home : float
        Elo rating of the home (or first) team.
    elo_away : float
        Elo rating of the away (or second) team.
    base_rate : float
        Baseline expected goals per team per match (default 1.35, approximate
        international average).
    scale : float
        Elo scale parameter (default 400, consistent with standard Elo systems).
 
    Returns
    -------
    lambda_home, lambda_away : Tuple[float, float]
        Expected goal rates for each team.
    """
    diff = elo_home - elo_away
    weight = 1.0 / (1.0 + 10 ** (-diff / scale))  # logistic function [0, 1]
 
    # Offensive rate scales with relative strength
    lambda_home = base_rate * (1 + 0.5 * (weight - 0.5))
    lambda_away = base_rate * (1 - 0.5 * (weight - 0.5))
 
    return max(lambda_home, 0.1), max(lambda_away, 0.1)
 
 
def spi_to_goal_rates(off_a: float, def_b: float,
                      off_b: float, def_a: float) -> Tuple[float, float]:
    """
    Estimate expected goal rates from SPI offensive and defensive ratings.
 
    Following Maher (1982), the expected goals for each team are estimated
    as the average of the team's offensive output and the opponent's defensive
    vulnerability:
 
        lambda_A = 0.5 * (off_A + def_B)
        lambda_B = 0.5 * (off_B + def_A)
 
    Parameters
    ----------
    off_a, def_b : float
        Offensive rating of team A; defensive rating of team B.
    off_b, def_a : float
        Offensive rating of team B; defensive rating of team A.
 
    Returns
    -------
    lambda_a, lambda_b : Tuple[float, float]
        Expected goal rates.
    """
    lambda_a = 0.5 * (off_a + def_b)
    lambda_b = 0.5 * (off_b + def_a)
    return max(lambda_a, 0.1), max(lambda_b, 0.1)
 
 
# ---------------------------------------------------------------------------
# Match simulation
# ---------------------------------------------------------------------------
 
def simulate_match(lambda_a: float, lambda_b: float,
                   n: int = 10_000,
                   elimination: bool = False,
                   seed: int = None) -> Union[str, Tuple[str, str]]:
    """
    Simulate a match outcome by drawing from independent Poisson distributions.
 
    Parameters
    ----------
    lambda_a : float
        Expected goals for team A.
    lambda_b : float
        Expected goals for team B.
    n : int
        Number of Monte Carlo iterations (default 10,000).
    elimination : bool
        If True, draws are resolved via a Bernoulli tie-break (simulating
        extra time / penalties). If False, draws are a valid outcome.
    seed : int, optional
        Random seed for reproducibility.
 
    Returns
    -------
    winner : str or Tuple[str, str]
        "A" if team A wins, "B" if team B wins, or ("A", "B") on a draw
        (only when elimination=False).
    """
    rng = np.random.default_rng(seed)
 
    goals_a = rng.poisson(lambda_a, n)
    goals_b = rng.poisson(lambda_b, n)
 
    counter_a = collections.Counter(goals_a)
    counter_b = collections.Counter(goals_b)
 
    max_goals = 5
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = (counter_a[i] + counter_b[j]) / (10 * n)
 
    p_a_wins = sum(
        prob_matrix[i, j]
        for i in range(max_goals)
        for j in range(i)
    )
    p_draw = sum(prob_matrix[k, k] for k in range(max_goals))
    p_b_wins = 1.0 - p_a_wins - p_draw
 
    if p_a_wins >= p_b_wins and p_a_wins >= p_draw:
        return "A"
    elif p_draw >= p_a_wins and p_draw >= p_b_wins:
        if elimination:
            return _penalty_tiebreak(rng)
        return ("A", "B")
    else:
        return "B"
 
 
def match_outcome_probabilities(lambda_a: float,
                                lambda_b: float,
                                n: int = 10_000,
                                seed: int = None) -> dict:
    """
    Return the full probability distribution over match outcomes.
 
    Parameters
    ----------
    lambda_a, lambda_b : float
        Expected goal rates.
    n : int
        Monte Carlo iterations.
    seed : int, optional
        Random seed.
 
    Returns
    -------
    dict with keys 'p_a_wins', 'p_draw', 'p_b_wins'.
    """
    rng = np.random.default_rng(seed)
    goals_a = rng.poisson(lambda_a, n)
    goals_b = rng.poisson(lambda_b, n)
 
    p_a_wins = float(np.mean(goals_a > goals_b))
    p_draw   = float(np.mean(goals_a == goals_b))
    p_b_wins = float(np.mean(goals_a < goals_b))
 
    return {"p_a_wins": p_a_wins, "p_draw": p_draw, "p_b_wins": p_b_wins}
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _penalty_tiebreak(rng: np.random.Generator) -> str:
    """
    Resolve a drawn knockout match via a Bernoulli trial (p = 0.5).
 
    Jordet et al. (2007) document that penalty shootout outcomes are
    approximately coin-flip random at the population level.
    """
    return "A" if rng.random() < 0.5 else "B"
 
