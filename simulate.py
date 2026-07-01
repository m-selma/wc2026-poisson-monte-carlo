"""
simulate.py
-----------
Main entry point for the 2026 FIFA World Cup Monte Carlo simulation.

Usage
-----
    python simulate.py --n_simulations 100000 --data data/ratings_preprocessed.csv

Author: Selma Marrakchi (2026)
"""

import argparse
import collections
import itertools
import math
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Group assignments — 2026 FIFA World Cup (exact dataset names)
# ---------------------------------------------------------------------------

GROUPS_2026 = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czechia'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Turkey', 'Australia', 'Paraguay'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

FAC = [math.factorial(i) for i in range(15)]


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def poisson_probs(la, lb, nmax=10):
    """Analytically compute win/draw/loss probabilities from Poisson rates."""
    pa_win = pd_ = 0.0
    for i in range(nmax + 1):
        poi_a = math.exp(-la) * la**i / FAC[i]
        for j in range(nmax + 1):
            poi_b = math.exp(-lb) * lb**j / FAC[j]
            if i > j:    pa_win += poi_a * poi_b
            elif i == j: pd_    += poi_a * poi_b
    return pa_win, pd_, 1 - pa_win - pd_


def build_prob_cache(all_teams, ratings):
    """Precompute pairwise match probabilities for all team combinations."""
    cache = {}
    for a, b in itertools.combinations(all_teams, 2):
        la = 0.5 * (ratings[a]['off'] + ratings[b]['def'])
        lb = 0.5 * (ratings[b]['off'] + ratings[a]['def'])
        cache[(a, b)] = poisson_probs(la, lb)
        cache[(b, a)] = (cache[(a,b)][2], cache[(a,b)][1], cache[(a,b)][0])
    return cache


def sim_match(a, b, prob_cache, elimination=False):
    pa, pd_, pb = prob_cache[(a, b)]
    r = np.random.random()
    if r < pa:       return a
    elif r < pa+pd_: return (a if np.random.random() < 0.5 else b) if elimination else (a, b)
    else:            return b


def group_stage(group, ratings):
    pts = {t: 0 for t in group}
    gd  = {t: 0 for t in group}
    gf  = {t: 0 for t in group}
    for a, b in itertools.combinations(group, 2):
        la = 0.5 * (ratings[a]['off'] + ratings[b]['def'])
        lb = 0.5 * (ratings[b]['off'] + ratings[a]['def'])
        sa, sb = np.random.poisson(la), np.random.poisson(lb)
        gf[a] += sa; gf[b] += sb
        gd[a] += sa - sb; gd[b] += sb - sa
        if sa > sb:   pts[a] += 3
        elif sa < sb: pts[b] += 3
        else:         pts[a] += 1; pts[b] += 1
    ranked = sorted(group, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    standings = {t: {'pts': pts[t], 'gd': gd[t], 'gf': gf[t]} for t in group}
    return ranked[0], ranked[1], ranked[2], standings


def run_knockout(teams, prob_cache):
    remaining = list(teams)
    while len(remaining) > 1:
        remaining = [
            sim_match(remaining[i], remaining[i+1], prob_cache, elimination=True)
            for i in range(0, len(remaining), 2)
        ]
    return remaining[0]


def simulate_tournament(groups, ratings, prob_cache):
    group_results, thirds = {}, []
    for g, group in groups.items():
        first, second, third, standings = group_stage(group, ratings)
        group_results[g] = (first, second)
        thirds.append((third, standings[third]))
    thirds.sort(key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']), reverse=True)
    r32 = [t for f, s in group_results.values() for t in (f, s)]
    r32 += [t for t, _ in thirds[:8]]
    return run_knockout(r32, prob_cache)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='2026 FIFA World Cup Monte Carlo simulation'
    )
    parser.add_argument('--data', required=True,
                        help='Path to preprocessed ratings CSV')
    parser.add_argument('--n_simulations', type=int, default=100_000)
    parser.add_argument('--output', default='results/win_probabilities.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load preprocessed ratings
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} teams from {args.data}")
    print(f"Elo range: {df['rating'].min():.0f}–{df['rating'].max():.0f}\n")

    ratings = df.set_index('country')[['off', 'def']].to_dict('index')
    all_teams = df['country'].tolist()

    # Precompute pairwise probabilities
    prob_cache = build_prob_cache(all_teams, ratings)
    print(f"Cached {len(prob_cache)} pairwise match probabilities")

    # Run simulation
    win_counts = collections.defaultdict(int)
    t0 = time.time()
    N = args.n_simulations
    print(f"Running {N:,} simulations...\n")

    for i in range(N):
        win_counts[simulate_tournament(GROUPS_2026, ratings, prob_cache)] += 1
        if (i + 1) % 20_000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,}/{N:,} | {(i+1)/el:.0f} sims/s | ETA {(N-i-1)/((i+1)/el):.0f}s")

    elapsed = time.time() - t0
    results = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6}{'Team':<35}{'Wins':>8}{'Win %':>10}")
    print('─' * 61)
    for rank, (team, wins) in enumerate(results, 1):
        print(f"{rank:<6}{team:<35}{wins:>8,}{wins/N:>9.1%}")

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(t, w, round(w/N, 4)) for t, w in results],
        columns=['team', 'wins', 'win_probability']
    ).to_csv(args.output, index=False)

    print(f"\nCompleted {N:,} simulations in {elapsed:.1f}s | seed={args.seed}")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
