# Probabilistic Outcome Prediction with Poisson Regression and Monte Carlo Methods
## Applied to the 2026 FIFA World Cup

**Author:** Selma Marrakchi  
**Published:** June 2026  
**Companion article:** 

---

## Overview

This repository contains the full implementation of a probabilistic framework for predicting match outcomes and tournament results in the 2026 FIFA World Cup. The model combines:

- **Poisson regression** for match-level goal rate estimation, using Elo-derived offensive and defensive ratings
- **Monte Carlo simulation** (N = 100,000 runs) for bracket-level tournament inference
- **Bayesian updating** (optional extension) to revise rating estimates after observed match results

The 2026 World Cup introduces structural novelty — 48 teams, 12 groups, a dual qualification pathway (top 2 per group + 8 best third-place finishers), and a new Round of 32 — that makes full tournament simulation meaningfully more complex than prior editions. This implementation handles the complete bracket structure including third-place qualification logic.

---

## Repository Structure

```
wc2026-poisson-monte-carlo/
│
├── data/
│   ├── elo_ratings_wc2026.csv          # Pre-tournament Elo ratings (Cruz, 2026)
│   └── groups_2026.csv                 # Official group assignments (Groups A–L)
│
├── results/
│   └── win_probabilities.csv           # Simulated win probabilities per team
│
├── simulate.py                         # Main simulation script
├── model.py                            # Poisson model and match prediction functions
├── tournament.py                       # Tournament bracket logic (group stage + knockout)
├── bayesian.py                         # Bayesian updating after observed matches
├── utils.py                            # Data loading and preprocessing
├── preprocess.py                       # Standalone preprocessing script
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Data Sources

### Primary: World Football Elo Ratings (2026 World Cup)
- **Kaggle dataset:** [2026 FIFA World Cup — Historical Elo Ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)
- **Author:** Afonso Fernandes Cruz (2026)
- **Upstream source:** World Football Elo Ratings (eloratings.net), maintained by Kirill Bukin and Erik Gebhardt
- **License:** CC BY-SA 4.0
- **Coverage:** 125 years of Elo ratings for all 48 qualified teams (1901–2026)
- **Citation:**
```
Cruz, A. F. (2026). 2026 FIFA World Cup — Historical Elo Ratings [Dataset]. 
Kaggle. https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
```

---

## Installation

```bash
git clone https://github.com/m-selma/wc2026-poisson-monte-carlo.git
cd wc2026-poisson-monte-carlo
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocess data

```bash
# If you've downloaded the CSV manually into data/:
python preprocess.py --input data/elo_ratings_wc2026.csv

# Or download directly from Kaggle (requires Kaggle API credentials):
python preprocess.py --download
```

This runs the full preprocessing pipeline:
1. Filters to pre-tournament snapshots (≤ June 10, 2026)
2. Retains the most recent snapshot per team
3. Filters to the 48 qualified teams
4. Derives offensive/defensive ratings from Elo
5. Validates and saves to `data/ratings_preprocessed.csv`

### Step 2 — Run the full tournament simulation

```bash
python simulate.py --n_simulations 100000 --data data/ratings_preprocessed.csv
```

Results are written to `results/win_probabilities.csv`.

### Step 3 (optional) — Bayesian updating after observed matches

After each knockout round, update win probabilities using observed scorelines:

```bash
python bayesian.py \
    --ratings data/ratings_preprocessed.csv \
    --results data/group_stage_results.csv \
    --remaining data/remaining_teams.txt \
    --prior_strength 5.0 \
    --n_simulations 100000
```

`--prior_strength` controls how many virtual matches the pre-tournament rating represents. Higher values (10–20) make ratings more resistant to updating; lower values (1–3) let the model adapt quickly to tournament form. Default is 5.

---

## Methodology

### Goal Rate Estimation

For a match between team A and team B, expected goal rates are estimated as:

```
λ_A = 0.5 * (off_A + def_B)
λ_B = 0.5 * (off_B + def_A)
```

Where `off` and `def` are derived from each team's Elo rating via logistic normalization relative to the field mean:

```
off = 1.35 * (1 + 0.5 * (elo - mean_elo) / 400)
def = 1.35 * (1 - 0.5 * (elo - mean_elo) / 400)
```

### Match Simulation

Match outcome probabilities are computed analytically by summing over the joint Poisson distribution up to n = 10 goals per team, then cached for all 2,256 pairwise combinations before the simulation loop begins.

### Third-Place Qualification (2026-specific)

The eight best third-place finishers advance to the Round of 32. In each simulation run, third-place teams are ranked globally across all 12 groups by: (1) points, (2) goal differential, (3) goals scored. Only the top 8 advance.

### Knockout Rounds

Drawn matches are resolved via a Bernoulli trial (p = 0.5).

---

## References

Cruz, A. F. (2026). *2026 FIFA World Cup — Historical Elo Ratings* [Dataset]. Kaggle.

Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.

Lasek, J., & Gagolewski, M. (2021). Interpretable sports team rating models based on the gradient descent algorithm. *International Journal of Forecasting*, 37(3), 1061–1071.

Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.

---

## License

MIT License. Data files are subject to their respective upstream licenses (CC BY-SA 4.0 — see Data Sources above).

---

## Citation

```
Marrakchi, S. (2026). Probabilistic Outcome Prediction with Poisson Regression and 
Monte Carlo Methods: Applied to the 2026 FIFA World Cup. 
GitHub. https://github.com/selmamarrakchi/wc2026-poisson-monte-carlo
```
