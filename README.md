# Probabilistic Outcome Prediction with Poisson Regression and Monte Carlo Methods
## Applied to the 2026 FIFA World Cup

**Author:** Selma Marrakchi  
**Published:** June 2026  
**Companion article:**
---

## Overview

This repository contains the full implementation of a probabilistic framework for predicting match outcomes and tournament results in the 2026 FIFA World Cup. The model combines:

- **Poisson regression** for match-level goal rate estimation, using Elo-derived offensive and defensive ratings
- **Monte Carlo simulation** (N = 10,000 runs) for bracket-level tournament inference
- **Bayesian updating** (optional extension) to revise rating estimates after observed group stage results

The 2026 World Cup introduces structural novelty — 48 teams, 12 groups, a dual qualification pathway (top 2 per group + 8 best third-place finishers), and a new Round of 32 — that makes full tournament simulation meaningfully more complex than prior editions. This implementation handles the complete bracket structure including third-place qualification logic.

---

## Repository Structure

```
wc2026-poisson-monte-carlo/
│
├── data/
│   ├── elo_ratings_wc2026.csv          # Pre-tournament Elo ratings for all 48 teams
│   ├── spi_ratings_2026.csv            # SPI offensive/defensive ratings (derived)
│   ├── groups_2026.csv                 # Group assignments (Groups A–L)
│   └── group_stage_results.csv         # Observed group stage results (post-tournament)
│
├── notebooks/
│   └── exploration.ipynb               # Exploratory analysis and rating visualization
│
├── results/
│   ├── win_probabilities.csv           # Simulated win probabilities per team
│   └── group_stage_predictions.csv     # Pre-tournament group stage predictions
│
├── simulate.py                         # Main simulation script
├── model.py                            # Poisson model and match prediction functions
├── tournament.py                       # Tournament bracket logic (group stage + knockout)
├── bayesian.py                         # Bayesian updating after observed matches
├── utils.py                            # Helper functions
├── preprocess.py                       # Data preprocessing script
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
- **Coverage:** 125 years of Elo ratings for all 48 qualified teams (1901–2026), updated daily during the tournament
- **Citation:**
```
Cruz, A. F. (2026). 2026 FIFA World Cup — Historical Elo Ratings [Dataset]. 
Kaggle. https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
```

### Secondary: Live Match Results
- **Kaggle dataset:** [FIFA World Cup 2026 — Live Results & Updated Stats](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset)
- **Author:** MD Mominul Islam (2026)
- **License:** CC0 1.0 Universal (Public Domain)
- **Coverage:** Real match results, xG, squads, match events — updated daily

---

## Installation

```bash
git clone https://github.com/selmamarrakchi/wc2026-poisson-monte-carlo.git
cd wc2026-poisson-monte-carlo
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Download and preprocess data

```bash
# Download from Kaggle (requires Kaggle API credentials configured)
python preprocess.py --download

# Or if you've already downloaded the CSV manually:
python preprocess.py --input data/elo_ratings_wc2026.csv
```

This runs the full preprocessing pipeline:
1. Normalizes column names from the Kaggle schema
2. Filters to pre-tournament snapshots (≤ June 10, 2026)
3. Retains the most recent snapshot per team
4. Filters to the 48 qualified teams
5. Derives offensive/defensive ratings from Elo
6. Validates and saves to `data/ratings_preprocessed.csv`

### Step 2 — Run the full tournament simulation

```bash
python simulate.py --n_simulations 100000 --data data/ratings_preprocessed.csv
```

### Step 3 (optional) — Bayesian updating after observed matches

After each knockout round, update win probabilities using observed scorelines:

```bash
# Create a CSV of observed results with columns: team_a,goals_a,team_b,goals_b
# Example: data/group_stage_results.csv

python bayesian.py \
    --ratings data/ratings_preprocessed.csv \
    --results data/group_stage_results.csv \
    --remaining data/remaining_teams.txt \
    --prior_strength 5.0 \
    --n_simulations 100000
```

`--prior_strength` controls how many virtual matches the pre-tournament rating represents. Higher values (10–20) make ratings more resistant to updating; lower values (1–3) let the model adapt quickly to tournament form. Default is 5.

```bash
python simulate.py --n_simulations 10000 --data data/ratings_preprocessed.csv
```

### Output

Results are written to `results/win_probabilities.csv`:

```
team,win_probability,r32_probability,r16_probability,qf_probability,sf_probability,final_probability
France,0.182,0.94,0.71,0.52,0.38,0.26
Argentina,0.167,0.96,0.74,0.54,0.39,0.27
Spain,0.134,0.91,0.67,0.48,0.34,0.22
...
```

### Run group stage only

```bash
python simulate.py --stage group --n_simulations 10000
```

### Update with observed results (Bayesian mode)

```bash
python simulate.py --bayesian --results data/group_stage_results.csv
```

---

## Methodology

### Goal Rate Estimation

For a match between team A and team B, expected goal rates are estimated as:

```
λ_A = 0.5 * (off_A + def_B)
λ_B = 0.5 * (off_B + def_A)
```

Where `off` and `def` are derived from Elo ratings using a calibrated mapping to offensive/defensive goal rates, following the approach of Maher (1982) and Dixon & Coles (1997).

### Match Simulation

Each match is simulated by drawing independently from:

```
G_A ~ Poisson(λ_A)
G_B ~ Poisson(λ_B)
```

Over N = 10,000 iterations. Win/draw/loss probabilities are derived from the empirical joint distribution of simulated scorelines.

### Third-Place Qualification (2026-specific)

The eight best third-place finishers advance to the Round of 32. In each simulation run, third-place teams are ranked globally across all 12 groups by: (1) points, (2) goal differential, (3) goals scored. Only the top 8 advance.

### Knockout Rounds

Drawn matches in the knockout stage are resolved via a Bernoulli trial (p = 0.5), reflecting the near-random nature of penalty shootouts empirically documented in the literature.

---

## References

Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265–280.

Jordet, G., Hartman, E., Visscher, C., & Lemmink, K. A. P. M. (2007). Kicks from the penalty mark in soccer: The roles of stress, skill, and fatigue for kick outcomes. *Journal of Sports Sciences*, 25(2), 121–129.

Lasek, J., & Gagolewski, M. (2021). Interpretable sports team rating models based on the gradient descent algorithm. *International Journal of Forecasting*, 37(3), 1061–1071. https://doi.org/10.1016/j.ijforecast.2020.11.008

Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.

---

## License

MIT License. Data files are subject to their respective upstream licenses (CC BY-SA 4.0 and CC0 1.0 — see Data Sources above).

---

## Citation

If you use this code or methodology, please cite:

```
Marrakchi, S. (2026). Probabilistic Outcome Prediction with Poisson Regression and 
Monte Carlo Methods: Applied to the 2026 FIFA World Cup. 
GitHub. https://github.com/selmamarrakchi/wc2026-poisson-monte-carlo
```
