import numpy as np
import pandas as pd
import collections
import itertools
import math
import time

np.random.seed(42)

# ── Load & preprocess Cruz (2026) Kaggle dataset ───────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/elo_ratings_wc2026.csv')
df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
df = df[df['snapshot_date'] <= pd.Timestamp('2026-06-10')]
df = df.sort_values('snapshot_date')
df = df.groupby('country').last().reset_index()
# native column names: country, rating

# Official 2026 group assignments — names matched exactly to dataset
group_map = {
    'Mexico':'A', 'South Korea':'A', 'South Africa':'A', 'Czechia':'A',
    'Canada':'B', 'Switzerland':'B', 'Qatar':'B', 'Bosnia and Herzegovina':'B',
    'Brazil':'C', 'Morocco':'C', 'Scotland':'C', 'Haiti':'C',
    'United States':'D', 'Turkey':'D', 'Australia':'D', 'Paraguay':'D',
    'Germany':'E', 'Ivory Coast':'E', 'Ecuador':'E', 'Curaçao':'E',
    'Netherlands':'F', 'Japan':'F', 'Sweden':'F', 'Tunisia':'F',
    'Belgium':'G', 'Egypt':'G', 'Iran':'G', 'New Zealand':'G',
    'Spain':'H', 'Cape Verde':'H', 'Saudi Arabia':'H', 'Uruguay':'H',
    'France':'I', 'Senegal':'I', 'Iraq':'I', 'Norway':'I',
    'Argentina':'J', 'Algeria':'J', 'Austria':'J', 'Jordan':'J',
    'Portugal':'K', 'DR Congo':'K', 'Uzbekistan':'K', 'Colombia':'K',
    'England':'L', 'Croatia':'L', 'Ghana':'L', 'Panama':'L',
}

df['group'] = df['country'].map(group_map)
missing = df[df['group'].isna()]['country'].tolist()
if missing:
    print(f"WARNING — unmapped: {missing}")
df = df.dropna(subset=['group'])
assert len(df) == 48, f"Expected 48 teams, got {len(df)}"

# Derive off/def from Elo
BASE_RATE, SCALE = 1.35, 400.0
mean_rating = df['rating'].mean()
df['off'] = (BASE_RATE * (1 + 0.5*(df['rating']-mean_rating)/SCALE)).clip(0.3, 3.5)
df['def'] = (BASE_RATE * (1 - 0.5*(df['rating']-mean_rating)/SCALE)).clip(0.3, 3.5)

print(f"Dataset: Cruz (2026) Kaggle | Snapshot: 2026-05-27 | Teams: {len(df)}")
print(f"Mean Elo: {mean_rating:.0f} | Range: {df['rating'].min()}-{df['rating'].max()}")
print(f"\nTop 10 by Elo:")
print(df.sort_values('rating', ascending=False)[['country','rating','off','def','group']].head(10).to_string(index=False))

groups  = {g: gdf['country'].tolist() for g, gdf in df.groupby('group')}
ratings = df.set_index('country')[['off','def']].to_dict('index')

# ── Precompute pairwise match probabilities ────────────────────────────────
FAC = [math.factorial(i) for i in range(15)]

def poisson_probs(la, lb, nmax=10):
    pa_win = pd_ = 0.0
    for i in range(nmax+1):
        poi_a = math.exp(-la) * la**i / FAC[i]
        for j in range(nmax+1):
            poi_b = math.exp(-lb) * lb**j / FAC[j]
            if i > j:    pa_win += poi_a * poi_b
            elif i == j: pd_    += poi_a * poi_b
    return pa_win, pd_, 1-pa_win-pd_

def get_lambdas(a, b):
    return (0.5*(ratings[a]['off']+ratings[b]['def']),
            0.5*(ratings[b]['off']+ratings[a]['def']))

all_teams = df['country'].tolist()
prob_cache = {}
for a, b in itertools.combinations(all_teams, 2):
    la, lb = get_lambdas(a, b)
    prob_cache[(a,b)] = poisson_probs(la, lb)
    prob_cache[(b,a)] = (prob_cache[(a,b)][2], prob_cache[(a,b)][1], prob_cache[(a,b)][0])

print(f"\nCached {len(prob_cache)} pairwise match probabilities")

# ── Match / group / knockout functions ─────────────────────────────────────
def sim_match(a, b, elimination=False):
    pa, pd_, pb = prob_cache[(a,b)]
    r = np.random.random()
    if r < pa: return a
    elif r < pa+pd_:
        return (a if np.random.random()<0.5 else b) if elimination else (a,b)
    return b

def group_stage(group):
    pts={t:0 for t in group}; gd={t:0 for t in group}; gf={t:0 for t in group}
    for a,b in itertools.combinations(group,2):
        la,lb = get_lambdas(a,b)
        sa,sb = np.random.poisson(la), np.random.poisson(lb)
        gf[a]+=sa; gf[b]+=sb; gd[a]+=sa-sb; gd[b]+=sb-sa
        if sa>sb: pts[a]+=3
        elif sa<sb: pts[b]+=3
        else: pts[a]+=1; pts[b]+=1
    ranked = sorted(group, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
    return ranked[0], ranked[1], ranked[2], {t:{'pts':pts[t],'gd':gd[t],'gf':gf[t]} for t in group}

def run_knockout(teams):
    rem = list(teams)
    while len(rem)>1:
        rem = [sim_match(rem[i],rem[i+1],elimination=True) for i in range(0,len(rem),2)]
    return rem[0]

def simulate_tournament():
    group_results, thirds = {}, []
    for g, group in groups.items():
        f,s,t,standings = group_stage(group)
        group_results[g]=(f,s)
        thirds.append((t, standings[t]))
    thirds.sort(key=lambda x:(x[1]['pts'],x[1]['gd'],x[1]['gf']),reverse=True)
    r32 = [t for f,s in group_results.values() for t in (f,s)] + [t for t,_ in thirds[:8]]
    return run_knockout(r32)

# ── Run 100,000 simulations ────────────────────────────────────────────────
N = 100_000
win_counts = collections.defaultdict(int)
t0 = time.time()
print(f"\nRunning {N:,} simulations...")

for i in range(N):
    win_counts[simulate_tournament()] += 1
    if (i+1) % 20000 == 0:
        el = time.time()-t0
        print(f"  {i+1:,}/{N:,} | {(i+1)/el:.0f} sims/s | ETA {(N-i-1)/((i+1)/el):.0f}s")

results = sorted(win_counts.items(), key=lambda x:x[1], reverse=True)
elapsed = time.time()-t0

print(f"\n{'Rank':<6}{'Team':<35}{'Wins':>8}{'Win %':>10}")
print('─'*61)
for rank,(team,wins) in enumerate(results,1):
    print(f"{rank:<6}{team:<35}{wins:>8,}{wins/N:>9.1%}")

out = pd.DataFrame([(t,w,round(w/N,4)) for t,w in results],
                   columns=['team','wins','win_probability'])
out.to_csv('/home/claude/win_probabilities_100k_final.csv', index=False)
print(f"\nCompleted {N:,} simulations in {elapsed:.1f}s | Seed: 42")
print(f"Source: Cruz (2026) Kaggle — elo_ratings_wc2026.csv — snapshot 2026-05-27")
