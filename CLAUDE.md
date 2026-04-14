# Tennis Prediction Model — Project Spec

## Project Overview

Pre-match tennis outcome prediction for ATP Tour, WTA Tour, and ATP Challenger events.

**Output per prediction:**
- Win probability (Player A vs Player B)
- Key prediction drivers
- Confidence level
- Market edge flagged when model vs closing implied probability delta is meaningful

---

## Data Sources

| Source | Details |
|--------|---------|
| ATP + Challenger | Jeff Sackmann `tennis_atp` GitHub CSVs |
| WTA | Jeff Sackmann `tennis_wta` GitHub CSVs |
| Odds | Pinnacle historical odds or OddsPortal |
| Weather | Match-day API by venue — qualitative flag only |

---

## Tech Stack

- Python 3.11+
- pandas, numpy, scikit-learn, XGBoost, scipy, requests

---

## Feature Tiers

### TIER 1 — CORE (highest weight)

- **Surface-specific Elo**
- **Serve + return stats (surface-specific):**
  - hold%, break%
  - first serve%, first serve points won%, second serve points won%
  - first serve return points won%, second serve return points won%
- **Recent form:** last 10–15 matches, weighted by opponent strength and recency
- **H2H:** last 4 years, surface-specific, minimum 3 matches, heavily weighted. Fallback to surface Elo + serve/return when H2H unavailable
- **Match format:** best-of-3 vs best-of-5

### TIER 2 — STRUCTURAL EDGE (medium weight)

- **Playing style matchup:** big server vs weak returner, aggressor vs counterpuncher, lefty vs righty splits
- **Rally length profile:** win% at 0–4 shots, 5–8 shots, 9+ shots
- **Surface + court speed:** fast vs slow hard, clay variants, grass
- Surface-specific Elo handles surface transition via natural decay
- **Tournament/venue history:** last 4–5 editions
- **Ranking trajectory:** current vs 3 and 6 months ago
- **Scheduling and fatigue:** days rest, previous match duration, travel flag for 5+ hour time zone changes only

### TIER 3 — SITUATIONAL OVERLAY (qualitative adjustment)

- **Injury flag** from observable proxies only: retirement in last match, MTO taken, match duration spike, late withdrawal — treated as a multiplier, not a score
- **Mental/pressure:** break points saved%, break points converted%, tiebreak record, deciding set performance
- **Motivation/incentive:** defending points at this event, rankings race, tournament level
- **Weather:** wind and extreme heat — qualitative override only
- **Crowd/home advantage:** manual note for obvious cases only

---

## Market Layer

- Opening vs closing odds
- Direction and magnitude of line movement
- Sharp-sided movement indicator: closing toward underdog = sharp money
- Model probability vs implied probability delta
- **CLV tracked and logged for every prediction from day one**

---

## Advanced Modeling

- **Interaction effects:** surface × playing style, fatigue × rally length, serve weakness × return strength
- **Point-level decomposition:** hold/break probability per service game from serve + return stats
- **Match simulation:** built after point decomposition is working

> Build interaction effects and match simulation only after core pipeline is working.

---

## Design Rules

1. **Never modify raw data files.** All transformations happen on copies in `data/processed/`.
2. All predictions logged to `output/predictions.csv` with full metadata.
3. CLV logged to `output/clv_tracker.csv` for every prediction.
4. H2H always filtered to last 4 years and minimum 3 matches before being used.
5. Surface-specific Elo is the fallback when any feature has insufficient data.
6. Build interaction effects and match simulation only after core pipeline is working.

---

## Folder Structure

```
tennis-prediction-model/
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                   # Jeff Sackmann CSVs — never modified
│   ├── processed/             # Cleaned, merged, feature-ready data
│   └── odds/                  # Pinnacle / OddsPortal files
├── features/
│   ├── elo.py                 # Surface-specific Elo computation
│   ├── serve_return.py        # Serve/return stat aggregation
│   ├── form.py                # Recent form, recency-weighted
│   ├── h2h.py                 # H2H filtered to 4yr / 3 match min
│   ├── matchup.py             # Playing style matchup features
│   ├── context.py             # Fatigue, scheduling, venue history
│   ├── injury.py              # Injury proxies and multiplier logic
│   ├── market.py              # Odds parsing, implied prob, CLV
│   └── pipeline.py            # Assembles all feature tiers into one row
├── models/
│   ├── train.py               # XGBoost / sklearn training
│   ├── evaluate.py            # Calibration, log loss, Brier score
│   ├── simulate.py            # Match simulation (post point-decomp)
│   └── saved/                 # Serialised model artefacts
├── notebooks/
│   └── exploration.ipynb
├── output/
│   ├── predictions.csv        # Every prediction with full metadata
│   └── clv_tracker.csv        # CLV log from day one
└── utils/
    ├── data_loader.py         # Load + merge Sackmann CSVs
    └── helpers.py             # Shared utilities
```

---

## Output Schema

### predictions.csv

| Column | Description |
|--------|-------------|
| match_id | Unique match identifier |
| date | Match date |
| tournament | Tournament name |
| surface | hard / clay / grass |
| player_a | Player A name |
| player_b | Player B name |
| prob_a | Model win probability for Player A |
| prob_b | Model win probability for Player B |
| key_drivers | Top features driving the prediction |
| confidence | low / medium / high |
| model_edge | prob_a minus closing implied prob |
| closing_odds_a | Closing decimal odds Player A |
| closing_odds_b | Closing decimal odds Player B |

### clv_tracker.csv

| Column | Description |
|--------|-------------|
| match_id | Unique match identifier |
| date | Match date |
| player_a | Player A name |
| predicted_prob | Model probability at prediction time |
| opening_implied | Opening implied probability |
| closing_implied | Closing implied probability |
| clv | Closing line value (predicted_prob minus closing_implied) |
| result | Actual match result (filled post-match) |
