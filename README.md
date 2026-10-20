# League of Legends match forecasting

A probabilistic model that forecasts professional *League of Legends* matches: the chance of each team winning a single game, a best-of-3 or best-of-5 series, and each exact scoreline. It is built from about 41,000 professional games (2023 to 2026) across every major league and international event, and is benchmarked against both a standard Elo rating and market prices.

The core is a **Bayesian dynamic rating model**. Each team's strength and each league's overall level is a Normal belief (a best estimate plus an uncertainty). Every game updates those beliefs, and the uncertainty grows while a team is idle or changes players.

## Results

Every game is predicted *before* it is played (walk-forward), using only earlier games. The scores below are for 2026. Run `python evaluate.py` to reproduce them.

| 2026 games | Games | Elo log loss | Model log loss | Model accuracy | Model AUC |
|---|---|---|---|---|---|
| All leagues | 9,644 | 0.638 | **0.609** | 66.0% | 0.723 |
| Top leagues + international | 2,823 | 0.652 | **0.639** | 63.4% | 0.676 |
| International events only | 399 | 0.636 | **0.592** | 66.9% | 0.736 |
| January–March (new rosters) | 2,430 | 0.662 | **0.610** | 65.7% | 0.724 |

Log loss: lower is better, and a coin flip scores 0.693.

- **Calibrated:** when the model says 72%, the team wins about 71% of the time. Across seven probability bands, predicted and actual rates in 2026 differ by at most 2.6 points.
- **Market benchmark:** I compared the model with Pinnacle's prices on 185 top-tier series in 2026. It was about as accurate as the opening price (log loss 0.603 against 0.608), but less accurate than the closing price (0.589). The closing price includes late information, such as confirmed line-ups, that a model built only on results does not have.

## How it works

**1. Ratings.** A team's strength in a game is its own rating plus its league's level. The model works in log-odds (logit) terms:

$$P(\text{blue wins}) = \sigma\!\left(\frac{(\theta_b + \mu_{L_b}) - (\theta_r + \mu_{L_r}) + h}{\sqrt{1 + \pi v / 8}}\right)$$

- θ is the team rating and μ the league level; h is the blue-side advantage.
- v is the combined uncertainty. Uncertainty pulls the probability towards 50%, which is the probit approximation to averaging over the beliefs.
- After each game, the beliefs are updated by assumed-density filtering, a one-step Bayesian update that keeps every belief Normal.

**2. Margin of victory.** Gold difference at 15 minutes, and the end-of-game gold-spent difference, are each treated as a noisy linear measurement of the strength gap. They are added with a Kalman update, so a dominant win teaches the model more than a narrow one.

**3. Things that change over time.**
- Uncertainty grows by a fixed amount per idle day (a random walk).
- A roster change adds uncertainty for each new player.
- Ratings shrink towards average between seasons.
- A rebranded team inherits its old rating when at least three of five starters carry over.
- Off-season cups count as half a game.

**4. League levels.** These are learned from games between leagues. Games between a top-league team and a lower-tier team (promotion tournaments, for example) count as a quarter of a game. They say little about the top of a league, and on held-out data this improved predictions for international games between top-league teams.

**5. Series and scorelines.** Games in a series aren't independent: teams have good and bad days. Each series shares one "form on the day" shock e ~ N(0, s²). The model solves for a latent z so that E[σ(z + e)] equals the single-game probability, keeping that probability unchanged, then integrates with 40-point Gauss–Hermite quadrature.
- s is 0.7 when both teams are from top leagues and 1.0 otherwise.
- It was fitted on 2023–24 scorelines and tested on 2025–26.
- It fixed over-confident best-of-5 predictions (log loss improved by 0.0135 ± 0.0033) and the under-prediction of 3–0 and 2–0 sweeps.

**6. Market comparison.** Bookmaker margins are removed with the power method, which finds the k where q₁ᵏ + q₂ᵏ = 1, and the fair probabilities are compared with the model's.

## What was tested and left out

Each idea below was fitted on earlier seasons and scored on later ones. It was kept only if it improved held-out log loss.

| Idea | Result |
|---|---|
| Recent form in objective stats (first blood, dragons, towers, gold and XP at 10 minutes) and player form | No gain over the ratings for predicting the winner (`research/feature_test.py`, `idea_tests.py`) |
| Weighting playoff games more heavily | Helped early in the season, hurt later; left off (`research/tune_weights.py`) |
| Team-specific red/blue-side strength | No persistence from one season to the next; made 2026 predictions worse (`research/side_team_test.py`) |
| Separate levels for the top and bottom of each league | Made international predictions worse (`research/league_tilt.py`) |
| Re-estimating the series "form" spread over time | No better than the fixed values (`research/adaptive_spread.py`) |
| Champion comfort and melee v ranged mid laners | Mostly explained by player experience; no stable champion effect (`research/meta_review.py`) |
| **Line-up experience** (average career games of the five starters) | **Promising: −0.0079 ± 0.0013 log loss on 2026 when fitted on 2025. Not yet in the model** (`research/experience_test.py`) |

## Limitations

- **Some settings were chosen after seeing 2026 data**, so the 2026 scores are slightly optimistic. The core settings were tuned on earlier seasons.
- **League levels are uncertain.** They rest on a few hundred games between leagues each year.
- **Results only.** The model doesn't know about late line-up news, illness or substitutes, which is much of why closing market prices are sharper.

## Running it

```bash
pip install -r requirements.txt
```

1. Download the match data CSVs for 2023–2026 from [Oracle's Elixir](https://oracleselixir.com/tools/downloads) into this folder. Keep the downloaded file names, e.g. `2026_LoL_esports_match_data_from_OraclesElixir.csv`.
2. Run the commands:

```bash
python evaluate.py                                   # the results table above
python predict.py rankings                           # current team ratings
python predict.py match "Gen.G" "Bilibili Gaming" --bo 5
python simulate_playin.py                            # Monte Carlo of a 4-team double-elimination bracket
python export_ratings.py && python pricer/build_pricer.py   # rebuild the interactive page
```

`pricer/rift_pricer.html` is a self-contained page: pick two teams and a format to see the model's probabilities, scorelines and full working. It contains a snapshot of the ratings as of October 2026.

## Repository layout

| Path | Contents |
|---|---|
| `ratings.py` | The model: rating engine, data loading, series functions |
| `series_math.py` | "Form on the day" series and scoreline maths (Gauss–Hermite) |
| `evaluate.py` | Walk-forward evaluation against Elo |
| `predict.py` | Rankings and match pricing from the command line |
| `simulate_playin.py` | Tournament bracket simulation (team strengths drawn from the posterior) |
| `daily_review.py` | Compares predictions with results for newly added games |
| `fetch_odds.py`, `market_test.py` | Collect historical odds (OddsPapi API) and compare model with market |
| `pricer/` | Interactive single-page pricer (JavaScript port of the model) |
| `research/` | Every experiment behind the decisions above; run from the repo root, e.g. `python research/league_tilt.py` |

Some research scripts need intermediate files. Run `python make_predictions.py` first for `meta_review.py` and `experience_test.py`. For the series work, run `series_data.py`, then `series_fit.py`, before `series_eval.py`, `scoreline_check.py` and `adaptive_spread.py`.

## Data

Match data is from [Oracle's Elixir](https://oracleselixir.com) by Tim Sevenhuysen. Historical odds come from [OddsPapi](https://oddspapi.io). Neither dataset is included in this repository.

---
Sam Diggens · Economics graduate, University of Kent
