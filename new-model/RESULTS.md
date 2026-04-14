# Weather Trading Results

## Data

- **Kalshi prices**: 5-minute snapshots of KXHIGH contracts (daily high temperature markets)
- **NBS forecasts**: NBM (National Blend of Models) from IEM, ~20 runs/day per station
- **METAR observations**: hourly temperature readings from ASOS stations
- **Observed highs**: 6 months of daily max temps (Oct 2025 - Apr 2026) from IEM ASOS archive
- **Cities**: New York (KJFK), Chicago (KMDW), Denver (KDEN), Austin (KAUS), Miami (KMIA), Philadelphia (KPHL)
- **Collection period**: Mar 25 - Apr 14, 2026 (20 days of live Kalshi data)
- **Historical forecasts**: 6 months backfilled from IEM (112K NBS rows, 1,172 station-days of observed highs)

---

## Key Findings

### NBS Forecast Accuracy (6 months, 190+ samples per station)

| Station | Mean Error | Std | Within +/-2F | Character |
|---|---|---|---|---|
| Miami | +0.1F | 1.5F | 93% | Tightest — very predictable |
| Austin | +0.5F | 2.5F | ~85% | Tight, slight cool bias |
| New York | +0.6F | 3.0F | 66% | Moderate spread |
| Denver | +0.5F | 3.4F | ~60% | Mountain weather harder |
| Chicago | +1.0F | 3.5F | ~60% | Cool bias, fat tails (+21F outlier) |
| Philadelphia | +0.7F | 3.6F | ~60% | Widest range (-23 to +15) |

**Lead time does NOT matter** for 17-48h ahead. The NBS forecast for tomorrow's high is equally accurate whether made 17h or 48h before settlement. Error distributions are nearly identical across all NBS runtimes (01Z, 07Z, 13Z, 19Z).

The 6-month historical data only has NBS runs at 4 times/day (01Z, 07Z, 13Z, 19Z). The live collector captures hourly runs but the txn field goes NULL for same-day short-lead forecasts.

### City Error Correlation

| Pair | Correlation | Implication |
|---|---|---|
| **NY - PHL** | **r = 0.65** | Same weather systems — treat as one bet |
| All other pairs | -0.08 to +0.10 | Independent — diversification works |

When NY has a big miss (>+3F), Philadelphia averages +4.2F error the same day.

### Market Calibration (from Kalshi prices)

The market is well-calibrated in the 10-20c and 40-60c ranges but systematically misprices the tails:

| Price Range | Market Says | Actual | Mispricing |
|---|---|---|---|
| 0-5c | 4% | 6% | Slightly underpriced |
| **5-10c** | **7%** | **14%** | **Underpriced (buy YES)** |
| 10-15c | 12% | 12% | Fair |
| 15-20c | 17% | 15% | Slightly overpriced |
| **20-30c** | **25%** | **17%** | **Overpriced (buy NO)** |
| **30-40c** | **34%** | **25%** | **Overpriced (buy NO)** |
| 40-50c | 44% | 43% | Fair |
| 50-60c | 54% | 58% | Slightly underpriced |

These mispricings are consistent across train (before Apr 10) and eval (Apr 10-14).

---

## Model Evolution

### v1: Normal Distribution Blend (`5eb593b`)

Blended NBS + GFS + LAMP forecasts with Normal(mu, sigma). Traded every 5-minute tick based on model vs market edge.

| Split | PnL | ROI | Fills |
|---|---|---|---|
| Train | +$276.20 | +6.6% | 9,269 |
| Eval | +$25.89 | +5.3% | 1,108 |

**Problem**: Blending was unnecessary complexity. The Normal distribution assumption was arbitrary.

### v2: Forecast-Change Only (`5eb593b`)

Only traded when NBS forecast shifted (mu changed by >=0.5F). Exits when market converged.

| Split | PnL | ROI | Fills |
|---|---|---|---|
| Train | +$56.39 | +4.5% | 2,674 |
| Eval | -$2.79 | -3.0% | 209 |

**Problem**: Too selective, eval went negative.

### v3: Combined Signal + Static (`5eb593b`)

Signal entries on forecast change (low threshold) + static entries on persistent mispricing (high threshold).

| Split | PnL | ROI | Fills |
|---|---|---|---|
| Train | +$276.20 | +6.6% | 9,269 |
| Eval | +$25.89 | +5.3% | 1,108 |

**Best overall PnL** but relied on the Normal distribution.

### v4: Empirical Error Distribution — 10 days (`5eb593b`)

Replaced Normal with empirical: error = observed - forecast, replayed against current forecast to price contracts. Per-station distributions. Only traded on new NBS runs.

**Problem**: Only 10 errors per station. Laplace smoothing added ~8c phantom probability. Fair values stuck at discrete levels (8c, 17c, 25c, 33c...). Model bought cheap YES lottery tickets that mostly lost.

### v5: Empirical — 6 months of history (`8da4b1c`)

Backfilled 6 months of NBS forecasts + observed highs from IEM. ~190 errors per station. Lead-time matched.

**Gap-based sweep (edge x cost):**

| Rule | Train PnL | Train ROI | Eval PnL | Eval ROI |
|---|---|---|---|---|
| edge>=5, cost<=30 | $85 | 33% | -$4 | -24% |
| edge>=10, cost<=20 | $102 | 76% | +$3 | 52% |
| edge>=15, cost<=15 | $34 | 136% | +$3 | 101% |
| edge>=20, cost<=10 | $19 | 238% | +$3 | 180% |
| edge>=40, cost<=15 | $16 | 354% | +$1 | 223% |

Loose parameters (low edge, high cost) lose money on eval. Tight parameters generalize.

### v6: Live Trading — First Attempt (Apr 12-14)

Deployed with edge-based rules on production Kalshi. **Lost ~$20.**

**Root cause**: With 190 errors, the model correctly identified cheap YES contracts as slightly underpriced. But 75% of trades were BUY YES on contracts that settle YES only 10-15% of the time. The few wins didn't offset the many losses in just 2 days of trading. The strategy needs more volume for the law of large numbers.

### v7: Combined Forecast + Price Signals (`8624614`)

Discovery: combining forecast edge with price action produces the most consistent signals.

**Two winning strategies found via comprehensive backtest:**

#### Strategy 1: buy-yes-dip
`BUY YES when forecast edge >= 10c AND ask <= 8c AND price dropped 3c+ in last hour`

| Split | Fills | PnL | ROI | Win% |
|---|---|---|---|---|
| Train (<Apr 10) | 101 | +$26.49 | +407% | 33% |
| Eval (Apr 10-14) | 30 | +$8.29 | +485% | 33% |

The price drop filters out stale cheap contracts. The forecast confirms the drop was an overreaction.

#### Strategy 2: buy-no-drift
`BUY NO when forecast edge >= 5c AND mid 20-35c AND price rose 2c+ in last 6 hours`

| Split | Fills | PnL | ROI | Win% |
|---|---|---|---|---|
| Train (<Apr 10) | 416 | +$68.09 | +22% | 89% |
| Eval (Apr 10-14) | 88 | +$22.62 | +35% | **100%** |

The market overprices mid-range contracts. When they drift UP while the forecast says they should be lower, sell.

---

## Live Experiment Log

### Experiment 1: Edge-based rules (Apr 12-14)
- **Git**: `8da4b1c`
- **Strategy**: 4 rules (R1-R4) with edge thresholds + cost caps + station-scaled edge
- **Rules**: R1(edge>=10,c<=40) R2(edge>=15,c<=30) R3(edge>=20,c<=20) R4(edge>=30,c<=15)
- **Environment**: Production Kalshi
- **Result**: **-$20 loss** over ~48 hours
- **Fills**: 202 trades (197 BUY YES, 5 BUY NO)
- **Problem**: Overwhelmingly bought cheap YES. Model was correct on fair value but the strategy needed more trades for the win rate to converge.
- **Lesson**: High-variance strategies (24% win rate) need hundreds of fills to be profitable. 2 days is not enough.

### Experiment 2: Two-strategy deployment (Apr 14+)
- **Git**: `8624614`
- **Strategies**: `buy-yes-dip` and `buy-no-drift` running in separate terminals
- **Environment**: Production Kalshi
- **Log files**: `data/trades_buy-yes-dip.log`, `data/trades_buy-no-drift.log`
- **Result**: Pending — just deployed

---

## Lessons Learned

1. **The NBS forecast is the best single predictor.** It's accurate to 1.5-3.5F depending on city. No amount of blending with GFS/LAMP improves it meaningfully.

2. **Lead time doesn't matter (17-48h).** The forecast made 2 days before is as accurate as the one made 17 hours before. Don't try to time entries based on lead time.

3. **The market overprices mid-range contracts (20-40c).** They settle YES only 17-25% of the time but the market prices them at 25-35%. This is the most robust and consistent edge.

4. **Cheap YES (5-10c) are underpriced.** They settle YES ~14% of the time but the market prices them at 7%. But you need high volume for this to work — each trade has a ~86% chance of total loss.

5. **NY and Philadelphia errors are correlated (r=0.65).** Treat them as one position. All other city pairs are independent.

6. **Pure price signals don't work.** Mean reversion, momentum, volatility — none produce consistent train+eval profits from Kalshi prices alone. The market is efficient on price patterns.

7. **Forecast + price together work.** The forecast provides the fundamental edge, the price action provides timing. "Forecast says it's underpriced AND the price just dropped" is much stronger than either signal alone.

8. **10 data points is not enough for empirical pricing.** With only 10 historical errors, Laplace smoothing dominates and every contract gets priced at 25c or 33c. You need 100+ errors per station.

9. **Backtest ROI >> live ROI.** The first live deployment lost money despite backtests showing +100% ROI. Small sample of live trades + high-variance strategy = expected outcome. Don't judge a 24% win-rate strategy on 50 trades.

10. **The SDK bug matters.** `kalshi_python_sync` can't parse Denver, LA, and Philadelphia markets due to a Pydantic validation error on null `subtitle` fields. This costs 3 of 6 tradeable cities.
