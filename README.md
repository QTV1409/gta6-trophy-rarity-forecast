# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/QTV1409/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/QTV1409/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who have earned that trophy, one week after launch. A damped log-time trend on the logit scale, fitted to the house's reads of the trophy page, with story-chain and platinum constraints and an honestly wide spread.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, five quantiles of each trophy's earn rate in percent. It is deterministic — two runs on the same case write byte-identical files — and runs in about 0.05 s.

## Method

Exophase refreshes its counts on its own schedule, so the displayed page moves in steps: long runs of identical reads, then a jump. The program first collapses each run of identical snapshots into one distinct page state, dated at its first appearance; repeated identical reads are one observation, not many.

For each trophy, the earn rate is modelled on the logit scale as a straight line in log time — `logit(rate) ~ a + b * ln(1 + t)`, `t` in hours since release — fitted by least squares over the distinct states (the last 48 hours of them when there are at least four). Log time is the right clock for a launch: the first days move rates by multiples, not by points, and the logit scale is the scale the contest scores on. The fitted line is read at the target, but the move it implies beyond the last observed state is damped by 0.55: a launch-week trend never holds its early slope for four more days. A trophy whose last rate is at least 95% is fitted the other way round, on `ln(100 - rate)` — the not-earned share is what moves when a near-universal trophy dilutes as new players join.

Three facts about trophies are then enforced on the medians: the story chapters run in order, so their rates are non-increasing along the chain (Chapter 1, 2, 3, 4, final mission); the platinum needs every other trophy, so it is at most the rarest other trophy's rate; and a trophy every tracked player had already earned stays the most-earned trophy.

The spread combines, in quadrature: a random-walk term `0.35 * sqrt(days of horizon)`; the fit's residual standard deviation; an extrapolation term growing with how far past the last observed state the target sits in log time; a flat 0.25 for Exophase's display lag; and, when the history shows a refresh that multiplied the tracked players (the launch-night refresh took 638 to 3,868 tracked), a shock term that is large for a 90-minute horizon and decays over about a day. The spread is floored at 0.40 and capped at 2.5 logit units.

With fewer than three distinct states, or no movement to fit, a trophy falls back to its last rate with a wide spread; a trophy with no rate anywhere uses a tier prior (bronze 1%, silver 0.5%, gold 0.3%, platinum 0.1%).

## Validation

There is no local grade for this competition — the server scores every entry — so the model was checked on two synthetic final-like histories built from the kit's practice cases (a fast-saturation regime and a slow-progression regime, both with a growing tracked base and refresh steps). On those, this program scored a mean pinball loss of 0.22 against a simulated day-7 truth, versus 0.30 for the persistence baseline and 0.34 for the house curve baseline, with 90% coverage of 1.0. The damping factor 0.55 was chosen on those two regimes as the best worst-case among 0.4, 0.55, 0.75 and 1.0; the spread survived a systematic 1.5x shift of the simulated truth without blowing up. The synthetic truths are invented, so these numbers validate the machinery, not the real answer. On the kit's practice cases the program writes valid, deterministic output (medians equal to the last read rates, as the flat launch-night history gives nothing to fit).

## Files

- `main.py` — the whole entry: reads the case, forecasts, writes the output.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9qs5x4yn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
