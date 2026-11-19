"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at the case's target instant.

    python3 main.py            (reads $MAKEFUN_INPUT_DIR or /input,
                                writes $MAKEFUN_OUTPUT_DIR or /output)

Python 3.13 standard library only, no network, deterministic: the same
case file always produces the same bytes.

Model
-----
The case gives the house's reads of Exophase's trophy page: every read is
one snapshot of every trophy's earn rate and the players tracked. Exophase
refreshes its counts on its own schedule, so the displayed page moves in
steps: long runs of identical reads, then a jump. The first step is to
collapse each run of identical snapshots into one distinct page state,
dated at its first appearance; repeated identical reads are one
observation, not many.

For each trophy, the earn rate is then modelled on the logit scale as a
straight line in log time:

    logit(rate)  ~  a + b * ln(1 + t),   t = hours since release

fitted by least squares over the distinct states (the last 48 hours of
them when there are at least four, otherwise all of them). Log time is
the right clock for a launch: the first days move rates by multiples,
not by points, and the logit scale is the scale the contest scores on,
so a miss from 1% to 2% counts like a miss from 33% to 50%.

The fitted line is read at the target, but the move it implies beyond
the last observed state is damped (multiplied by 0.55): a launch-week
trend never holds its early slope for four more days, and an undamped
extrapolation is how forecasts overshoot. Two synthetic final-like
histories (fast saturation and slow progression) put 0.55 ahead of
0.4, 0.75 and 1.0 on the worst case, and ahead of persistence in both. A trophy whose last rate is
at least 95% is fitted the other way round, on ln(100 - rate): the
not-earned share is what moves when a near-universal trophy dilutes.

Three facts about trophies are then enforced on the medians, by pooling
only where the fit violates them:

  * the story chapters run in order (Chapter 1, 2, 3, 4, final mission),
    so their rates are non-increasing along that chain;
  * the platinum needs every other trophy, so it is at most the rarest
    other trophy's rate;
  * a trophy every tracked player had already earned (rate >= 99.5% in
    the last read) stays the most-earned trophy.

Spread
------
Each trophy's five quantiles are its median moved on the logit scale by
z_tau * s, with z = (-1.645, -0.674, 0, 0.674, 1.645). The spread s
combines, in quadrature: a random-walk term 0.35 * sqrt(days of horizon);
the fit's residual standard deviation; an extrapolation term that grows
with how far past the last observed state the target sits in log time;
a flat 0.25 for Exophase's display lag (the answer is the page as of
its last refresh before the target, not the live truth); and, when the
history shows a refresh that multiplied the tracked players (the
launch-night refresh took 638 to 3,868 tracked), a shock term that is
large for a 90-minute horizon and decays over about a day. The spread
is floored at 0.40 and capped at 2.5 logit units: pinball loss punishes
a band that is too narrow far harder than one that is honestly wide.

Fallbacks
---------
With fewer than three distinct states, or no movement to fit, a trophy
falls back to its last rate with a wide spread. A trophy with no rate
in the last read uses its last non-null rate; one with no rate anywhere
uses a tier prior (bronze 1%, silver 0.5%, gold 0.3%, platinum 0.1%).

Output
------
One JSON object per case, exactly {"trophies": [{"id": n, "q": [q05,
q25, q50, q75, q95]}, ...]}, one row per trophy, quantiles in percent,
non-decreasing, rounded to 4 decimals half-up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (fixed in advance, not tuned on any answer) ---
SIGMA_RW = 0.35        # random-walk spread, logit units per sqrt day
DAMP = 0.55           # shrink of the fitted move past the last state
DISPLAY_LAG = 0.25    # logit spread for Exophase's refresh lag
SPREAD_FLOOR = 0.40
SPREAD_CAP = 2.5
WINDOW_HOURS = 48.0   # prefer the last 48 h of states when enough exist
MIN_STATES_WINDOW = 4
MIN_STATES_FIT = 3
SATURATED_AT = 95.0   # rate >= this is fitted on the not-earned share
FIRST_TROPHY_AT = 99.5  # a trophy at/above this stays the most-earned
TIER_PRIOR = {"bronze": 1.0, "silver": 0.5, "gold": 0.3, "platinum": 0.1}
# Story chain, by trophy id in the case's ordering: Chapter 1, 2, 3, 4,
# final mission. Rates along it are non-increasing.
STORY_CHAIN = [3, 4, 5, 6, 7]
PLATINUM_ID = 1


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_values(levels):
    if levels == LEVELS:
        return Z
    return [norm_ppf(p) for p in levels]


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_states(history):
    """Collapse runs of identical snapshots into one state each, dated
    at the run's first slot. Returns [(slot_datetime, players, rates)]."""
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        states.append((parse_time(read["slot"]), read.get("players_tracked"), rates))
    return states


def forecast_case(case, name):
    trophies = case["trophies"]
    levels = case.get("levels", LEVELS)
    zs = z_values(levels)
    horizon_days = case["horizon_minutes"] / 1440.0

    release = case.get("game", {}).get("release_date")
    if release:
        release_dt = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    else:
        release_dt = None

    history = case.get("history") or []
    states = distinct_states(history)
    if release_dt is None and states:
        release_dt = states[0][0]
    if release_dt is None:
        release_dt = parse_time(case["at"])

    checkpoint = parse_time(case["at"])
    x_target = math.log1p(max(0.0, (checkpoint - release_dt).total_seconds() / 3600.0
                              + case["horizon_minutes"] / 60.0))

    # Did the history ever show a refresh that multiplied the tracked
    # players? That is the launch-night shock: rates fell because the
    # denominator grew faster than the earners.
    big_refresh = False
    for (_, n0, _), (_, n1, _) in zip(states, states[1:]):
        if n0 and n1 and n1 >= 2.0 * n0:
            big_refresh = True
            break
    shock = 0.9 * math.exp(-horizon_days) if big_refresh else 0.0

    # Fit window: the last WINDOW_HOURS of distinct states when there
    # are enough of them, else every distinct state.
    if states:
        cutoff = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
        recent = [s for s in states if s[0].timestamp() >= cutoff]
        window = recent if len(recent) >= MIN_STATES_WINDOW else states
    else:
        window = []

    medians = {}
    spreads = {}
    for idx, trophy in enumerate(trophies):
        tid = trophy["id"]
        points = [(math.log1p(max(0.0, (slot - release_dt).total_seconds() / 3600.0)),
                   rates[idx])
                  for (slot, _n, rates) in window
                  if idx < len(rates) and rates[idx] is not None]
        last_rate = None
        for (_slot, _n, rates) in reversed(states):
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        if last_rate is None:
            last_rate = TIER_PRIOR.get(trophy.get("tier"), 1.0)

        saturated = last_rate >= SATURATED_AT
        xs = [x for x, _r in points]
        if saturated:
            ys = [math.log(max(0.01, 100.0 - r)) for _x, r in points]
        else:
            ys = [logit(r) for _x, r in points]
        distinct_x = len(set(round(x, 9) for x in xs))

        if len(points) >= MIN_STATES_FIT and distinct_x >= MIN_STATES_FIT:
            a, b, resid = ols(xs, ys)
            x_last = xs[-1]
            y_last_fit = a + b * x_last
            y_target = y_last_fit + DAMP * b * (x_target - x_last)
            dx = max(0.0, x_target - x_last)
            if saturated:
                u_med = math.exp(max(math.log(0.01), min(math.log(99.99), y_target)))
                median = clamp(100.0 - u_med)
            else:
                median = clamp(inv_logit(y_target))
            spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                               + resid ** 2
                               + (0.35 * dx) ** 2
                               + DISPLAY_LAG ** 2
                               + shock ** 2)
        else:
            median = clamp(last_rate)
            spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                               + 0.5 ** 2
                               + DISPLAY_LAG ** 2
                               + shock ** 2)
        medians[tid] = median
        spreads[tid] = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))

    # --- structural constraints on the medians (pool only violations) ---
    by_id = {t["id"]: t for t in trophies}

    # Story chain non-increasing.
    chain = [i for i in STORY_CHAIN if i in medians]
    for prev, cur in zip(chain, chain[1:]):
        if medians[cur] > medians[prev]:
            medians[cur] = medians[prev]

    # Platinum needs every other trophy.
    if PLATINUM_ID in medians:
        others = [v for k, v in medians.items() if k != PLATINUM_ID]
        if others:
            medians[PLATINUM_ID] = min(medians[PLATINUM_ID], min(others))

    # A trophy everyone had already earned stays the most-earned.
    for pos, trophy in enumerate(trophies):
        tid = trophy["id"]
        last = None
        for (_slot, _n, rates) in reversed(states):
            if pos < len(rates) and rates[pos] is not None:
                last = rates[pos]
                break
        if last is not None and last >= FIRST_TROPHY_AT:
            medians[tid] = max(medians[tid], max(medians.values()))

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        median = clamp(medians[tid])
        s = spreads[tid]
        if median >= SATURATED_AT:
            # Quantiles of the rate via quantiles of the not-earned share.
            u_med = max(0.01, 100.0 - median)
            qs = [clamp(100.0 - u_med * math.exp(-z * s)) for z in zs]
        else:
            lg = logit(median)
            qs = [clamp(inv_logit(lg + z * s)) for z in zs]
        qs = [round4(q) for q in qs]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit(f"no case files in {INPUT_DIR}")
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise SystemExit(f"{path.name}: forecast over the 65,536-byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
