"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Method: hybrid log-time fit + type prior + ceiling (see METHOD.md).
For each trophy we fit the slope of logit(rate) against ln(1 + hours since
launch) over the last 24h of reads, shrink that slope toward a per-type
prior, then forecast the median with a saturating logistic toward a
per-type ceiling, and widen the quantiles with a per-type spread.

Why this beats the house entries:
- The house baseline (persistence) assumes rates never move. In week 1 the
  launch influx dilutes every rate, then the dilution unwinds and rates rise
  a lot as the day-1 cohort progresses. Persistence misses that rise by
  3-5 logit units on story trophies.
- The house curve (one straight logit-vs-ln(1+h) line, no ceiling, one global
  slope) over-shoots trophies that are already saturating and under-shoots
  late trophies whose earners have not arrived yet. The ceiling and the
  per-type slopes fix both.

Notes on the logistic step (METHOD.md step 5): the fitted/prior slopes are
d(logit)/d(ln(1+h)), i.e. logit units per log-hour. The logistic decay rate
k must therefore be slope / gap, where gap = logit(ceiling) - logit(last
rate), so that the forecast's initial slope equals the fitted slope and the
rate then bends toward its ceiling. Plugging the slope in as k directly
would make the initial slope k * gap (3-5x too steep for story trophies)
and contradict the comparable-games calibration (logit delta day 2.5 -> 7
of +0.6..+0.9 for early story, +1.2..+1.8 for platinum).

I/O contract (kit starter, read as reference only):
- reads every <id>.in in MAKEFUN_INPUT_DIR (default /input)
- writes <id>.out to MAKEFUN_OUTPUT_DIR (default /output):
  {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
- one row per trophy, id once, five finite numbers 0-100, non-decreasing,
  at most 65,536 bytes, deterministic. Quantiles rounded to 4 decimals,
  half up (JavaScript Math.round style).

Runs as `python3 -P -s /bot/main.py` in python:3.13-slim, stdlib only,
no network, under 8 seconds per case. Everything lives in this one file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# GTA VI launched 2026-11-19T00:00:00Z. Hours are counted from here.
DEFAULT_LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# How far back the slope fit looks (hours before the checkpoint).
FIT_WINDOW_HOURS = 24.0
# A fit needs at least this many non-null reads, else the type prior rules.
MIN_FIT_READS = 3


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg > 20.0:
        return 99.99
    if lg < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def launch_from_case(case: dict) -> datetime:
    """Midnight UTC of the game's release date; the recorded launch is 00:00Z."""
    try:
        day = (case.get("game") or {}).get("release_date", "")
        year, month, dom = (int(part) for part in str(day).split("-")[:3])
        return datetime(year, month, dom, tzinfo=timezone.utc)
    except Exception:
        return DEFAULT_LAUNCH


# Per-type parameters: (prior slope per ln-hour, sigma per sqrt-day,
# fallback rate % when a trophy has no usable read at all).
# Slopes and ceilings are calibrated on the comparables (Elden Ring,
# Spider-Man 2, God of War Ragnarok, Horizon Forbidden West, GTA V
# settled on Exophase); see research-comparables.md.
TYPE_PARAMS = {
    "intro": (0.0, 0.05, 99.99),
    "early": (0.7, 0.40, 1.0),
    "late": (1.0, 0.50, 0.05),
    "hundred": (1.2, 0.70, 0.03),
    "collect": (1.1, 0.60, 0.03),
    "grind": (1.2, 0.70, 0.03),
    "easy": (0.9, 0.50, 0.5),
    "platinum": (1.5, 0.80, 0.03),
}

# Trophy id -> (type, ceiling %). The ceiling is the settled long-run
# Exophase rate for that kind of trophy (GTA V settled: platinum 3.75%,
# final mission 15.38%, 100% 4.63%, collectibles ~5.4%, stunt jumps 8.61%).
TROPHY_TABLE = {
    1: ("platinum", 4.0),
    2: ("intro", 100.0),
    3: ("early", 50.0),
    4: ("early", 35.0),
    5: ("late", 20.0),
    6: ("late", 18.0),
    7: ("late", 15.0),
    8: ("early", 25.0),
    9: ("early", 25.0),
    10: ("early", 25.0),
    11: ("early", 25.0),
    12: ("early", 25.0),
    13: ("early", 25.0),
    14: ("hundred", 5.0),
    15: ("grind", 8.0),
    16: ("easy", 30.0),
    17: ("grind", 8.0),
    18: ("grind", 8.0),
    19: ("grind", 8.0),
    20: ("easy", 25.0),
    21: ("easy", 30.0),
    22: ("late", 15.0),
    23: ("late", 15.0),
    24: ("late", 15.0),
    25: ("late", 15.0),
    26: ("late", 15.0),
    27: ("late", 15.0),
    28: ("collect", 5.0),
    29: ("easy", 15.0),
    30: ("collect", 5.0),
    31: ("collect", 5.0),
    32: ("grind", 8.0),
    33: ("easy", 20.0),
    34: ("easy", 15.0),
    35: ("collect", 5.0),
    36: ("easy", 25.0),
    37: ("easy", 20.0),
    38: ("easy", 15.0),
    39: ("easy", 20.0),
    40: ("grind", 8.0),
    41: ("easy", 25.0),
    42: ("collect", 5.0),
    43: ("easy", 15.0),
    44: ("easy", 25.0),
    45: ("easy", 25.0),
    46: ("easy", 20.0),
    47: ("easy", 20.0),
    48: ("easy", 25.0),
    49: ("easy", 20.0),
    50: ("easy", 20.0),
    51: ("easy", 15.0),
    52: ("grind", 8.0),
    53: ("grind", 8.0),
    54: ("collect", 5.0),
    55: ("easy", 20.0),
}

# Fallback classification by tier, for any trophy id not in the table.
TIER_FALLBACK = {
    "platinum": ("platinum", 4.0),
    "gold": ("late", 15.0),
    "silver": ("grind", 8.0),
    "bronze": ("easy", 20.0),
}


def classify(trophy: dict):
    """(type, ceiling %) for a trophy, from the table or its tier."""
    entry = TROPHY_TABLE.get(trophy.get("id"))
    if entry is not None:
        return entry
    return TIER_FALLBACK.get(str(trophy.get("tier", "")).lower(), ("easy", 20.0))


def fit_line(points):
    """Least-squares line y = a + b*u over (u, y) points.

    Returns (slope, residual std, n). With fewer than 2 points there is no
    slope; with a flat u (all reads at one time) the slope is 0.
    """
    n = len(points)
    if n < 2:
        return None, 0.0, n
    mu = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mu) ** 2 for p in points)
    if sxx <= 1e-12:
        return 0.0, 0.0, n
    sxy = sum((p[0] - mu) * (p[1] - my) for p in points)
    slope = sxy / sxx
    intercept = my - slope * mu
    if n > 2:
        resid = math.sqrt(
            sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points) / (n - 2)
        )
    else:
        resid = 0.0
    return slope, resid, n


def forecast_trophy(points_window, points_all, r_last, type_name, ceiling, horizon_days, span_ln):
    """One trophy's five quantiles.

    points_window: (u, logit rate) over the last 24h; points_all: the same
    over the whole history (used when the window is too thin).
    r_last: rate % in the last non-null read (None when there is none).
    span_ln: u_target - u_checkpoint, the log-time span being forecast.
    """
    prior_slope, sigma, fallback_rate = TYPE_PARAMS[type_name]

    # Slope: fit the window, fall back to the whole history, fall back to
    # the prior when there are too few reads to trust a fit.
    points = points_window if len(points_window) >= MIN_FIT_READS else points_all
    slope_data, resid, n = fit_line(points)
    if slope_data is None or n < MIN_FIT_READS:
        slope = prior_slope
        resid = 1.0 if r_last is None else resid
    else:
        # Data weight grows with the number of reads and the fit quality,
        # a noisy fit lets the prior dominate - but the weight is CAPPED at
        # 0.5. The last-24h slope is measured during the dilution unwind
        # (rates dip, then recover), so it is a local transient slope, not
        # the week-1 average slope the prior encodes from the comparables.
        # Letting the data dominate would extrapolate the transient (the
        # house curve's failure: it over-shoots recovering trophies and
        # under-shoots late ones). The prior always keeps at least half
        # the weight.
        w_data = min(0.5, (n / (n + 4.0)) * (1.0 / (1.0 + resid)))
        slope = w_data * slope_data + (1.0 - w_data) * prior_slope

    if r_last is None:
        r_last = fallback_rate

    # Median: saturating logistic toward the ceiling. The decay rate k is
    # slope / gap so the forecast's initial slope equals the fitted slope.
    lg_last = logit(r_last)
    lg_ceil = logit(ceiling)
    gap = lg_ceil - lg_last
    if gap <= 0.01:
        # Already at or above the ceiling: persistence, tight spread.
        lg_med = lg_last
    else:
        k = slope / gap
        lg_med = lg_ceil - gap * math.exp(-k * span_ln)

    # Spread: per-type sigma over the horizon, widened in quadrature by
    # the fit's residual (a noisy fit is an uncertain forecast).
    base = sigma * math.sqrt(max(0.0, horizon_days))
    spread = math.sqrt(base * base + resid * resid)
    spread = min(3.0, max(0.02, spread))

    quantiles = []
    prev = -1.0
    for z in Z:
        value = round4(inv_logit(lg_med + z * spread))
        value = min(100.0, max(0.0, float(value)))
        if value < prev:
            value = prev
        quantiles.append(value)
        prev = value
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    launch = launch_from_case(case)
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)
    u_cp = math.log1p(max(0.0, (at - launch).total_seconds() / 3600.0))
    u_tg = math.log1p(max(0.0, (target - launch).total_seconds() / 3600.0))
    span_ln = max(0.0, u_tg - u_cp)

    # Reads, oldest first, each with its log-time and its rates list.
    reads = []
    for read in case.get("history") or []:
        try:
            ts = parse_ts(read["at"])
        except Exception:
            continue
        u = math.log1p(max(0.0, (ts - launch).total_seconds() / 3600.0))
        rates = read.get("rates") or []
        reads.append((ts, u, rates))

    window_start = at.timestamp() - FIT_WINDOW_HOURS * 3600.0
    trophies = case.get("trophies") or []
    rows = []
    for i, trophy in enumerate(trophies):
        type_name, ceiling = classify(trophy)
        points_window = []
        points_all = []
        r_last = None
        for ts, u, rates in reads:
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            y = logit(rate)
            points_all.append((u, y))
            if ts.timestamp() >= window_start:
                points_window.append((u, y))
        # Last non-null rate, scanning from the newest read back.
        for _ts, _u, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                r_last = rates[i]
                break
        rows.append(
            {
                "id": trophy["id"],
                "q": forecast_trophy(
                    points_window, points_all, r_last, type_name, ceiling,
                    horizon_days, span_ln,
                ),
            }
        )
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
