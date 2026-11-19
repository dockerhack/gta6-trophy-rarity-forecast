# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dockerhack/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dockerhack/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — five quantiles (0.05, 0.25, 0.5,
0.75, 0.95) per trophy, scored by mean pinball loss on the logit scale.

Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

## Method

A hybrid log-time fit + type prior + ceiling, calibrated on comparable
big PS5 open-world launches (Elden Ring, Spider-Man 2, God of War Ragnarok,
Horizon Forbidden West, GTA V settled on Exophase):

1. **Fit the slope from the data.** For each trophy, least-squares fit of
   logit(rate) against ln(1 + hours since launch) over the last 24 hours
   of reads (whole history when the window is too thin).
2. **Shrink toward a type prior.** The data slope is averaged with a
   per-type prior slope (early story 0.7, late story 1.0, collectibles 1.1,
   grinds 1.2, platinum 1.5 per log-hour), weighted by read count and fit
   quality, so a noisy or dilution-dominated window cannot drive a wild
   extrapolation.
3. **Forecast the median with a saturating logistic toward a ceiling.**
   logit(target) = logit(ceiling) - (logit(ceiling) - logit(last rate)) *
   exp(-k * L), where k = slope / gap and L is the log-time span to the
   target. The rate rises toward its long-run ceiling and bends as it
   approaches it — no over-shoot on saturating trophies, no under-shoot
   on late trophies.
4. **Spread the quantiles by type.** s = sigma_type * sqrt(horizon days),
   widened in quadrature by the fit residual. Platinum and grinds get the
   widest bands; the intro trophy (Gone Fishing, ~100%) gets the narrowest.

The key fact the method is built on: in week 1 the launch influx dilutes
every rate (median 0.053x from launch night to the 19 Nov 17:06 UTC
snapshot), then the dilution unwinds and rates rise a lot as the day-1
cohort progresses — so a persistence forecast (the house baseline) misses
by 3-5 logit units on story trophies, and a single straight line (the house
curve) has no ceiling and one global slope.

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic,
under 8 seconds per case. Reads every `<id>.in` in the input directory
(`MAKEFUN_INPUT_DIR`, default `/input`) and writes `<id>.out` to the output
directory (`MAKEFUN_OUTPUT_DIR`, default `/output`):

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

Run locally:

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
