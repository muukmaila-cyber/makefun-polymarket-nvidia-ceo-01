"""Polymarket next-NVIDIA-CEO forecaster (Python 3.13, stdlib only).

Method (validated on the 6 practice cases, pinball loss 0.2813 vs house
baseline 0.363589, top dev entry 0.190, v1 0.2848):
  - Forecast in logit space: the Yes price is a probability, so a symmetric
    random walk in logit space becomes the right skewed spread in price
    space (a small market can double far more easily than it can halve).
  - Two-component normal mixture in logit space, per sqrt-minute:
    85% narrow (sigma 0.014) + 15% wide (sigma 0.08). The mixture's
    quantiles are Z_MIX below (precomputed by bisection on the mixture CDF,
    verified CDF(Z_MIX) == tau). This gives a narrow centre with heavy
    tails, matching the data: median |240-min move| is only 0.003, but
    5.5% of market-cases jump >0.05 and the max is 0.223. A single normal
    is too wide in the centre and too narrow in the tails; a Student-t and
    per-market / global realised vol were tested and are worse (realised
    vol overestimates future vol).
  - Mild mean reversion: the centre drifts 5% per hour toward the mean
    logit of the last 24 hours (1440 min). A 24h anchor beats 6h (0.2848)
    and 48h/72h (0.288/0.292); the price reverts toward its longer-term
    level, not just the last few hours.
  - q_tau(h) = sigmoid(logit(x) + drift*h/60 + Z_MIX[tau] * sqrt(h)),
    clamped to [0, 1], rounded to 6 decimals, non-decreasing per row.
  - Deterministic: no randomness, no network, no time dependence.
"""
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Quantiles of the 85% N(0, 0.014^2) + 15% N(0, 0.08^2) mixture, per
# sqrt-minute, in logit units. Precomputed by bisection on the mixture CDF.
Z_MIX = [-0.038344, -0.011022, 0.0, 0.011022, 0.038344]
MR_WINDOW = 1440        # minutes of history the mean reverts toward (24h)
MR_RATE = 0.05          # per hour toward the window mean logit


def logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0:
        return 1.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return e / (1.0 + e)


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon):
    x = prices[-1]
    lx = logit(x)
    # mean logit of the last MR_WINDOW minutes (or all history if shorter)
    hist = prices[-MR_WINDOW:]
    mean_logit = sum(logit(v) for v in hist) / len(hist)
    drift = (mean_logit - lx) * MR_RATE  # logit drift per hour
    rows = []
    for h in range(1, horizon + 1):
        center = lx + drift * h / 60.0
        sh = math.sqrt(h)
        row = [price6(sigmoid(center + z * sh)) for z in Z_MIX]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["quantiles"] != QUANTILES:
            raise ValueError(f"{path.name} asks for {case['quantiles']}")
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], case["horizon_minutes"])}
            for m in case["markets"]
        ]
        (out_dir / f"{path.stem}.out").write_text(
            json.dumps({"forecasts": forecasts}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
