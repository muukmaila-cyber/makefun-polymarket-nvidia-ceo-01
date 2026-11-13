"""Polymarket next-NVIDIA-CEO forecaster (Python 3.13, stdlib only).

Method (validated on the 6 practice cases, pinball loss 0.2848 vs house
baseline 0.363589, top dev entry 0.190):
  - Forecast in logit space: the Yes price is a probability, so a symmetric
    random walk in logit space becomes the right skewed spread in price
    space (a small market can double far more easily than it can halve).
  - Fixed logit volatility sigma = 0.02 per sqrt-minute for every market.
    Tested and rejected on the practice cases: per-market realised vol
    (6h window, 0.380; per-market logit vol, 0.293-0.318), a wider or
    narrower fixed sigma (0.0015: 0.333; 0.0035: 0.407), asymmetric
    quantile shapes in price space (0.376-0.414), and a horizon exponent
    other than sqrt (alpha 0.4: 0.300; 0.6: 0.303). Recent realised vol is
    too noisy and the tiny markets' logit vol blows up; a single fixed
    sigma is the most robust choice.
  - Mild mean reversion: the centre drifts 5% per hour toward the mean
    logit of the last 6 hours (news jumps partially revert). Tested:
    no reversion 0.2893, 5%/h 0.2848, 10%/h 0.2864, 20%/h 0.3053.
  - q_tau(h) = sigmoid(logit(x) + drift*h/60 + z_tau * 0.02 * sqrt(h)),
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
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
SIGMA = 0.02            # logit volatility per sqrt-minute (fixed)
MR_WINDOW = 360         # minutes of history the mean reverts toward
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
        s = SIGMA * math.sqrt(h)
        row = [price6(sigmoid(center + z * s)) for z in Z]
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
