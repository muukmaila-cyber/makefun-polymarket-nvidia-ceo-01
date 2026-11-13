# Polymarket next NVIDIA CEO forecaster — make.fun entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead: five quantiles
(5%, 25%, 50%, 75%, 95%) per market per minute.

Method (validated on the kit's 6 practice cases): forecast in logit
space with a fixed logit volatility of 0.02 per sqrt-minute, plus a
mild mean reversion (5% per hour toward the mean logit of the last
6 hours). Practice pinball loss 0.2848 vs the house baseline's
0.363589. Python 3.13, stdlib only, no network, deterministic.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see LICENSE.
