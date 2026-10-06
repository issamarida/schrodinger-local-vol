"""SVI benchmark, arbitrage checks, hedging deltas and the surface models (no market data)."""

import numpy as np
import pandas as pd
import pytest
from qpricing.arbitrage import butterfly_stats, market_butterflies
from qpricing.models import SVI, BlackScholes, Schrodinger, Slice, SVIPrice, black76


def _slice(days=10, F=3600.0, lo=3150.0, hi=3950.0):
    T = days / 365
    strikes = np.arange(lo, hi, 25.0)
    return Slice(F, F - 5, T, days, 0.03, np.exp(-0.03 * T), strikes, strikes >= F,
                 np.zeros_like(strikes))


@pytest.mark.parametrize("model", [SVI(), SVIPrice()])
def test_svi_recovers_its_own_smile(model):
    s = _slice()
    truth = np.array([0.0006, 0.02, -0.7, 0.01, 0.03]) * np.array([36.5, 36.5, 1, 1, 1])
    s.mid = model.price(truth, s)
    fitted = model.calibrate(s)
    assert np.max(np.abs(model.price(fitted, s) - s.mid)) < 0.02


def test_butterfly_check_flags_an_arbitrageable_svi_and_passes_clean_models():
    # Axel Vogt's raw SVI (Gatheral & Jacquier 2014): positive variance, negative density.
    a, b, rho, m, sig = -0.0410, 0.1331, 0.3060, 0.3586, 0.4153
    vogt = np.array([a + b * sig * np.sqrt(1 - rho ** 2), b, rho, m, sig])  # T = 1: w = v
    s = Slice(100.0, 100.0, 1.0, 365, 0.0, 1.0, np.linspace(30.0, 300.0, 40),
              np.linspace(30.0, 300.0, 40) >= 100.0, np.zeros(40))
    assert butterfly_stats(SVI(), vogt, s, wings=False)["violation"]
    assert not butterfly_stats(BlackScholes(), np.array([0.3]), s, wings=True)["violation"]
    well = np.array([0.02, 1.0, -0.7, 0.0, 0.05])
    st = butterfly_stats(Schrodinger(), well, _slice(), wings=True)
    assert not st["violation"] and st["neg_mass"] == 0.0


def test_market_butterfly_mixes_puts_and_calls_through_parity():
    s = _slice()
    s.mid = black76(s.is_call, s.F, s.strikes, s.T, 0.25, s.df)
    g = pd.DataFrame({"strike": s.strikes, "is_call": s.is_call, "bid": s.mid - 0.05,
                      "ask": s.mid + 0.05, "mid": s.mid, "F": s.F, "df": s.df})
    assert market_butterflies(g) == {"violation": False, "executable": False}
    g.loc[10, ["bid", "ask", "mid"]] += 5.0   # a body far too rich: sell it, buy the wings
    assert market_butterflies(g)["executable"]
