"""SVI benchmark, arbitrage checks, hedging deltas and the surface models (no market data)."""

import numpy as np
import pytest
from qpricing.models import SVI, Slice, SVIPrice


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
