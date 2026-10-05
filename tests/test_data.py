"""Data-layer tests on small hand-built frames (the licensed raw files are not needed)."""

import numpy as np
import pandas as pd
import pytest
from qpricing.data import QuoteFilter, apply_filter, implied_forwards, interpolate_rate


def test_rate_interpolation_is_linear_and_flat_outside():
    curve = np.array([0.01, 0.02, 0.03, np.nan, 0.04, 0.04, 0.05])
    assert interpolate_rate(curve, np.array([0.01]))[0] == pytest.approx(0.01)   # flat below 1M
    mid = (1 / 12 + 0.25) / 2
    assert interpolate_rate(curve, np.array([mid]))[0] == pytest.approx(0.015)
    assert interpolate_rate(curve, np.array([3.5]))[0] == pytest.approx(0.03625)  # 1Y-5Y, skips 2Y
    assert interpolate_rate(curve, np.array([40.0]))[0] == pytest.approx(0.05)


def _chain(F=3605.0, S=3600.0, r=0.03, T=10 / 365):
    rows = []
    for K in np.arange(3500.0, 3725.0, 25.0):
        disc = np.exp(-r * T)
        c = disc * max(F - K, 0) + 20.0   # parity-consistent: C - P = df (F - K)
        p = c - disc * (F - K)
        for is_call, mid in ((True, c), (False, p)):
            rows.append(dict(root="SPXW", expiration=pd.Timestamp("2022-10-21"), strike=K,
                             is_call=is_call, bid=mid - 0.1, ask=mid + 0.1, mid=mid, T=T, r=r, S=S))
    return pd.DataFrame(rows)


def test_parity_forward_is_recovered():
    F = implied_forwards(_chain())
    assert F.iloc[0] == pytest.approx(3605.0, rel=1e-12)


def test_filter_keeps_only_liquid_otm_quotes():
    df = _chain()
    df["F"], df["days"] = 3605.0, 10.0
    out = apply_filter(df, QuoteFilter())
    assert ((out.is_call & (out.strike >= out.F)) | (~out.is_call & (out.strike < out.F))).all()
    assert len(out) == (df.strike >= 3605).sum() / 2 + (df.strike < 3605).sum() / 2
