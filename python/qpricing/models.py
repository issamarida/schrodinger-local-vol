"""Pricing models compared in the backtest, behind one calibrate/price interface.

* :class:`BlackScholes`  - one flat volatility per expiry slice (the best single-vol fit).
* :class:`AdHocBlackScholes` - Black-Scholes with a quadratic smile in log-moneyness, the
                            "practitioner" benchmark of Dumas, Fleming & Whaley (1998).
* :class:`Heston`        - stochastic volatility, calibrated and priced with QuantLib.
* :class:`Schrodinger`   - the imaginary-time Schrödinger solver (C++ core).

Every model is calibrated on the same quotes with the same objective (least squares on option
prices in index points), so differences in error come from the model, not from the fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import QuantLib as ql
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import ndtr

from qpricing import _qpcore


@dataclass
class Slice:
    """One expiry of one quote date: everything a model needs to price it."""

    F: float
    S: float
    T: float
    days: int
    r: float
    df: float
    strikes: np.ndarray
    is_call: np.ndarray
    mid: np.ndarray

    @classmethod
    def from_frame(cls, g) -> Slice:
        first = g.iloc[0]
        return cls(
            F=float(first.F), S=float(first.S), T=float(first["T"]), days=int(first.days),
            r=float(first.r), df=float(first["df"]), strikes=g.strike.to_numpy(float),
            is_call=g.is_call.to_numpy(bool), mid=g.mid.to_numpy(float),
        )

    @property
    def k(self) -> np.ndarray:
        return np.log(self.strikes / self.F)

    def atm_vol(self) -> float:
        """Implied vol of the quote closest to the forward (initial guesses only)."""
        i = int(np.argmin(np.abs(self.k)))
        iv = _qpcore.black76_implied_vol(bool(self.is_call[i]), self.F, float(self.strikes[i]),
                                         self.T, float(self.mid[i]), self.df)
        return float(iv) if np.isfinite(iv) else 0.2


def reanchor(model, params: np.ndarray, s: Slice, anchor: np.ndarray) -> np.ndarray:
    """Refits the model's single volatility-level parameter to the `anchor` quotes of `s`.

    Every model exposes ``with_level(params, u)``, a one-parameter family through ``params``
    (u = 1 returns ``params``) that moves the overall volatility level while freezing the shape
    of the smile. Used for the "re-anchored" next-day test.
    """
    sub = Slice(s.F, s.S, s.T, s.days, s.r, s.df, s.strikes[anchor], s.is_call[anchor],
                s.mid[anchor])

    def sse(u: float) -> float:
        return float(np.sum((model.price(model.with_level(params, u), sub) - sub.mid) ** 2))

    res = minimize_scalar(sse, bounds=(0.25, 4.0), method="bounded", options={"xatol": 1e-6})
    return model.with_level(params, res.x)


def black76(is_call: np.ndarray, F: float, K: np.ndarray, T: float, sigma: float,
            df: float) -> np.ndarray:
    """Vectorised Black-76 (validated against QuantLib in tests/test_models.py)."""
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return df * np.where(is_call, call, put)


class BlackScholes:
    name = "black_scholes"
    n_params = 1

    def calibrate(self, s: Slice, warm: np.ndarray | None = None) -> np.ndarray:
        def sse(sigma: float) -> float:
            model = black76(s.is_call, s.F, s.strikes, s.T, sigma, s.df)
            return float(np.sum((model - s.mid) ** 2))

        res = minimize_scalar(sse, bounds=(0.01, 3.0), method="bounded",
                              options={"xatol": 1e-7})
        return np.array([res.x])

    def price(self, params: np.ndarray, s: Slice) -> np.ndarray:
        return black76(s.is_call, s.F, s.strikes, s.T, float(params[0]), s.df)

    @staticmethod
    def with_level(params: np.ndarray, u: float) -> np.ndarray:
        return params * u


class AdHocBlackScholes:
    """Practitioner Black-Scholes: sigma(k) = max(b0 + b1 k + b2 k^2, 1%), k = ln(K/F)."""

    name = "adhoc_bs"
    n_params = 3

    @staticmethod
    def _vols(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        return np.maximum(params[0] + params[1] * k + params[2] * k * k, 0.01)

    def calibrate(self, s: Slice, warm: np.ndarray | None = None) -> np.ndarray:
        x0 = warm if warm is not None else np.array([s.atm_vol(), -1.0, 5.0])
        k = s.k
        res = least_squares(
            lambda p: black76(s.is_call, s.F, s.strikes, s.T, self._vols(p, k), s.df) - s.mid,
            x0, method="lm", max_nfev=2000)
        return res.x

    def price(self, params: np.ndarray, s: Slice) -> np.ndarray:
        return black76(s.is_call, s.F, s.strikes, s.T, self._vols(params, s.k), s.df)

    @staticmethod
    def with_level(params: np.ndarray, u: float) -> np.ndarray:
        return np.array([params[0] * u, params[1], params[2]])  # shift ATM vol, keep shape


class Schrodinger:
    """Effective local variance sigma^2(x) = a + b (rho (x - m) + sqrt((x - m)^2 + s^2)).

    Optimised in the coordinates (v_min, b, rho, m, s) with a = v_min - b s sqrt(1 - rho^2), so
    simple box bounds guarantee a strictly positive variance everywhere.
    """

    name = "schrodinger"
    n_params = 5
    lower = np.array([1e-4, 0.0, -0.999, -0.5, 1e-3])
    upper = np.array([4.0, 50.0, 0.999, 0.5, 1.0])

    def __init__(self, grid: _qpcore.GridSpec | None = None,
                 eval_grid: _qpcore.GridSpec | None = None):
        self.grid = grid or _qpcore.GridSpec(800, 120)
        self.eval_grid = eval_grid or _qpcore.GridSpec(1200, 200)

    @staticmethod
    def to_params(theta: np.ndarray) -> _qpcore.EffectiveVolParams:
        v_min, b, rho, m, s = (float(t) for t in theta)
        a = v_min - b * s * np.sqrt(1.0 - rho * rho)
        return _qpcore.EffectiveVolParams(a, b, rho, m, s)

    def _price(self, theta: np.ndarray, s: Slice, grid: _qpcore.GridSpec) -> np.ndarray:
        return _qpcore.schrodinger_price(self.to_params(theta), s.F, s.T, s.df,
                                         s.strikes, s.is_call, grid)

    def calibrate(self, s: Slice, warm: np.ndarray | None = None) -> np.ndarray:
        atm = s.atm_vol()
        starts = [
            np.array([0.8 * atm**2, 1.0, -0.7, 0.0, 0.05]),
            np.array([0.6 * atm**2, 4.0, -0.9, 0.02, 0.02]),
        ]
        if warm is not None:
            starts.append(np.clip(warm, self.lower + 1e-9, self.upper - 1e-9))
        best = None
        for x0 in starts:
            x0 = np.clip(x0, self.lower + 1e-9, self.upper - 1e-9)
            try:
                res = least_squares(lambda th: self._price(th, s, self.grid) - s.mid, x0,
                                    bounds=(self.lower, self.upper), method="trf",
                                    x_scale=np.array([atm**2, 1.0, 0.5, 0.05, 0.05]),
                                    diff_step=1e-4, max_nfev=200)
            except (ValueError, RuntimeError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            raise RuntimeError("Schrodinger calibration failed from every start")
        return best.x

    def price(self, params: np.ndarray, s: Slice) -> np.ndarray:
        return self._price(params, s, self.eval_grid)

    @staticmethod
    def with_level(params: np.ndarray, u: float) -> np.ndarray:
        v_min, b, rho, m, s = params
        return np.array([v_min * u * u, b * u * u, rho, m, s])  # sigma(x) -> u sigma(x)


class Heston:
    """Heston (1993) via QuantLib: AnalyticHestonEngine + Levenberg-Marquardt calibration."""

    name = "heston"
    n_params = 5
    _today = ql.Date(3, ql.January, 2022)
    _dc = ql.Actual365Fixed()

    def _market(self, s: Slice):
        ql.Settings.instance().evaluationDate = self._today
        q = s.r - np.log(s.F / s.S) / s.T  # dividend yield implied by the parity forward
        rts = ql.YieldTermStructureHandle(ql.FlatForward(self._today, s.r, self._dc))
        qts = ql.YieldTermStructureHandle(ql.FlatForward(self._today, float(q), self._dc))
        spot = ql.QuoteHandle(ql.SimpleQuote(s.S))
        return rts, qts, spot

    def calibrate(self, s: Slice, warm: np.ndarray | None = None) -> np.ndarray:
        rts, qts, spot = self._market(s)
        atm = s.atm_vol()
        x0 = warm if warm is not None else np.array([atm**2, 5.0, atm**2, 1.5, -0.7])
        v0, kappa, theta, sigma, rho = (float(v) for v in x0)
        model = ql.HestonModel(ql.HestonProcess(rts, qts, spot, v0, kappa, theta, sigma, rho))
        engine = ql.AnalyticHestonEngine(model)
        period = ql.Period(s.days, ql.Days)
        helpers = []
        for K, is_call, mid in zip(s.strikes, s.is_call, s.mid, strict=True):
            iv = _qpcore.black76_implied_vol(bool(is_call), s.F, float(K), s.T, float(mid), s.df)
            if not np.isfinite(iv):
                continue
            h = ql.HestonModelHelper(period, ql.NullCalendar(), s.S, float(K),
                                     ql.QuoteHandle(ql.SimpleQuote(iv)), rts, qts,
                                     ql.BlackCalibrationHelper.PriceError)
            h.setPricingEngine(engine)
            helpers.append(h)
        model.calibrate(helpers, ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8),
                        ql.EndCriteria(500, 100, 1e-8, 1e-8, 1e-8))
        return np.array(list(model.params()))  # theta, kappa, sigma, rho, v0 (QuantLib order)

    def price(self, params: np.ndarray, s: Slice) -> np.ndarray:
        rts, qts, spot = self._market(s)
        theta, kappa, sigma, rho, v0 = (float(p) for p in params)
        model = ql.HestonModel(ql.HestonProcess(rts, qts, spot, v0, kappa, theta, sigma, rho))
        engine = ql.AnalyticHestonEngine(model)
        maturity = self._today + s.days
        exercise = ql.EuropeanExercise(maturity)
        out = np.empty(len(s.strikes))
        for i, (K, is_call) in enumerate(zip(s.strikes, s.is_call, strict=True)):
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, float(K))
            opt = ql.VanillaOption(payoff, exercise)
            opt.setPricingEngine(engine)
            out[i] = opt.NPV()
        return out

    @staticmethod
    def with_level(params: np.ndarray, u: float) -> np.ndarray:
        theta, kappa, sigma, rho, v0 = params
        return np.array([theta, kappa, sigma, rho, v0 * u * u])  # instantaneous variance only

    @staticmethod
    def warm_from(params: np.ndarray) -> np.ndarray:
        """QuantLib parameter order -> constructor order (v0, kappa, theta, sigma, rho)."""
        theta, kappa, sigma, rho, v0 = params
        return np.array([v0, kappa, theta, sigma, rho])
