"""Turns a backtest into the tables and figures in ``results/``.

Only aggregate statistics and charts are written - never per-contract quotes - in line with the
data licence (permitted results, no redistribution of the underlying data).

Confidence intervals are percentile intervals from a day-level block bootstrap (quote dates are
resampled with replacement), because errors within a day are strongly correlated.

Usage::

    uv run qp-report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qpricing import _qpcore  # noqa: E402
from qpricing.data import PROCESSED_DIR, ROOT, load_dataset  # noqa: E402
from qpricing.models import Schrodinger  # noqa: E402
from qpricing.regimes import REGIMES, atm_vol_30d, classify  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
HOLDOUT_START = pd.Timestamp("2022-09-01")  # Jul-Aug = development window, Sep-Dec = holdout
SHORT_DATED_DAYS = 30

MODELS = ["schrodinger", "heston", "adhoc_bs", "black_scholes"]
LABELS = {
    "schrodinger": "Schrödinger (this work)",
    "heston": "Heston (QuantLib)",
    "adhoc_bs": "Ad-hoc Black-Scholes",
    "black_scholes": "Black-Scholes",
}
# Reference categorical palette (light mode), fixed order - validated adjacent pairs.
COLORS = {"schrodinger": "#2a78d6", "heston": "#eb6834", "adhoc_bs": "#1baf7a",
          "black_scholes": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PROTOCOLS = {
    "in": "In-sample fit (day t)",
    "out": "Next day, parameters frozen",
    "anchored": "Next day, ATM level re-marked",
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _daily_abs_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of absolute errors and quote count per quote date, per model."""
    err = pd.DataFrame({m: (df[m] - df.mid).abs() for m in MODELS})
    err["quote_date"] = df.quote_date.to_numpy()
    sums = err.groupby("quote_date")[MODELS].sum()
    sums["n"] = err.groupby("quote_date").size()
    return sums


def bootstrap_mae(df: pd.DataFrame, reps: int = 2000, seed: int = 7) -> dict:
    """MAE per model and the Schrödinger improvement vs each benchmark, with 95% CIs."""
    daily = _daily_abs_errors(df)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(reps, len(daily)))
    S = daily[MODELS].to_numpy()[idx].sum(axis=1)       # (reps, models)
    N = daily["n"].to_numpy()[idx].sum(axis=1)[:, None]
    boot = S / N
    point = daily[MODELS].sum().to_numpy() / daily["n"].sum()
    out = {"n_quotes": int(daily["n"].sum()), "n_days": len(daily)}
    for j, m in enumerate(MODELS):
        out[f"mae_{m}"] = point[j]
        out[f"mae_{m}_lo"], out[f"mae_{m}_hi"] = np.percentile(boot[:, j], [2.5, 97.5])
    for j, m in enumerate(MODELS[1:], start=1):
        imp = 1.0 - boot[:, 0] / boot[:, j]
        out[f"impr_vs_{m}"] = 1.0 - point[0] / point[j]
        out[f"impr_vs_{m}_lo"], out[f"impr_vs_{m}_hi"] = np.percentile(imp, [2.5, 97.5])
    return out


def summarise(bt: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    bt = bt.dropna(subset=MODELS)
    bt = bt.join(regimes["regime"], on="quote_date")
    bt["period"] = np.where(bt.calib_date >= HOLDOUT_START, "holdout", "development")
    rows = []
    short = bt[bt.days <= SHORT_DATED_DAYS]
    for protocol in PROTOCOLS:
        for period in ("development", "holdout", "full"):
            p = short[short["sample"] == protocol]
            if period != "full":
                p = p[p.period == period]
            for regime in ("all", *REGIMES):
                r = p if regime == "all" else p[p.regime == regime]
                if r.empty:
                    continue
                rows.append({"protocol": protocol, "period": period, "regime": regime,
                             **bootstrap_mae(r)})
    return pd.DataFrame(rows)


def by_maturity(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.dropna(subset=MODELS)
    bt = bt[(bt["sample"] == "anchored") & (bt.calib_date >= HOLDOUT_START)]
    buckets = pd.cut(bt.days, [0, 7, 14, 30, 45], labels=["2-7d", "8-14d", "15-30d", "31-45d"])
    rows = []
    for b, g in bt.groupby(buckets, observed=True):
        rows.append({"maturity": str(b), **bootstrap_mae(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(row, m):
    return (f"**{100 * row[f'impr_vs_{m}']:+.1f}%** "
            f"[{100 * row[f'impr_vs_{m}_lo']:+.1f}, {100 * row[f'impr_vs_{m}_hi']:+.1f}]")


def markdown(summary: pd.DataFrame, maturity: pd.DataFrame, regimes: pd.DataFrame) -> str:
    q1, q2 = regimes.attrs["cuts"]
    lines = [
        "# Backtest results",
        "",
        "Generated by `uv run qp-report`. SPX/SPXW out-of-the-money options with at most "
        f"{SHORT_DATED_DAYS} calendar days to expiry. Errors are mean absolute differences to "
        "the bid-ask mid, in index points. Improvement = 1 - MAE(Schrödinger) / MAE(benchmark); "
        "a positive number means the Schrödinger model is more accurate. Brackets are 95% "
        "day-block bootstrap intervals.",
        "",
        f"Regimes are terciles of the 30-day ATM implied vol computed from the data: "
        f"low < {100 * q1:.1f}% <= medium < {100 * q2:.1f}% <= high.",
        "",
    ]
    for protocol, title in PROTOCOLS.items():
        lines += [f"## {title}", ""]
        lines += ["| period | regime | quotes | Schrödinger | Heston | Ad-hoc BS | Black-Scholes "
                  "| vs Heston | vs Ad-hoc BS | vs Black-Scholes |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in summary[summary.protocol == protocol].iterrows():
            lines.append(
                f"| {r.period} | {r.regime} | {r.n_quotes:,} | {r.mae_schrodinger:.3f} | "
                f"{r.mae_heston:.3f} | {r.mae_adhoc_bs:.3f} | {r.mae_black_scholes:.3f} | "
                f"{_pct(r, 'heston')} | {_pct(r, 'adhoc_bs')} | {_pct(r, 'black_scholes')} |")
        lines.append("")
    lines += ["## Holdout, ATM re-marked, by maturity", "",
              "| maturity | quotes | Schrödinger | Heston | Ad-hoc BS | Black-Scholes | vs Heston "
              "| vs Ad-hoc BS | vs Black-Scholes |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in maturity.iterrows():
        lines.append(
            f"| {r.maturity} | {r.n_quotes:,} | {r.mae_schrodinger:.3f} | {r.mae_heston:.3f} | "
            f"{r.mae_adhoc_bs:.3f} | {r.mae_black_scholes:.3f} | {_pct(r, 'heston')} | "
            f"{_pct(r, 'adhoc_bs')} | {_pct(r, 'black_scholes')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_mae_by_regime(summary: pd.DataFrame, path: Path) -> None:
    s = summary[(summary.protocol == "anchored") & (summary.period == "holdout")
                & (summary.regime.isin(REGIMES))].set_index("regime").loc[list(REGIMES)]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    width = 0.2
    x = np.arange(len(REGIMES))
    for j, m in enumerate(MODELS):
        vals = s[f"mae_{m}"].to_numpy()
        err = np.vstack([vals - s[f"mae_{m}_lo"], s[f"mae_{m}_hi"] - vals])
        ax.bar(x + (j - 1.5) * width, vals, width - 0.02, color=COLORS[m], label=LABELS[m],
               yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, [f"{r} vol" for r in REGIMES], color=INK)
    ax.set_ylabel("Mean absolute error (index points)", color=INK_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Next-day pricing error, short-dated OTM SPX options (holdout Sep-Dec 2022)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_regimes(regimes: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=SURFACE)
    _style(ax)
    lv = regimes.atm_vol_30d * 100
    ax.plot(lv.index, lv.to_numpy(), color=INK, linewidth=1.6)
    q1, q2 = (100 * c for c in regimes.attrs["cuts"])
    for y, lab in ((q1, "low | medium"), (q2, "medium | high")):
        ax.axhline(y, color=INK_2, linewidth=0.8, linestyle="--")
        ax.text(lv.index[-1], y + 0.25, lab, color=INK_2, fontsize=8, ha="right")
    ax.axvline(HOLDOUT_START, color=COLORS["schrodinger"], linewidth=1)
    ax.text(HOLDOUT_START, lv.max(), "  holdout begins", color=INK_2, fontsize=8, va="top")
    ax.set_ylabel("30-day ATM implied vol (%)", color=INK_2, fontsize=9)
    ax.set_title("Volatility regimes, SPX 2022-H2 (terciles of 30-day ATM vol)", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_potential_wells(params: pd.DataFrame, regimes: pd.DataFrame, path: Path) -> None:
    """The calibrated quantum potential and its lowest bound states, one slice per regime."""
    finite = params["params"].map(lambda v: bool(np.all(np.isfinite(v))))
    p = params[(params.model == "schrodinger") & finite]
    p = p.join(regimes["regime"], on="calib_date")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=SURFACE, sharey=False)
    for ax, regime in zip(axes, REGIMES, strict=True):
        _style(ax)
        ax.grid(False)
        cand = p[(p.regime == regime) & p.days.between(8, 10)]
        # the slice whose ATM vol level is the median of its regime: a typical well
        lvl = cand["params"].map(lambda v: v[0])
        row = cand.iloc[int(np.argsort(lvl.to_numpy())[len(lvl) // 2])]
        T = row.days / 365
        sol = _qpcore.solve(Schrodinger.to_params(np.array(row["params"])), T,
                            _qpcore.GridSpec(1200, 200), 4)
        y, V = np.asarray(sol["y"]), np.asarray(sol["potential"])
        E = np.asarray(sol["energies"])
        # Plot where the particle actually lives: +-4 sd of the unit-diffusion Y-process.
        centre = y[np.argmax(np.abs(np.asarray(sol["states"][0])))]
        window = np.abs(y - centre) < 4.0 * np.sqrt(T)
        top = 2.2 * E[2]
        bottom = -0.6 * top
        ax.plot(y[window], np.clip(V[window], bottom, None), color=INK, linewidth=1.6)
        scale = 0.35 * (E[1] - E[0])
        for n in range(3):
            phi = np.asarray(sol["states"][n])
            ax.axhline(E[n], color=GRID, linewidth=0.8)
            ax.plot(y[window], E[n] + scale * phi[window] / np.abs(phi[window]).max(),
                    color=COLORS["schrodinger"], linewidth=1.4, alpha=1.0 - 0.25 * n)
            ax.text(y[window][-1], E[n], f" E{n}", color=INK_2, fontsize=8, va="center")
        ax.set_ylim(bottom, top)
        i_min = int(np.argmin(np.where(window, V, np.inf)))
        if V[i_min] < bottom:
            ax.annotate(f"attractive spike\n(depth {V[i_min]:.0f})", xy=(y[i_min], bottom),
                        xytext=(y[i_min] + 0.15 * (y[window][-1] - y[window][0]), 0.55 * bottom),
                        color=INK_2, fontsize=7,
                        arrowprops={"arrowstyle": "->", "color": INK_2, "linewidth": 0.8})
        ax.set_title(f"{regime} vol: {row.calib_date.date()}, {row.days}d", color=INK, fontsize=9)
        ax.set_xlabel("y (Lamperti coordinate)", color=INK_2, fontsize=8)
    axes[0].set_ylabel("V(y) and bound states", color=INK_2, fontsize=9)
    fig.suptitle("Calibrated potential wells V(y) = (b² + b')/2 with their lowest eigenstates",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_smile(bt: pd.DataFrame, path: Path) -> None:
    """Implied-vol smiles of one holdout slice: market vs next-day model predictions."""
    a = bt[(bt["sample"] == "anchored") & (bt.calib_date >= HOLDOUT_START)
           & bt.days.between(6, 9)].dropna(subset=MODELS)
    sizes = a.groupby(["calib_date", "root", "expiration"]).size()
    key = sizes[sizes >= 80].index[len(sizes[sizes >= 80]) // 2]
    g = a.set_index(["calib_date", "root", "expiration"]).loc[key].sort_values("k")
    T = g.days.iloc[0] / 365

    def iv(prices):
        return np.array([_qpcore.black76_implied_vol(bool(c), float(F), float(K), T, float(p), 1.0)
                         for c, F, K, p in zip(g.is_call, g.F, g.strike, prices, strict=True)])

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.scatter(g.k, 100 * iv(g.mid), s=14, color=INK_2, label="Market (mid)", zorder=3)
    for m in MODELS:
        ax.plot(g.k, 100 * iv(g[m]), color=COLORS[m], linewidth=2, label=LABELS[m])
    ax.set_xlabel("log-moneyness ln(K/F)", color=INK_2, fontsize=9)
    ax.set_ylabel("Implied volatility (%)", color=INK_2, fontsize=9)
    ax.set_title(f"Next-day smile prediction: {g.quote_date.iloc[0].date()}, {key[1]} "
                 f"{key[2].date()} ({g.days.iloc[0]:.0f}d), calibrated {key[0].date()}",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backtest", type=Path, default=PROCESSED_DIR / "backtest.parquet")
    args = ap.parse_args(argv)

    bt = pd.read_parquet(args.backtest)
    params = pd.read_parquet(args.backtest.with_name(args.backtest.stem + "_params.parquet"))
    regimes = classify(atm_vol_30d(load_dataset()))

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    out_regimes = regimes.copy()
    out_regimes.index.name = "quote_date"
    out_regimes.to_csv(RESULTS / "regimes.csv", float_format="%.5f")

    summary = summarise(bt, regimes)
    maturity = by_maturity(bt)
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.5f")
    maturity.to_csv(RESULTS / "by_maturity.csv", index=False, float_format="%.5f")
    (RESULTS / "RESULTS.md").write_text(markdown(summary, maturity, regimes))

    timing = params.groupby("model").seconds.median().rename("median_calibration_seconds")
    timing.to_csv(RESULTS / "calibration_time.csv", float_format="%.4f")

    fig_mae_by_regime(summary, FIGURES / "mae_by_regime.png")
    fig_regimes(regimes, FIGURES / "regimes.png")
    fig_potential_wells(params, regimes, FIGURES / "potential_wells.png")
    fig_smile(bt, FIGURES / "smile_prediction.png")
    print((RESULTS / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
