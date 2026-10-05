"""Option pricing with an imaginary-time Schrödinger solver.

The heavy lifting (potential construction, Crank-Nicolson propagation, payoff integration and the
spectral decomposition) lives in the C++ core, exposed here as :mod:`qpricing._qpcore`.
"""

from qpricing._qpcore import (
    EffectiveVolParams,
    GridSpec,
    black76_implied_vol,
    black76_price,
    schrodinger_price,
    solve,
)

__all__ = [
    "EffectiveVolParams",
    "GridSpec",
    "black76_implied_vol",
    "black76_price",
    "schrodinger_price",
    "solve",
]
__version__ = "0.1.0"
