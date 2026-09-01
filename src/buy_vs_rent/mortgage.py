"""Vectorized fixed-rate mortgage math."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def monthly_payment(
    balance: FloatArray,
    annual_rate: FloatArray,
    remaining_months: NDArray[np.int_],
) -> FloatArray:
    """Return the contractual monthly principal-and-interest payment."""
    balance = np.asarray(balance, dtype=float)
    rate = np.asarray(annual_rate, dtype=float) / 12.0
    months = np.asarray(remaining_months, dtype=int)
    payment = np.zeros_like(balance)
    active = (balance > 0) & (months > 0)
    zero_rate = active & (np.abs(rate) < 1e-12)
    payment[zero_rate] = balance[zero_rate] / months[zero_rate]
    regular = active & ~zero_rate
    payment[regular] = (
        balance[regular]
        * rate[regular]
        / (1.0 - (1.0 + rate[regular]) ** (-months[regular]))
    )
    return payment


def amortize_year(
    balance: FloatArray,
    annual_rate: FloatArray,
    remaining_months: NDArray[np.int_],
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int_]]:
    """Apply up to twelve monthly payments and return payment, interest, new state."""
    balance = np.asarray(balance, dtype=float)
    rate = np.asarray(annual_rate, dtype=float) / 12.0
    months = np.asarray(remaining_months, dtype=int)
    count = np.minimum(months, 12)
    payment = monthly_payment(balance, annual_rate, months)
    total_payment = payment * count

    ending = balance.copy()
    active = (balance > 0) & (count > 0)
    zero_rate = active & (np.abs(rate) < 1e-12)
    ending[zero_rate] = balance[zero_rate] - total_payment[zero_rate]
    regular = active & ~zero_rate
    growth = (1.0 + rate[regular]) ** count[regular]
    ending[regular] = (
        balance[regular] * growth
        - payment[regular] * (growth - 1.0) / rate[regular]
    )
    ending = np.maximum(ending, 0.0)
    principal = balance - ending
    interest = np.maximum(total_payment - principal, 0.0)
    new_months = np.maximum(months - count, 0)
    return total_payment, interest, ending, new_months

